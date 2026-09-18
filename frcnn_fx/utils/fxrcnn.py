# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

# --- Import the custom inference functions and helper ---
from utils.background import fast_rcnn_inference_with_bg, split_predictions
from detectron2.structures import Boxes
# ---

__all__ = ["FXGeneralizedRCNN", "FXProposalNetwork"]

# Implement our own move_device_like helper function
def move_device_like(src, dst):
    """
    Moves a tensor to the same device as another tensor.
    """
    return src.to(dst.device)


@META_ARCH_REGISTRY.register()
class FXGeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN modified to extract features and background scores.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        # Add parameters needed by the custom inference function explicitly
        test_score_thresh: float = 0.05,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
    ):
        """
        Args:
            backbone: Backbone module
            proposal_generator: Proposal generator module
            roi_heads: ROI heads module
            pixel_mean: Pixel mean for normalization
            pixel_std: Pixel std for normalization
            input_format: Image format (e.g., BGR or RGB)
            vis_period: Visualization period during training
            test_score_thresh: Threshold for custom inference
            test_nms_thresh: NMS threshold for custom inference
            test_topk_per_image: Top-K threshold for custom inference
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # Store inference parameters
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        roi_heads = build_roi_heads(cfg, backbone.output_shape())
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": roi_heads,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # Pass inference parameters from config
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # Pass features explicitly if needed by ROIHeads training logic
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    @torch.no_grad()
    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]],
                  detected_instances: Optional[List[Instances]] = None,
                  do_postprocess: bool = True):
        """
        Run inference, extracting features and background scores.

        Args:
            batched_inputs (list[dict]): Input data.
            detected_instances (None or list[Instances]): Optional pre-computed detections.
            do_postprocess (bool): Whether to apply post-processing (rescaling).

        Returns:
            list[dict]: Each dict contains an "instances" key with final Instance objects
                        (including "pred_features" and "bg_scores").
        """
        assert not self.training
        
        # If pre-computed detection is provided, skip the forward process
        if detected_instances is not None:
            processed_results = []
            for i, input in enumerate(batched_inputs):
                height = input.get("height", detected_instances[i].image_size[0])
                width = input.get("width", detected_instances[i].image_size[1])
                
                if do_postprocess:
                    r = detector_postprocess(detected_instances[i], height, width)
                else:
                    r = detected_instances[i]
                processed_results.append({"instances": r})
            return processed_results

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            
        # Handle empty proposals case
        if len(proposals) == 0 or all(len(p) == 0 for p in proposals):
            # Return empty instances for each input
            processed_results = []
            for input_i, image_size_i in zip(batched_inputs, images.image_sizes):
                height = input_i.get("height", image_size_i[0])
                width = input_i.get("width", image_size_i[1])
                r = Instances((height, width))
                r.pred_boxes = Boxes(torch.zeros(0, 4, device=self.device))
                r.scores = torch.zeros(0, device=self.device)
                r.pred_classes = torch.zeros(0, dtype=torch.int64, device=self.device)
                r.bg_scores = torch.zeros(0, device=self.device)
                r.pred_features = torch.zeros((0, self.roi_heads.box_head.output_shape), device=self.device)
                # Add empty logits field
                r.pred_logits = torch.zeros((0, self.roi_heads.box_predictor.num_classes + 1), device=self.device)
                processed_results.append({"instances": r})
            return processed_results

        # Extract features for proposals using ROI pooler
        features_list = [features[f] for f in self.roi_heads.box_in_features]
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features_pooled = self.roi_heads.box_pooler(features_list, proposal_boxes)

        # Pass through box head to get FC features
        box_features = self.roi_heads.box_head(box_features_pooled)

        # Pass through box predictor (yields classification scores and box deltas)
        predictions = self.roi_heads.box_predictor(box_features)
        pred_logits_raw = predictions[0] # Extract raw logits

        # Get raw scores (including background) and predicted boxes (absolute coordinates)
        pred_scores_raw = self.roi_heads.box_predictor.predict_probs(predictions, proposals)
        pred_boxes_raw = self.roi_heads.box_predictor.predict_boxes(predictions, proposals)

        # Split the batch results into per-image lists
        list_pred_boxes_raw = split_predictions(pred_boxes_raw, proposals)
        list_pred_scores_raw = split_predictions(pred_scores_raw, proposals)
        list_box_features = split_predictions(box_features, proposals)
        list_pred_logits_raw = split_predictions(pred_logits_raw, proposals) # Split logits

        image_shapes = [x.image_size for x in proposals]

        # Call the custom inference function
        results_per_image = fast_rcnn_inference_with_bg(
            list_pred_boxes_raw,
            list_pred_scores_raw,
            list_box_features,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            list_pred_logits_raw,  # Pass logits
        )

        # Post-process (rescale boxes to original image dimensions)
        if do_postprocess:
            processed_results = []
            for results_i, input_i, image_size_i in zip(results_per_image, batched_inputs, images.image_sizes):
                height = input_i.get("height", image_size_i[0])
                width = input_i.get("width", image_size_i[1])
                r = detector_postprocess(results_i, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            # Return results without rescaling, wrapped in dicts
            return [{"instances": r} for r in results_per_image]

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
        )
        return images

# Missing FXProposalNetwork implementation that was in __all__ but not defined
@META_ARCH_REGISTRY.register()
class FXProposalNetwork(FXGeneralizedRCNN):
    """
    FX version of the ProposalNetwork, same relationship as in original Detectron2.
    Only outputs object proposals, not final detections.
    """
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        
        return proposals, proposal_losses

    @torch.no_grad()
    def inference(self, batched_inputs):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in `forward()`

        Returns:
            list[Instances]: same as `forward()`.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results