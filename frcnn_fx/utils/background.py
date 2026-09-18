# faster_rcnn_bg.py
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

# --- Helper function to split tensors based on proposals per image ---
def split_predictions(predictions, proposals):
    """Split concatenated predictions, or pass through an already per-image tuple.

    Detectron2 `predict_boxes` / `predict_probs` already return a tuple of
    per-image tensors. Treating that tuple as "class-specific, take [0]" and
    splitting again breaks any batch size > 1 (split_sizes sum to 2N vs N).
    """
    sizes = [len(p) for p in proposals]
    if isinstance(predictions, (list, tuple)):
        if len(predictions) == len(proposals):
            return list(predictions)
        if len(predictions) == 1:
            predictions = predictions[0]
        else:
            raise ValueError(
                f"cannot split prediction collection of length {len(predictions)} "
                f"across {len(proposals)} images"
            )
    return list(predictions.split(sizes, dim=0))

# --------------------------------------------------------------------

def fast_rcnn_inference_with_bg(
    boxes, scores, box_features, image_shapes, score_thresh, nms_thresh, topk_per_image, logits=None
):
    """
    Modified version of fast_rcnn_inference that preserves background scores
    and returns indices to select corresponding features.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted boxes for each image.
            Element i has shape (Ri, K*4) or (Ri, 1*4) for class-agnostic regression.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1).
        box_features (list[Tensor]): A list of Tensors of features for each proposal box.
             Element i has shape (Ri, feature_dim).
        image_shapes (list[tuple]): A list of (width, height) for each image.
        score_thresh (float): Threshold for filtering detections based on score.
        nms_thresh (float): Non-maximum suppression threshold.
        topk_per_image (int): Maximum number of detections to return per image.

    Returns:
        list[Instances]: A list of N Instances, one for each image. The
            Instance objects contain fields "pred_boxes", "scores",
            "pred_classes", "bg_scores", "pred_features", and "pred_logits".
    """
    results_per_image = []

    for i, (features_per_image, scores_per_image, boxes_per_image, image_shape) in enumerate(zip(
        box_features, scores, boxes, image_shapes
    )):
        logits_per_image = logits[i] if logits is not None else None
        result_single = fast_rcnn_inference_single_image_with_bg(
            boxes_per_image,
            scores_per_image,
            features_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
            logits_per_image,
        )
        results_per_image.append(result_single)

    return results_per_image

def fast_rcnn_inference_single_image_with_bg(
    boxes, scores, features, image_shape, score_thresh, nms_thresh, topk_per_image, logits=None
):
    """
    Single-image inference with background scores preserved and features added.

    Args:
        boxes (Tensor or tuple): Boxes for one image, shape (R, K*4) or (R, 4).
            If tuple, the first element is used.
        scores (Tensor or tuple): Scores for one image, shape (R, K + 1).
        features (Tensor): Features for one image, shape (R, feat_dim).
        image_shape (tuple): (height, width) for the image.
        score_thresh (float): Threshold on score.
        nms_thresh (float): Non-maximum suppression threshold.
        topk_per_image (int): Max detections to keep.

    Returns:
        Instances: Resulting instances with fields "pred_boxes", "scores",
                   "pred_classes", "bg_scores", "pred_features", and "pred_logits".
    """
    # Handle cases where boxes is a tuple (first element contains the boxes)
    if isinstance(boxes, tuple):
        boxes = boxes[0]
        
    # Handle cases where scores is a tuple (first element contains the scores)
    if isinstance(scores, tuple):
        scores = scores[0]
        
    # Ensure boxes and scores are tensors before checking isfinite
    if not isinstance(boxes, torch.Tensor):
        print(f"Warning: boxes is not a tensor, type: {type(boxes)}")
        if len(boxes) == 0:
            # Handle empty case
            result = Instances(image_shape)
            result.pred_boxes = Boxes(torch.zeros(0, 4, device=features.device))
            result.scores = torch.zeros(0, device=features.device)
            result.pred_classes = torch.zeros(0, dtype=torch.int64, device=features.device)
            result.bg_scores = torch.zeros(0, device=features.device)
            result.pred_features = torch.zeros((0, features.shape[1]), device=features.device)
            result.pred_logits = torch.zeros((0, scores.shape[1]), device=features.device)
            return result
        else:
            # Convert to tensor if possible
            boxes = torch.tensor(boxes, device=features.device)
    
    if not isinstance(scores, torch.Tensor):
        print(f"Warning: scores is not a tensor, type: {type(scores)}")
        if len(scores) == 0:
            # Handle empty case
            result = Instances(image_shape)
            result.pred_boxes = Boxes(torch.zeros(0, 4, device=features.device))
            result.scores = torch.zeros(0, device=features.device)
            result.pred_classes = torch.zeros(0, dtype=torch.int64, device=features.device)
            result.bg_scores = torch.zeros(0, device=features.device)
            result.pred_features = torch.zeros((0, features.shape[1]), device=features.device)
            result.pred_logits = torch.zeros((0, scores.shape[1]), device=features.device)
            return result
        else:
            # Convert to tensor if possible
            scores = torch.tensor(scores, device=features.device)
        
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        features = features[valid_mask]

    # Store proposal indices for tracking
    original_indices = torch.arange(scores.shape[0], device=scores.device)

    # Handle empty inputs
    if scores.shape[0] == 0:
        result = Instances(image_shape)
        result.pred_boxes = Boxes(torch.zeros(0, 4, device=scores.device))
        result.scores = torch.zeros(0, device=scores.device)
        result.pred_classes = torch.zeros(0, dtype=torch.int64, device=scores.device)
        result.bg_scores = torch.zeros(0, device=scores.device)
        result.pred_features = torch.zeros((0, features.shape[1]), device=scores.device)
        result.pred_logits = torch.zeros((0, scores.shape[1]), device=scores.device)
        return result

    # Save background scores
    bg_scores = scores[:, -1]

    # Process foreground scores
    scores_fg = scores[:, :-1]  # R x K
    num_bbox_reg_classes = boxes.shape[1] // 4
    
    # Filter results based on detection scores
    filter_mask = scores_fg > score_thresh  # R x K
    
    # Handle class-agnostic vs. class-specific cases consistently
    if num_bbox_reg_classes == 1:
        # Class-agnostic case (R, 4) boxes
        # Get the max foreground score and corresponding class for each proposal
        max_scores, max_classes = scores_fg.max(dim=1)
        
        # Only keep proposals with score > threshold
        keep_indices = (max_scores > score_thresh).nonzero().squeeze(1)
        
        if keep_indices.numel() == 0:
            # No proposals above threshold
            result = Instances(image_shape)
            result.pred_boxes = Boxes(torch.zeros(0, 4, device=scores.device))
            result.scores = torch.zeros(0, device=scores.device)
            result.pred_classes = torch.zeros(0, dtype=torch.int64, device=scores.device)
            result.bg_scores = torch.zeros(0, device=scores.device)
            result.pred_features = torch.zeros((0, features.shape[1]), device=scores.device)
            result.pred_logits = torch.zeros((0, scores.shape[1]), device=scores.device)
            return result
        
        # Get boxes for all kept proposals (same box for all classes in class-agnostic case)
        boxes_filtered = boxes[keep_indices]  # (num_kept, 4)
        scores_filtered = max_scores[keep_indices]  # (num_kept,)
        class_inds_filtered = max_classes[keep_indices]  # (num_kept,)
        original_indices_filtered = original_indices[keep_indices]  # (num_kept,)
        features_filtered = features[keep_indices]  # (num_kept, feat_dim)
        bg_scores_filtered = bg_scores[keep_indices]  # (num_kept,)
        
    else:
        # Class-specific case (R, K*4) boxes
        filter_inds = filter_mask.nonzero()  # (num_filtered, 2)
        if filter_inds.numel() == 0:
            # No proposals above threshold
            result = Instances(image_shape)
            result.pred_boxes = Boxes(torch.zeros(0, 4, device=scores.device))
            result.scores = torch.zeros(0, device=scores.device)
            result.pred_classes = torch.zeros(0, dtype=torch.int64, device=scores.device)
            result.bg_scores = torch.zeros(0, device=scores.device)
            result.pred_features = torch.zeros((0, features.shape[1]), device=scores.device)
            result.pred_logits = torch.zeros((0, scores.shape[1]), device=scores.device)
            return result
            
        # Convert boxes from shape (R, K*4) to (R*K, 4)
        boxes_reshaped = boxes.view(-1, num_bbox_reg_classes, 4)
        
        # Select boxes for specific classes
        boxes_filtered = boxes_reshaped[filter_inds[:, 0], filter_inds[:, 1]]  # Select boxes for specific classes
        scores_filtered = scores_fg[filter_mask]  # Select scores that passed threshold
        class_inds_filtered = filter_inds[:, 1]  # Class indices for filtered entries
        original_indices_filtered = original_indices[filter_inds[:, 0]]  # Map back to original proposals
        features_filtered = features[filter_inds[:, 0]]  # Features for filtered proposals
        bg_scores_filtered = bg_scores[filter_inds[:, 0]]  # Background scores for filtered proposals

    # Clip boxes to image boundaries
    boxes_filtered_obj = Boxes(boxes_filtered)
    boxes_filtered_obj.clip(image_shape)
    boxes_filtered = boxes_filtered_obj.tensor

    # Apply NMS
    keep = batched_nms(
        boxes_filtered, 
        scores_filtered, 
        class_inds_filtered, 
        nms_thresh
    )
    
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]

    # Get original logits for the final detections
    final_original_indices = original_indices_filtered[keep]
    
    # Use passed logits if available, otherwise use scores
    if logits is not None:
        final_logits = logits[final_original_indices]
    else:
        final_logits = scores[final_original_indices]

    # Build result
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes_filtered[keep])
    result.scores = scores_filtered[keep]
    result.pred_classes = class_inds_filtered[keep]
    result.bg_scores = bg_scores_filtered[keep]
    result.pred_features = features_filtered[keep]
    result.pred_logits = final_logits  # Add raw logits

    return result
