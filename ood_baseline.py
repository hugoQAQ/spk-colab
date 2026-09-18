#!/usr/bin/env python3
"""Detector-independent, offline Table-1 OOD baselines (NumPy / sklearn).

Install: pip install numpy scipy scikit-learn
Example:
  python ood_logits_baselines.py --train train.npz --id-val val.npz \
    --near-ood near.npz --far-ood far.npz --out results \
    --bam-density 50 --outlier mhood-iqr

Input .npy: raw PRE-softmax/PRE-sigmoid logits [N,C], one final detection per row.
Input .npz: logits [N,C]; optional pred_labels [N] (integer ORIGINAL logit-column
indices), detection_ids [N] (unique strings), class_names [C] (ordered strings).
All splits must have identical class order and detector checkpoint. Detection
labels should be supplied: final detections need not have the argmax class.
If absent, foreground argmax labels and split-local row IDs are generated.
No GT matching, TP filtering, NMS, confidence filtering, or image subsampling is
performed. Supply exactly the desired training pool and ALL evaluation detections.
Separate detector/dataset configurations must be evaluated in separate runs.

Eight implementations:
  MSP: max softmax; background remains in denominator if supplied.
  EBO: T*logsumexp(z/T), i.e. NEGATIVE energy (higher = more ID).
  MLS: maximum foreground logit.
  SCALE-logits: m-hood's logit-space scaling, NOT original feature-space SCALE.
  MDS-logits: nearest class mean with shared empirical covariance, argmax
      pseudo-labels by default; no fictitious zero mean for absent classes.
  BAM-logits: class-conditional clustered boxes; negative min L1 outside distance.
  KNN-logits: m-hood default = train-standardized features, mean Euclidean kNN
      distance, negated. --knn-mode sun = L2 normalization + kth squared distance.
  iForest-logits: added implementation, global or class-conditional (default).

--background-index is explicit: no guessing from dimension. Default MSP keeps
background in normalization but maximizes foreground; --msp-denominator foreground
renormalizes foreground only. Other scores and fitted models exclude background.
--msp-activation sigmoid computes an explicitly named MaxSigmoid variant; EBO
remains categorical logsumexp, not an independent-Bernoulli energy.
SCALE's signed-logit sums can be zero or unstable. Default: record method failure,
continue others, exit status 2. --scale-degenerate identity uses unchanged logits
ONLY on invalid rows and labels the method as a guarded variant; counts are saved.

Output: per-method cached scores (.npz), report.json, summary.csv, fitted models
(.joblib), selected train row indices, optional outlier mask. Cache uses SHA256
of inputs, script, and fitting parameters. Evaluation-only changes reuse scores.
Models are for trusted local use only (joblib loading is pickle-based).
No test data fit scalers/models. Optional mhood-iqr removal is explicitly a
TRANSDUCTIVE diagnostic: standardize ID-val, query ID-val against itself (including
self), remove BOTH IQR tails; shared mask for every method. score-tail instead
removes each method's lowest scores, optionally by predicted class. Full-ID
metrics are always emitted. FPR95 uses a deterministic empirical threshold
achieving >=95% retention, including ties; no percentile interpolation. This may
differ from old np.percentile code. Percent metrics range 0..100. Remaining OOD
counts use the same threshold as FPR95. No counts are inferred from rounded FPR.

Reference snapshot: https://github.com/hugoQAQ/m-hood
commit a8903441d6569e875189f96f63f829fa6c1548dc, archive m-hood-main.zip.
Read: src/utils/ood_methods.py, ood_evaluation.py, outlier_removal.py,
      src/utils/bam/{construction,evaluation,box}.py.
The snapshot contains unresolved merge conflicts. This is an independent clean
implementation of its formulas, NOT a byte-for-byte reproduction. BAM uses a
specified deterministic KMeans/MiniBatchKMeans fit (not the conflicted one-pass
partial_fit branch). Its default box cap is an explicit speed tradeoff.
Original SCALE: https://github.com/kai422/SCALE/blob/main/openood/networks/scale_net.py
MDS: https://arxiv.org/abs/1807.03888 ; KNN: https://arxiv.org/abs/2204.06507
BAM: https://arxiv.org/abs/2403.18373 ; EBO: https://arxiv.org/abs/2010.03759
MSP: https://arxiv.org/abs/1610.02136
Use logits-space labels in a paper: these are not raw ROI-feature baselines.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import scipy
from scipy.special import expit, logsumexp
import sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

METHODS = ('MSP', 'EBO', 'MLS', 'SCALE', 'MDS', 'BAM', 'KNN', 'iForest')


def matrix(x, name):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError(f'{name}: require finite [N,C] matrix with C>=1')
    return x


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_split(path, name, background=None):
    raw = np.load(path, allow_pickle=False)
    if isinstance(raw, np.lib.npyio.NpzFile):
        with raw:
            z = matrix(raw['logits'], name)
            labels = np.array(raw['pred_labels']) if 'pred_labels' in raw else None
            ids = np.array(raw['detection_ids']) if 'detection_ids' in raw else None
            names = np.array(raw['class_names']).astype(str) if 'class_names' in raw else None
    else:
        z, labels, ids, names = matrix(raw, name), None, None, None
    c = z.shape[1]
    if background is not None and not 0 <= background < c:
        raise ValueError('background-index must be an explicit nonnegative column index')
    fg = np.array([i for i in range(c) if i != background], dtype=int)
    if not len(fg):
        raise ValueError('No foreground columns')
    inferred = labels is None
    if inferred:
        labels = fg[np.argmax(z[:, fg], axis=1)]
    if labels.shape != (len(z),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'{name}: pred_labels must be integer [N] column indices')
    if not np.isin(labels, fg).all():
        raise ValueError(f'{name}: detection label is background or outside foreground columns')
    generated = ids is None
    ids = np.array([f'{name}:{i}' for i in range(len(z))]) if generated else ids.astype(str)
    if ids.shape != (len(z),) or len(np.unique(ids)) != len(ids):
        raise ValueError(f'{name}: detection_ids must be unique [N]')
    if names is not None and (names.shape != (c,) or len(np.unique(names)) != c):
        raise ValueError(f'{name}: class_names must be unique [C]')
    return dict(logits=z, x=z[:, fg], labels=labels.astype(int), ids=ids, names=names,
                fg=fg, inferred_labels=inferred, generated_ids=generated)


def scale_logits(z, percentile=65., on_invalid='error'):
    """m-hood formula: z * exp(sum(z)/sum(top-k(z))); no ReLU/abs/clipping."""
    z = matrix(z, 'SCALE')
    k = z.shape[1] - int(np.round(z.shape[1] * percentile / 100.))
    if not 1 <= k <= z.shape[1]:
        raise ValueError(f'SCALE retains k={k} columns; lower --scale-percentile')
    top = np.partition(z, z.shape[1] - k, axis=1)[:, -k:]
    denom = top.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ratio = z.sum(axis=1) / denom
        scaled = z * np.exp(ratio[:, None])
    bad = (np.abs(denom) <= 1e-12) | ~np.isfinite(scaled).all(axis=1)
    if bad.any():
        if on_invalid == 'error':
            raise ValueError(f'SCALE undefined/unstable in {int(bad.sum())} rows; '
                             'inspect signed logits or explicitly choose --scale-degenerate identity')
        scaled[bad] = z[bad]
    return scaled, int(bad.sum())


def train_indices(y, cap, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for c in np.unique(y):
        ix = np.flatnonzero(y == c)
        if cap and len(ix) > cap:
            ix = rng.choice(ix, cap, replace=False)
        parts.append(ix)
    return np.sort(np.concatenate(parts))


def fit_model(method, x, labels, a):
    if method == 'MDS':
        y = x.argmax(axis=1) if a.mds_labels == 'argmax' else labels
        classes = np.unique(y)
        means = np.stack([x[y == c].mean(axis=0) for c in classes])
        residuals = np.concatenate([x[y == c] - mu for c, mu in zip(classes, means)])
        if len(x) <= len(classes):
            raise ValueError('MDS needs within-class variation, not one row per class')
        cov = (EmpiricalCovariance() if a.covariance == 'empirical' else LedoitWolf()).fit(residuals)
        if not np.any(np.diag(cov.covariance_) > 0):
            raise ValueError('MDS covariance is zero')
        return dict(means=means, precision=cov.precision_)
    if method == 'KNN':
        scaler = StandardScaler().fit(x) if a.knn_mode == 'mhood' else None
        bank = scaler.transform(x) if scaler else normalize(x)
        k = min(a.knn_k, len(bank))
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=a.jobs).fit(bank)
        return dict(scaler=scaler, nn=nn, k=k)
    if method == 'BAM':
        boxes = {}
        for c in np.unique(labels):
            xc = x[labels == c]
            k = min(len(xc), max(1, round(len(xc) / a.bam_density)))
            if a.bam_max_boxes:
                k = min(k, a.bam_max_boxes)
            if k == 1:
                cl = np.zeros(len(xc), dtype=int)
            else:
                kw = dict(n_clusters=k, random_state=a.seed, n_init=10, max_iter=300)
                km = (MiniBatchKMeans(batch_size=max(1024, k), **kw)
                      if a.bam_cluster == 'minibatch' else KMeans(**kw))
                cl = km.fit_predict(xc)
            lo, hi = [], []
            for j in np.unique(cl):
                v = xc[cl == j]
                lo.append(v.min(axis=0)); hi.append(v.max(axis=0))
            boxes[int(c)] = (np.stack(lo), np.stack(hi))
        return boxes
    if method == 'iForest':
        models = {}
        for c in np.unique(labels) if a.iforest_scope == 'classwise' else [-1]:
            xc = x[labels == c] if c != -1 else x
            if len(xc) < 2:
                raise ValueError(f'iForest class {c} has fewer than two training rows')
            sc = StandardScaler().fit(xc)
            f = IsolationForest(n_estimators=a.trees, max_samples=min(a.iforest_samples, len(xc)),
                                contamination='auto', random_state=a.seed, n_jobs=a.jobs)
            f.fit(sc.transform(xc))
            models[int(c)] = (sc, f)
        return models
    return None


def box_score(x, lo, hi, box_batch=64):
    """Bounded-memory min L1 distance to union of boxes, negated."""
    best = np.full(len(x), np.inf)
    for start in range(0, len(lo), box_batch):
        low, high = lo[start:start+box_batch], hi[start:start+box_batch]
        d = np.maximum(low[None] - x[:, None], 0)
        d += np.maximum(x[:, None] - high[None], 0)
        best = np.minimum(best, d.sum(axis=2).min(axis=1))
    return -best


def score_batch(method, model, data, a):
    z, x, labels = data['logits'], data['x'], data['labels']
    meta = {}
    if method == 'MSP':
        if a.msp_activation == 'sigmoid':
            return expit(x).max(axis=1), meta
        denom = z if a.msp_denominator == 'all' else x
        return np.exp(x.max(axis=1) - logsumexp(denom, axis=1)), meta
    if method == 'EBO':
        return a.temperature * logsumexp(x / a.temperature, axis=1), meta
    if method == 'MLS':
        return x.max(axis=1), meta
    if method == 'SCALE':
        scaled, nbad = scale_logits(x, a.scale_percentile, a.scale_degenerate)
        return logsumexp(scaled, axis=1), {'fallback_rows': nbad}
    if method == 'MDS':
        best = np.full(len(x), -np.inf)
        for mu in model['means']:
            d = x - mu
            best = np.maximum(best, -np.einsum('ij,jk,ik->i', d, model['precision'], d))
        return best, meta
    if method == 'KNN':
        q = model['scaler'].transform(x) if model['scaler'] else normalize(x)
        distances = model['nn'].kneighbors(q, return_distance=True)[0]
        return (-distances.mean(axis=1) if a.knn_mode == 'mhood' else -distances[:, -1]**2), meta
    result = np.empty(len(x))
    for c in np.unique(labels):
        idx = labels == c
        key = -1 if method == 'iForest' and a.iforest_scope == 'global' else int(c)
        if key not in model:
            raise ValueError(f'{method}: no fitted model for predicted class {c}; '
                             'supply training coverage rather than silently dropping detections')
        if method == 'BAM':
            result[idx] = box_score(x[idx], *model[key])
        else:
            scaler, forest = model[key]
            result[idx] = forest.score_samples(scaler.transform(x[idx]))
    return result, meta


def score(method, model, data, a):
    parts, fallbacks = [], 0
    for start in range(0, len(data['x']), a.batch_size):
        stop = start + a.batch_size
        batch = {k: data[k][start:stop] for k in ('logits', 'x', 'labels')}
        s, meta = score_batch(method, model, batch, a)
        if not np.isfinite(s).all():
            raise ValueError(f'{method}: non-finite scores at batch starting {start}')
        parts.append(s); fallbacks += meta.get('fallback_rows', 0)
    return (np.concatenate(parts) if parts else np.empty(0)), fallbacks


def threshold95(scores):
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Threshold requires nonempty finite ID scores')
    required = math.ceil(.95 * len(scores))
    return float(np.partition(scores, len(scores) - required)[len(scores) - required])


def metrics(id_scores, ood_scores, keep=None):
    keep = np.ones(len(id_scores), dtype=bool) if keep is None else keep
    selected = id_scores[keep]
    tau = threshold95(selected)
    accepted = int(np.count_nonzero(ood_scores >= tau))
    auc = (100 * roc_auc_score(np.r_[np.ones(len(selected)), np.zeros(len(ood_scores))],
                              np.r_[selected, ood_scores]) if len(ood_scores) else None)
    return dict(n_id_raw=len(id_scores), n_id_kept=len(selected), n_ood=len(ood_scores),
                threshold=tau, id_retention_selected_pct=100 * float(np.mean(selected >= tau)),
                id_retention_full_pct=100 * float(np.mean(id_scores >= tau)),
                fpr95_pct=100 * accepted / len(ood_scores) if len(ood_scores) else None,
                auroc_pct=auc, remaining_ood=accepted)


def iqr_mask(x, a):
    """Reference's ID-val self-kNN + two-sided IQR, shared across methods."""
    v = StandardScaler().fit_transform(x)
    k = min(a.outlier_k, len(v))
    nn = NearestNeighbors(n_neighbors=k, n_jobs=a.jobs).fit(v)
    scores = []
    for start in range(0, len(v), a.batch_size):
        scores.append(-nn.kneighbors(v[start:start+a.batch_size])[0].mean(axis=1))
    scores = np.concatenate(scores)
    q1, q3 = np.percentile(scores, [25, 75])
    low, high = q1 - a.iqr_factor*(q3-q1), q3 + a.iqr_factor*(q3-q1)
    return (scores >= low) & (scores <= high), scores


def tail_mask(scores, labels, fraction, per_class):
    keep = np.ones(len(scores), dtype=bool)
    for c in np.unique(labels) if per_class else [None]:
        ix = np.flatnonzero(labels == c) if c is not None else np.arange(len(scores))
        n = int(np.floor(fraction * len(ix)))
        keep[ix[np.argsort(scores[ix], kind='stable')[:n]]] = False
    return keep


def atomic_npz(path, **arrays):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def method_label(m, a):
    if m == 'MSP':
        return 'MaxSigmoid' if a.msp_activation == 'sigmoid' else 'MSP'
    if m in ('EBO', 'MLS'):
        return m
    return m + '-logits' + ('-guarded' if m == 'SCALE' and a.scale_degenerate == 'identity' else '')


def run(a):
    paths = {k: getattr(a, k) for k in ('train', 'id_val', 'near_ood', 'far_ood') if getattr(a, k)}
    data = {k: read_split(p, k, a.background_index) for k, p in paths.items()}
    if not len(data['id_val']['x']):
        raise ValueError('ID-val is empty')
    ref = data['id_val']
    for name, d in data.items():
        if d['logits'].shape[1] != ref['logits'].shape[1]:
            raise ValueError(f'{name}: inconsistent logit dimension')
        if (d['names'] is None) != (ref['names'] is None) or (d['names'] is not None and not np.array_equal(d['names'], ref['names'])):
            raise ValueError('class_names must match across every split, or be absent in all')
    fitted = set(a.methods) & {'MDS', 'BAM', 'KNN', 'iForest'}
    if fitted and ('train' not in data or not len(data['train']['x'])):
        raise ValueError(f'{sorted(fitted)} require nonempty --train')
    a.out.mkdir(parents=True, exist_ok=True)
    params = {k: v for k, v in vars(a).items() if k not in
              ('train','id_val','near_ood','far_ood','out','methods','outlier','outlier_k',
               'iqr_factor','tail_fraction','tail_scope','force','self_test')}
    identity = dict(inputs={k: digest(p) for k,p in paths.items()}, parameters=params,
                    script_sha256=digest(__file__))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    report = dict(reference_commit='a8903441d6569e875189f96f63f829fa6c1548dc',
                  config={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                  score_cache_identity=identity, score_direction='higher_is_ID',
                  versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                  splits={k:dict(n=len(d['x']), labels_inferred=d['inferred_labels'],
                                 row_ids_generated=d['generated_ids']) for k,d in data.items()},
                  methods={}, errors={})
    if 'train' in data and len(data['train']['x']):
        ix = train_indices(data['train']['labels'], a.max_train_per_class, a.seed)
        train_x, train_y = data['train']['x'][ix], data['train']['labels'][ix]
        atomic_npz(a.out/'train_selection.npz', indices=ix, detection_ids=data['train']['ids'][ix])
        report['training_counts'] = {str(int(c)):int(np.sum(train_y == c)) for c in np.unique(train_y)}
    else:
        train_x = train_y = None
    shared_mask = None
    if a.outlier == 'mhood-iqr':
        mask_key = hashlib.sha256(json.dumps(dict(
            input_sha256=identity['inputs']['id_val'], script=identity['script_sha256'],
            background=a.background_index, k=a.outlier_k, factor=a.iqr_factor
        ), sort_keys=True).encode()).hexdigest()
        mask_path = a.out/'id_val_outlier_mask.npz'
        if mask_path.exists() and not a.force:
            with np.load(mask_path, allow_pickle=False) as saved:
                if 'fingerprint' in saved and str(saved['fingerprint'].item()) == mask_key:
                    shared_mask = saved['keep'].copy()
        if shared_mask is None:
            shared_mask, outlier_scores = iqr_mask(ref['x'], a)
            atomic_npz(mask_path, keep=shared_mask, scores=outlier_scores,
                       detection_ids=ref['ids'], fingerprint=np.array(mask_key))
        if not shared_mask.any():
            raise ValueError('Outlier mask removed all ID-val detections')
    rows = []
    eval_names = [k for k in data if k != 'train']
    for method in a.methods:
        started = time.perf_counter()
        label = method_label(method, a)
        cache = a.out / f'{label}_scores.npz'
        scores, fallback_counts = {}, {}
        reused = False
        try:
            if cache.exists() and not a.force:
                with np.load(cache, allow_pickle=False) as saved:
                    if str(saved['fingerprint'].item()) == fingerprint:
                        scores = {k: saved[k].copy() for k in eval_names}
                        fallback_counts = json.loads(str(saved['fallback_counts'].item()))
                        reused = True
            if not reused:
                print(f'[{label}] fitting/scoring', flush=True)
                model = fit_model(method, train_x, train_y, a)
                for k in eval_names:
                    scores[k], fallback_counts[k] = score(method, model, data[k], a)
                payload = dict(fingerprint=np.array(fingerprint),
                               fallback_counts=np.array(json.dumps(fallback_counts)), **scores)
                for k in eval_names:
                    payload[k+'_detection_ids'] = data[k]['ids']
                    payload[k+'_pred_labels'] = data[k]['labels']
                atomic_npz(cache, **payload)
                if model is not None:
                    joblib.dump(dict(model=model, method=method, config=params, identity=identity),
                                a.out/f'{label}_model.joblib')
            masks = {'full_id': None}
            if a.outlier == 'mhood-iqr':
                masks['mhood_iqr_id_val_diagnostic'] = shared_mask
            elif a.outlier == 'score-tail':
                masks['score_tail_id_val_diagnostic'] = tail_mask(scores['id_val'], ref['labels'],
                                                                 a.tail_fraction, a.tail_scope == 'classwise')
                atomic_npz(a.out/f'{label}_outlier_mask.npz', keep=masks['score_tail_id_val_diagnostic'],
                           detection_ids=ref['ids'])
            result = dict(cache_reused=reused, fallback_rows=fallback_counts, evaluations={})
            for protocol, keep in masks.items():
                result['evaluations'][protocol] = {}
                for split in eval_names:
                    if split == 'id_val':
                        continue
                    r = metrics(scores['id_val'], scores[split], keep)
                    result['evaluations'][protocol][split] = r
                    rows.append(dict(method=label, protocol=protocol, split=split, **r))
            result['wall_seconds'] = time.perf_counter() - started
            report['methods'][label] = result
            print(f'[{label}] done in {result["wall_seconds"]:.2f}s (cache={reused})', flush=True)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
            report['errors'][label] = str(e)
            print(f'[{label}] FAILED: {e}', flush=True)
        atomic_json(a.out/'report.json', report)
        if rows:
            with open(a.out/'summary.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    return 2 if report['errors'] else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('train', 'id-val', 'near-ood', 'far-ood'):
        p.add_argument('--'+k, type=Path)
    p.add_argument('--out', type=Path, default=Path('ood_results'))
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--background-index', type=int)
    p.add_argument('--msp-denominator', choices=['all','foreground'], default='all')
    p.add_argument('--msp-activation', choices=['softmax','sigmoid'], default='softmax')
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--scale-percentile', type=float, default=65.)
    p.add_argument('--scale-degenerate', choices=['error','identity'], default='error')
    p.add_argument('--mds-labels', choices=['argmax','predicted'], default='argmax')
    p.add_argument('--covariance', choices=['empirical','ledoit-wolf'], default='empirical')
    p.add_argument('--knn-mode', choices=['mhood','sun'], default='mhood')
    p.add_argument('--knn-k', type=int, default=5)
    p.add_argument('--bam-density', type=float, default=50., help='m-hood uses VOC=5, BDD=50')
    p.add_argument('--bam-max-boxes', type=int, default=256, help='speed cap per class; 0=no cap')
    p.add_argument('--bam-cluster', choices=['minibatch','kmeans'], default='minibatch')
    p.add_argument('--iforest-scope', choices=['classwise','global'], default='classwise')
    p.add_argument('--trees', type=int, default=200)
    p.add_argument('--iforest-samples', type=int, default=512)
    p.add_argument('--max-train-per-class', type=int, default=0, help='0=all; cap detections, not images')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--outlier', choices=['none','mhood-iqr','score-tail'], default='none')
    p.add_argument('--outlier-k', type=int, default=5)
    p.add_argument('--iqr-factor', type=float, default=1.5)
    p.add_argument('--tail-fraction', type=float, default=.05)
    p.add_argument('--tail-scope', choices=['global','classwise'], default='classwise')
    p.add_argument('--force', action='store_true', help='recompute scores even if fingerprint matches')
    return p


if __name__ == '__main__':
    p = parser(); args = p.parse_args()
    if args.id_val is None or (args.near_ood is None and args.far_ood is None):
        p.error('--id-val and at least one of --near-ood / --far-ood are required')
    for name in ('temperature','knn_k','bam_density','trees','iforest_samples','jobs','batch_size','outlier_k'):
        if getattr(args, name) <= 0:
            p.error(name+' must be positive')
    if not (0 <= args.scale_percentile <= 100 and 0 <= args.tail_fraction < 1):
        p.error('invalid percentile or tail fraction')
    if args.max_train_per_class < 0 or args.bam_max_boxes < 0 or args.iqr_factor < 0:
        p.error('caps and IQR factor must be nonnegative')
    with threadpool_limits(limits=args.jobs):
        raise SystemExit(run(args))
