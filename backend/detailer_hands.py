"""Bounded CPU palm/21-landmark inference through OpenCV DNN.

Adapted from OpenCV Zoo's mp_palmdet.py and mp_handpose.py at
48b10fa505aeb51fbd127fc60a5c660040e823b7 (Apache-2.0;
LICENSES/OpenCV-Zoo.txt). Changed for AuraDream: generated anchors,
deterministic bounded NMS, checked output/geometry, cooperative cancellation,
and inverse-affine full-landmark boxes. No MediaPipe runtime, delegates,
telemetry, model downloads or library imports occur in this module.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from .backends.base import Cancelled

PALM_SCORE = 0.5
HAND_SCORE = 0.8
PALM_NMS = 0.3
MAX_PALMS = 16
MAX_PALM_CANDIDATES = 256


def _cancel(should_cancel: Callable[[], bool]) -> None:
    if should_cancel():
        raise Cancelled()


def palm_anchors(np: Any) -> Any:
    """MediaPipe's 2016 centre anchors, y/x order, repeated per feature cell.

    The first 24x24 cells have two anchors each; the 12x12 cells have six.
    Values are generated in float64 instead of a 2016-row rounded literal.
    """
    return np.array([
        ((x + 0.5) / side, (y + 0.5) / side)
        for side, repeats in ((24, 2), (12, 6))
        for y in range(side) for x in range(side) for _ in range(repeats)
    ], dtype=np.float64)


def _finite(np: Any, values: Any, shape: tuple[int, ...], label: str) -> Any:
    values = np.asarray(values)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"The hand detector returned invalid {label}.")
    return values


def _overlap(a: Any, b: Any) -> float:
    width = max(0.0, float(min(a[2], b[2]) - max(a[0], b[0])))
    height = max(0.0, float(min(a[3], b[3]) - max(a[1], b[1])))
    intersection = width * height
    area_a = float((a[2] - a[0]) * (a[3] - a[1]))
    area_b = float((b[2] - b[0]) * (b[3] - b[1]))
    return intersection / (area_a + area_b - intersection)


def decode_palms(np: Any, boxes: Any, scores: Any, scale: float, bias: Any) -> list[Any]:
    """Map model outputs to original pixels and suppress duplicate palms.

    NMS operates on true corner boxes (OpenCV NMSBoxes expects xywh).
    Score, corner coordinates and original anchor index form a total order.
    """
    boxes = _finite(np, boxes, (1, 2016, 18), "palm coordinates")[0].astype(np.float64)
    logits = _finite(np, scores, (1, 2016, 1), "palm scores")[0, :, 0]
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -80.0, 80.0)))
    anchors = palm_anchors(np)
    centres = (boxes[:, :2] / 192.0 + anchors) * scale
    sizes = boxes[:, 2:4] / 192.0 * scale
    corners = np.c_[centres - sizes / 2, centres + sizes / 2]
    corners -= np.tile(bias, 2)
    indexes = [int(i) for i in np.flatnonzero(scores >= PALM_SCORE)]
    if any((sizes[i] <= 0).any() for i in indexes):
        raise ValueError("The hand detector returned an empty palm box.")
    indexes.sort(key=lambda i: (-float(scores[i]), *map(float, corners[i]), i))
    if len(indexes) > MAX_PALM_CANDIDATES:
        raise ValueError("The hand detector returned too many palm candidates.")
    kept: list[int] = []
    for i in indexes:
        if all(_overlap(corners[i], corners[j]) <= PALM_NMS for j in kept):
            kept.append(i)
            if len(kept) == MAX_PALMS:
                # Refuse rather than silently omit candidates the review could
                # never show. Overlapping duplicate candidates remain harmless.
                if any(all(_overlap(corners[k], corners[j]) <= PALM_NMS for j in kept)
                       for k in indexes[indexes.index(i) + 1:]):
                    raise ValueError("The hand detector found more than 16 separate palms.")
                break
    results = []
    for i in kept:
        points = (boxes[i, 4:].reshape(7, 2) / 192.0 + anchors[i]) * scale - bias
        results.append(np.r_[corners[i], points.reshape(-1), scores[i]])
    return results


def _crop_pad(cv: Any, np: Any, image: Any, box: Any, rotating: bool) -> tuple[Any, Any]:
    """Upstream's shifted/enlarged crop and square padding, with its offset.

    The first crop grows the palm 4x before rotation; the second shifts it
    0.4 palm-heights upward and grows it 3x to include the fingers.
    """
    size = box[1] - box[0]
    if not np.isfinite(box).all() or (size <= 0).any():
        raise ValueError("The hand detector returned invalid palm geometry.")
    centre = box.mean(axis=0) + np.array([0.0, 0.0 if rotating else -0.4]) * size
    half = size * (4 if rotating else 3) / 2
    crop_box = np.array([centre - half, centre + half]).astype(np.int32)
    crop_box[:, 0] = np.clip(crop_box[:, 0], 0, image.shape[1])
    crop_box[:, 1] = np.clip(crop_box[:, 1], 0, image.shape[0])
    width, height = crop_box[1] - crop_box[0]
    if width <= 0 or height <= 0:
        raise ValueError("The hand detector returned an off-image palm.")
    crop = image[crop_box[0, 1]:crop_box[1, 1], crop_box[0, 0]:crop_box[1, 0]]
    side = int(math.hypot(width, height)) if rotating else int(max(width, height))
    left, top = (side - int(width)) // 2, (side - int(height)) // 2
    crop = cv.copyMakeBorder(crop, top, side - int(height) - top,
                             left, side - int(width) - left,
                             cv.BORDER_CONSTANT, value=(0, 0, 0))
    return crop, crop_box[0] - np.array([left, top])


def pose_input(cv: Any, np: Any, bgr: Any, palm: Any) -> tuple[Any, Any, Any, Any, float]:
    """Upright 224-square RGB NHWC plus transforms back to original pixels."""
    image, first_bias = _crop_pad(cv, np, bgr, palm[:4].reshape(2, 2), True)
    points = palm[4:18].reshape(7, 2) - first_bias
    direction = points[2] - points[0]
    if float(np.linalg.norm(direction)) < 1e-6:
        raise ValueError("The hand detector returned coincident palm landmarks.")
    radians = math.pi / 2 - math.atan2(-float(direction[1]), float(direction[0]))
    radians -= 2 * math.pi * math.floor((radians + math.pi) / (2 * math.pi))
    centre = palm[:4].reshape(2, 2).mean(axis=0) - first_bias
    rotation = cv.getRotationMatrix2D(tuple(map(float, centre)), math.degrees(radians), 1.0)
    rotated = cv.warpAffine(image, rotation, (image.shape[1], image.shape[0]))
    rotated_points = np.c_[points, np.ones(7)] @ rotation.T
    landmark_box = np.array([rotated_points.min(axis=0), rotated_points.max(axis=0)])
    crop, second_bias = _crop_pad(cv, np, rotated, landmark_box, False)
    side = float(crop.shape[0])
    rgb = cv.cvtColor(cv.resize(crop, (224, 224), interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32)[np.newaxis] / 255.0
    return blob, cv.invertAffineTransform(rotation), first_bias, second_bias, side


def restore_landmarks(np: Any, landmarks: Any, inverse: Any, first_bias: Any,
                      second_bias: Any, side: float) -> Any:
    """All 21 model landmarks, in original full-canvas x/y coordinates."""
    values = _finite(np, landmarks, (1, 63), "hand landmarks")[0].reshape(21, 3)
    xy = values[:, :2].astype(np.float64) * (side / 224.0) + second_bias
    return np.c_[xy, np.ones(21)] @ inverse.T + first_bias


def _network(cv: Any, path: Path) -> Any:
    network = cv.dnn.readNet(str(path))
    network.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return network


def detect_hands(cv: Any, np: Any, bgr: Any, palm_path: Path, pose_path: Path,
                 should_cancel: Callable[[], bool]) -> list[tuple[float, float, float, float, float]]:
    """Return full-hand landmark bounds; detection is no anatomy verdict.

    Only verified caller-supplied assets are read. No network or MediaPipe
    runtime is used. Each neural call runs on OpenCV's explicit CPU target.
    """
    _cancel(should_cancel)
    height, width = bgr.shape[:2]
    ratio = 192.0 / max(height, width)
    resized_w, resized_h = max(1, int(width * ratio)), max(1, int(height * ratio))
    resized = cv.resize(bgr, (resized_w, resized_h), interpolation=cv.INTER_LINEAR)
    left, top = (192 - resized_w) // 2, (192 - resized_h) // 2
    padded = cv.copyMakeBorder(resized, top, 192 - resized_h - top,
                              left, 192 - resized_w - left, cv.BORDER_CONSTANT, value=(0, 0, 0))
    blob = cv.cvtColor(padded, cv.COLOR_BGR2RGB).astype(np.float32)[np.newaxis] / 255.0
    palm_net = _network(cv, palm_path)
    _cancel(should_cancel)
    palm_net.setInput(blob)
    output = palm_net.forward(palm_net.getUnconnectedOutLayersNames())
    _cancel(should_cancel)
    if len(output) != 2:
        raise ValueError("The hand detector returned the wrong number of palm outputs.")
    palms = decode_palms(np, output[0], output[1], float(max(height, width)),
                         np.array([left / ratio, top / ratio]))
    if not palms:
        return []
    pose_net = _network(cv, pose_path)
    found = []
    for palm in palms:
        _cancel(should_cancel)
        blob, inverse, first_bias, second_bias, side = pose_input(cv, np, bgr, palm)
        pose_net.setInput(blob)
        output = pose_net.forward(pose_net.getUnconnectedOutLayersNames())
        _cancel(should_cancel)
        if len(output) != 4:
            raise ValueError("The hand detector returned the wrong number of landmark outputs.")
        conf = float(_finite(np, output[1], (1, 1), "hand confidence")[0, 0])
        if not 0 <= conf <= 1:
            raise ValueError("The hand detector returned invalid hand confidence.")
        if conf < HAND_SCORE:
            continue
        points = restore_landmarks(np, output[0], inverse, first_bias, second_bias, side)
        low, high = points.min(axis=0), points.max(axis=0)
        if not np.isfinite(points).all() or (high <= low).any():
            raise ValueError("The hand detector returned invalid hand geometry.")
        found.append((*map(float, low), *map(float, high), conf))
    return found
