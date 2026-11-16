"""CPU-only detector geometry/cancellation checks; no OpenCV or assets."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.backends.base import Cancelled
from backend.detailer_hands import decode_palms, detect_hands, palm_anchors, restore_landmarks


def palm_output(entries):
    boxes = np.zeros((1, 2016, 18), dtype=np.float32)
    logits = np.full((1, 2016, 1), -80.0, dtype=np.float32)
    anchors = palm_anchors(np)
    for index, (left, top, right, bottom), score in entries:
        centre = np.array([(left + right) / 2, (top + bottom) / 2])
        boxes[0, index, :2] = centre - anchors[index] * 192
        boxes[0, index, 2:4] = [right - left, bottom - top]
        # Seven different plausible palm points, in the same network units.
        points = np.array([[left + n, top + n] for n in range(7)])
        boxes[0, index, 4:] = (points - anchors[index] * 192).reshape(-1)
        logits[0, index, 0] = score
    return boxes, logits


def test_anchor_feature_map_order_and_repetitions():
    anchors = palm_anchors(np)
    assert anchors.shape == (2016, 2)
    np.testing.assert_array_equal(anchors[0], anchors[1])
    np.testing.assert_allclose(anchors[0], [0.5 / 24, 0.5 / 24])
    np.testing.assert_allclose(anchors[2], [1.5 / 24, 0.5 / 24])
    np.testing.assert_allclose(anchors[1151], [23.5 / 24, 23.5 / 24])
    np.testing.assert_allclose(anchors[1152:1158], np.tile([0.5 / 12, 0.5 / 12], (6, 1)))
    np.testing.assert_allclose(anchors[-1], [11.5 / 12, 11.5 / 12])


def test_palm_decode_nms_uses_corners_and_preserves_original_coordinate_space():
    boxes, logits = palm_output([
        (1, (20, 30, 50, 60), 3.0),
        (0, (20, 30, 50, 60), 4.0),
        (2, (100, 80, 120, 110), 2.0),
    ])
    found = decode_palms(np, boxes, logits, 384, np.array([0, 12]))
    assert len(found) == 2
    np.testing.assert_allclose(found[0][:4], [40, 48, 100, 108])
    np.testing.assert_allclose(found[1][:4], [200, 148, 240, 208])
    np.testing.assert_allclose(found[0][4:6], [40, 48])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_invalid_neural_values_are_errors_not_no_detections(bad):
    boxes, logits = palm_output([])
    boxes[0, 0, 0] = bad
    with pytest.raises(ValueError, match="invalid palm coordinates"):
        decode_palms(np, boxes, logits, 192, np.zeros(2))


def test_degenerate_box_is_refused():
    boxes, logits = palm_output([(0, (20, 30, 20, 60), 4.0)])
    with pytest.raises(ValueError, match="empty palm box"):
        decode_palms(np, boxes, logits, 192, np.zeros(2))


def test_too_many_distinct_palms_are_refused_instead_of_silently_truncated():
    entries = [(i, (float(i * 10), 0.0, float(i * 10 + 2), 2.0), 4.0) for i in range(17)]
    boxes, logits = palm_output(entries)
    with pytest.raises(ValueError, match="more than 16"):
        decode_palms(np, boxes, logits, 192, np.zeros(2))


def test_all_twenty_one_landmarks_follow_crop_scale_and_inverse_rotation():
    xyz = np.zeros((1, 63), dtype=np.float32)
    xyz.reshape(21, 3)[:, :2] = np.array([[n, n * 2] for n in range(21)])
    inverse = np.array([[0.0, -1.0, 90.0], [1.0, 0.0, 10.0]])
    restored = restore_landmarks(np, xyz, inverse, np.array([300, 400]), np.array([20, 30]), 448)
    assert restored.shape == (21, 2)
    np.testing.assert_allclose(restored[0], [360, 430])
    np.testing.assert_allclose(restored[-1], [280, 470])


def test_cancel_before_loading_any_network():
    with pytest.raises(Cancelled):
        detect_hands(None, np, np.zeros((192, 192, 3), dtype=np.uint8),
                     Path("palm.onnx"), Path("pose.onnx"), lambda: True)


def fake_cv(output):
    events = []

    class Net:
        def setPreferableBackend(self, value):
            events.append(("backend", value))

        def setPreferableTarget(self, value):
            events.append(("target", value))

        def setInput(self, value):
            assert value.shape == (1, 192, 192, 3)

        def getUnconnectedOutLayersNames(self):
            return ("boxes", "scores")

        def forward(self, names):
            events.append(("forward", names))
            return output

    def load(path):
        events.append(("load", path))
        return Net()

    return SimpleNamespace(
        dnn=SimpleNamespace(readNet=load, DNN_BACKEND_OPENCV=3, DNN_TARGET_CPU=0),
        INTER_LINEAR=1, BORDER_CONSTANT=0, COLOR_BGR2RGB=4,
        resize=lambda image, size, **kwargs: image,
        copyMakeBorder=lambda image, *args, **kwargs: image,
        cvtColor=lambda image, conversion: image[:, :, ::-1],
    ), events


def test_no_palms_never_loads_pose_and_explicitly_sets_cpu():
    cv, events = fake_cv(palm_output([]))
    found = detect_hands(cv, np, np.zeros((192, 192, 3), dtype=np.uint8),
                         Path("palm.onnx"), Path("pose.onnx"), lambda: False)
    assert found == []
    assert events[:3] == [("load", "palm.onnx"), ("backend", 3), ("target", 0)]
    assert len([event for event in events if event[0] == "load"]) == 1


def test_cancel_after_neural_call_drops_results():
    cv, events = fake_cv(palm_output([]))
    with pytest.raises(Cancelled):
        detect_hands(cv, np, np.zeros((192, 192, 3), dtype=np.uint8),
                     Path("palm.onnx"), Path("pose.onnx"),
                     lambda: any(event[0] == "forward" for event in events))


def test_wrong_output_count_is_not_empty_success():
    cv, _ = fake_cv([np.zeros((1, 2016, 18))])
    with pytest.raises(ValueError, match="wrong number"):
        detect_hands(cv, np, np.zeros((192, 192, 3), dtype=np.uint8),
                     Path("palm.onnx"), Path("pose.onnx"), lambda: False)
