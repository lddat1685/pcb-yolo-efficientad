"""R8 end-to-end postprocessing through predict; hand-built masks are the oracle."""

import numpy as np
import pytest

from helpers.step4_spec import assert_box, field


@pytest.mark.parametrize("offset,expected", [(56, 2), (55, 2), (54, 1)])
def test_global_nms_integration_threshold_strict_045(offset, expected):
    from pcb_lab.inference.tiling import TileManager
    # At offset=55: intersection=90*145, union=200*145, IoU exactly .45.
    boxes = np.array([[0, 0, 145, 145, .9], [offset, 0, offset + 145, 145, .7]], dtype=np.float64)
    actual = TileManager().global_nms(boxes, iou=.45)
    assert len(actual) == expected
    np.testing.assert_array_equal(actual[0], boxes[0])


def boxes_for(predict, heatmap):
    boxes = field(predict(heatmap), "boxes")
    assert isinstance(boxes, list)
    for box in boxes:
        assert_box(box)
    return sorted(boxes, key=lambda box: tuple(field(box, "xyxy")))


@pytest.mark.parametrize("value", [0.0, 0.49])
def test_below_tau_has_no_boxes(step4_predict_map, value):
    assert boxes_for(step4_predict_map, np.full((640, 640), value, np.float32)) == []


def test_threshold_then_rect_uses_peak_and_anomaly_class(step4_predict_map):
    heatmap = np.zeros((640, 640), np.float32)
    heatmap[100:110, 200:215] = 0.6
    heatmap[104, 206] = 0.9
    boxes = boxes_for(step4_predict_map, heatmap)
    assert len(boxes) == 1
    np.testing.assert_allclose(field(boxes[0], "xyxy"), [200, 100, 215, 110], atol=1e-5, rtol=0)
    assert field(boxes[0], "confidence") == pytest.approx(0.9, abs=1e-6)


def test_opening_removes_thin_component_even_when_area_exceeds_minimum(step4_predict_map):
    heatmap = np.zeros((640, 640), np.float32)
    heatmap[50:52, 100:160] = 0.9  # area=120; 3x3 opening removes it
    heatmap[80, 80] = 1.0
    assert boxes_for(step4_predict_map, heatmap) == []


def test_opening_breaks_one_pixel_bridge_before_components(step4_predict_map):
    heatmap = np.zeros((640, 640), np.float32)
    heatmap[100:110, 100:110] = 0.8
    heatmap[100:110, 130:140] = 0.7
    heatmap[104, 110:130] = 0.9
    boxes = boxes_for(step4_predict_map, heatmap)
    assert [list(field(box, "xyxy")) for box in boxes] == [[100, 100, 110, 110], [130, 100, 140, 110]]
    assert [field(box, "confidence") for box in boxes] == pytest.approx([0.8, 0.7], abs=1e-6)


@pytest.mark.parametrize("height,width,expected", [(3, 6, 0), (4, 5, 0), (3, 7, 1)])
def test_area_filter_is_strictly_greater_than_twenty(step4_predict_map, height, width, expected):
    heatmap = np.zeros((640, 640), np.float32)
    heatmap[100:100 + height, 100:100 + width] = 0.8
    assert len(boxes_for(step4_predict_map, heatmap)) == expected


def test_component_area_is_not_bounding_rectangle_area(step4_predict_map):
    heatmap = np.zeros((640, 640), np.float32)
    # Two overlapping 3x3 squares survive opening. One component, area=17,
    # bounding rectangle=25: a bbox-area filter would incorrectly retain it.
    heatmap[100:103, 100:103] = 0.8
    heatmap[102:105, 102:105] = 0.8
    assert boxes_for(step4_predict_map, heatmap) == []


def test_component_peak_does_not_include_another_component_inside_its_bbox(step4_predict_map):
    heatmap = np.zeros((640, 640), np.float32)
    heatmap[100:200, 100:200] = 0.7
    heatmap[106:194, 106:194] = 0
    heatmap[140:160, 140:160] = 0.95
    boxes = boxes_for(step4_predict_map, heatmap)
    assert len(boxes) == 2  # IoU=400/10000=.04, so both survive NMS
    assert list(field(boxes[0], "xyxy")) == [100, 100, 200, 200]
    assert field(boxes[0], "confidence") == pytest.approx(0.7, abs=1e-6)
    assert field(boxes[1], "confidence") == pytest.approx(0.95, abs=1e-6)


@pytest.mark.parametrize("inner_size,expected", [(60, 2), (70, 1)])
def test_global_nms_045_after_components_keeps_higher_peak(step4_predict_map, inner_size, expected):
    heatmap = np.zeros((640, 640), np.float32)
    heatmap[100:200, 100:200] = 0.9
    heatmap[106:194, 106:194] = 0
    start = 150 - inner_size // 2
    heatmap[start:start + inner_size, start:start + inner_size] = 0.7
    # Disjoint components with nested rectangles: IoU=.36 or .49.
    boxes = boxes_for(step4_predict_map, heatmap)
    assert len(boxes) == expected
    assert list(field(boxes[0], "xyxy")) == [100, 100, 200, 200]
    assert field(boxes[0], "confidence") == pytest.approx(0.9, abs=1e-6)


@pytest.mark.parametrize("y,x", [(220, 220), (444, 444), (0, 0), (630, 630)])
def test_boxes_crossing_tile_seams_or_image_edges_stay_global(step4_predict_map, y, x):
    heatmap = np.zeros((640, 640), np.float32)
    heatmap[y:y + 10, x:x + 10] = 0.8
    boxes = boxes_for(step4_predict_map, heatmap)
    assert len(boxes) == 1
    np.testing.assert_allclose(field(boxes[0], "xyxy"), [x, y, x + 10, y + 10], atol=1e-5, rtol=0)
