"""
Basic smoke tests — verify modules import and pure-logic functions behave
correctly, without needing a webcam, GPU, or downloaded model weights.

Run with:
    pytest tests/
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import config
from src.direction_detector import get_direction, direction_phrase
from src.obstacle_filter import _iou
from src.sentence_generator import event_to_sentence
from src.navigation_engine import NavEvent


def test_direction_left():
    assert get_direction((0, 0, 50, 50), frame_width=300) == "left"


def test_direction_right():
    assert get_direction((250, 0, 299, 50), frame_width=300) == "right"


def test_direction_center():
    assert get_direction((120, 0, 180, 50), frame_width=300) == "center"


def test_direction_phrase_mapping():
    assert direction_phrase("left") == "on your left"
    assert direction_phrase("center") == "ahead"


def test_iou_identical_boxes():
    box = (0, 0, 10, 10)
    assert _iou(box, box) == 1.0


def test_iou_no_overlap():
    assert _iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_sentence_hazard_critical():
    event = NavEvent(label="pothole", distance=1.0, direction="center",
                      hazard=True, urgent=True, critical=True)
    sentence = event_to_sentence(event)
    assert "Warning" in sentence
    assert "Pothole" in sentence


def test_sentence_general_object():
    event = NavEvent(label="person", distance=3.0, direction="left",
                      hazard=False, urgent=False, critical=False)
    sentence = event_to_sentence(event)
    assert "Person" in sentence
    assert "left" in sentence


def test_nav_classes_match_dataset_yaml_order():
    # training/dataset.yaml class indices must match config.NAV_CLASSES order
    assert config.NAV_CLASSES[0] == "stairs"
    assert config.NAV_CLASSES[1] == "pothole"
    assert len(config.NAV_CLASSES) == 9
