"""
╔══════════════════════════════════════════════════════════════════╗
║           EchoVision  –  src/sentence_generator.py               ║
║  Converts NavEvent objects into natural spoken sentences, e.g.:  ║
║    "Person ahead at approximately two metres."                   ║
║    "Stairs on your left."                                        ║
║    "Open drain on your right. Please avoid it."                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

from .direction_detector import direction_phrase

# Friendly display names for hazard classes (label → spoken word)
LABEL_DISPLAY = {
    "pothole": "Pothole",
    "stairs": "Stairs",
    "speed_breaker": "Speed breaker",
    "open_drain": "Open drain",
    "road_barrier": "Road barrier",
    "construction_block": "Construction area",
    "traffic_cone": "Traffic cone",
    "curb": "Curb",
    "low_hanging_obstacle": "Low hanging obstacle",
}

CAUTION_SUFFIX = {
    "pothole": " Step carefully.",
    "open_drain": " Please avoid it.",
    "stairs": " Please be careful.",
    "construction_block": " Please be careful.",
    "low_hanging_obstacle": " Mind your head.",
}

NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def _distance_phrase(distance) -> str:
    if distance is None:
        return ""
    rounded = int(round(distance))
    word = NUMBER_WORDS.get(rounded, str(rounded))
    unit = "metre" if rounded == 1 else "metres"
    return f" at approximately {word} {unit}"


def event_to_sentence(event) -> str:
    label = LABEL_DISPLAY.get(event.label, event.label.replace("_", " ").capitalize())
    place = direction_phrase(event.direction)

    if event.hazard:
        sentence = f"{label} {place}."
        if event.critical:
            sentence = f"Warning! {label} very close, {place}."
        elif event.distance is not None:
            sentence = f"{label} {place}{_distance_phrase(event.distance)}."
        sentence += CAUTION_SUFFIX.get(event.label, "")
        return sentence

    # General everyday object
    label_cap = label[0].upper() + label[1:]
    if event.direction == "center":
        return f"{label_cap} ahead{_distance_phrase(event.distance)}."
    return f"{label_cap} {place}{_distance_phrase(event.distance)}."


def generate_announcement(events) -> str:
    """events: list[NavEvent] from navigation_engine.py (already prioritized)"""
    if not events:
        return ""
    sentences = [event_to_sentence(e) for e in events]
    return " ".join(sentences)
