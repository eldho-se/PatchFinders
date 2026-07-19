UNIFIED_CLASSES = {
    0: "pothole",
    1: "longitudinal_crack",
    2: "transverse_crack",
    3: "alligator_crack",
    4: "rutting",
    5: "surface_deterioration"
}

LABEL_MAPPING = {
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
    "crack": "longitudinal_crack",
    "pothole": "pothole",
}

CLASS_TO_IDX = {v: k for k, v in UNIFIED_CLASSES.items()}