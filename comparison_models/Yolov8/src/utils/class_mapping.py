# src/utils/class_mapping.py

UNIFIED_CLASSES = {
    0: "pothole",
    1: "longitudinal_crack",
    2: "transverse_crack",
    3: "alligator_crack",
    4: "rutting",
    5: "surface_deterioration"
}

# Map raw dataset labels to unified schema
LABEL_MAPPING = {
    # RDD2022 labels
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
    # CRACK500 labels
    "crack": "longitudinal_crack",
    # Pothole600 labels
    "pothole": "pothole",
    # DeepCrack labels
    "Crack": "longitudinal_crack",
}

# Reverse mapping: name to index
CLASS_TO_IDX = {v: k for k, v in UNIFIED_CLASSES.items()}