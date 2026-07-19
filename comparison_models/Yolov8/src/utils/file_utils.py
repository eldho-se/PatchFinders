# File and folder helper functions

import os
import shutil
from pathlib import Path

def get_image_paths(folder: str) -> list:
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    paths = []
    for ext in extensions:
        paths.extend(Path(folder).glob(f"*{ext}"))
    return sorted(paths)

def get_label_paths(folder: str) -> list:
    return sorted(Path(folder).glob("*.txt"))

def copy_file(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def count_files(folder: str, ext: str = ".jpg") -> int:
    return len(list(Path(folder).glob(f"*{ext}")))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)