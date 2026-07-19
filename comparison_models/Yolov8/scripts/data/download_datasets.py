from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_key)

# Dataset 1 - Road Damage (srisow)
print("[INFO] Downloading Dataset 1 - Road Damage...")
project = rf.workspace("srisow").project("road-damage-5lxtz")
version = project.version(1)
version.download("yolov8", location="data/raw/RDD2022")
print("[INFO] Dataset 1 complete!")

# Dataset 2 - Road Damage (testing-skqqq)
print("[INFO] Downloading Dataset 2 - Road Damage...")
project = rf.workspace("testing-skqqq").project("road-damage-lzosi")
version = project.version(1)
version.download("yolov8", location="data/raw/CRACK500")
print("[INFO] Dataset 2 complete!")

# Dataset 3 - Road Damage (image-process-fe7aa)
print("[INFO] Downloading Dataset 3 - Road Damage...")
project = rf.workspace("image-process-fe7aa").project("road-damage-3jrtr")
version = project.version(1)
version.download("yolov8", location="data/raw/Pothole600")
print("[INFO] Dataset 3 complete!")

print("\n[INFO] All datasets downloaded successfully!")
print("[INFO] Checking downloaded files...")

for folder in ["data/raw/RDD2022", "data/raw/CRACK500", "data/raw/Pothole600"]:
    if os.path.exists(folder):
        count = sum([len(files) for r, d, files in os.walk(folder)])
        print(f"  {folder}: {count} files")
    else:
        print(f"  {folder}: NOT FOUND")