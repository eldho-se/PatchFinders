import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
load_dotenv()

DATASETS = {
    "RDD2022": {
        "description": "Road Damage Dataset 2022 - Global dashcam imagery",
        "url": "https://roboflow.com",
        "size": "~47,000 images",
    },
    "CRACK500": {
        "description": "Pavement crack detection dataset",
        "url": "https://roboflow.com",
        "size": "~500 images",
    },
    "Pothole600": {
        "description": "Pothole detection dataset",
        "url": "https://roboflow.com",
        "size": "~600 images",
    },
}

def list_datasets():
    print("\nAvailable Datasets:")
    print("-" * 50)
    for name, info in DATASETS.items():
        print(f"  {name}")
        print(f"    Description : {info['description']}")
        print(f"    Size        : {info['size']}")
        print()

def download_from_roboflow(api_key: str, workspace: str,
                            project: str, version: int,
                            save_dir: str = "data/raw"):
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location=save_dir)
        print(f"[INFO] Downloaded {project} to {save_dir}")
        return dataset
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        raise

if __name__ == "__main__":
    list_datasets()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY not found in .env file")
    else:
        print(f"[INFO] API Key found: {api_key[:8]}...")