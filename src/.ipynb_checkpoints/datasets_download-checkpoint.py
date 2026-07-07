import os
import shutil
import zipfile
import urllib.request
import tarfile

# 1. Clean up the broken/corrupted folders first
for folder in ["data/raw/RDD2022", "data/raw/CrackForest", "data/raw/Crack500"]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Cleared corrupted/incomplete directory: {folder}")

def get_folder_size(path):
    """Calculates actual size of a specific directory in GB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)

def download_and_extract(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    temp_file = os.path.join(target_dir, "downloaded_temp.bin")
    
    try:
        print(f"Downloading from: {url}")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, temp_file)
        print(f"Extracting contents into {target_dir}...")
        
        if zipfile.is_zipfile(temp_file):
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print("✓ Extracted ZIP file.")
        elif tarfile.is_tarfile(temp_file):
            with tarfile.open(temp_file, 'r:*') as tar_ref:
                tar_ref.extractall(target_dir)
            print("✓ Extracted TAR/TAR.GZ archive.")
        else:
            # Fallback for complex Figshare compression packages
            try:
                with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                print("✓ Extracted via fallback zip pipeline.")
            except:
                with tarfile.open(temp_file, 'r:*') as tar_ref:
                    tar_ref.extractall(target_dir)
                print("✓ Extracted via fallback tar pipeline.")

        if os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"✓ Successfully processed: {target_dir}\n")
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"❌ Failed processing {url}. Error: {e}\n")

# --- Run the corrected pipeline ---
print("Starting fresh direct downloads...\n")

# 2. CrackForest OOD Eval (~10 MB)[cite: 1]
download_and_extract("https://github.com/cuilimeng/CrackForest-dataset/archive/refs/heads/master.zip", "data/raw/CrackForest")

# 3. Crack500 OOD Eval (~350 MB)[cite: 1]
download_and_extract("https://github.com/fyangneil/pavement-crack-detection/archive/refs/heads/master.zip", "data/raw/Crack500")

# 4. Enforce placeholders[cite: 1]
os.makedirs("data/raw/SVRD", exist_ok=True)
os.makedirs("data/raw/UrbanSurface/images", exist_ok=True)
os.makedirs("data/raw/UrbanSurface/labels", exist_ok=True)

print(f"Actual data folder size on disk: {get_folder_size('data'):.2f} GB / 15.00 GB ceiling.")