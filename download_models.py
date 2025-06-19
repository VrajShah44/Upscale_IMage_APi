import os
import requests
from tqdm import tqdm

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

if __name__ == "__main__":
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    dest = "weights/RealESRGAN_x4plus.pth"

    if os.path.exists(dest):
        print("Model already exists.")
    else:
        print("Downloading RealESRGAN_x4plus.pth...")
        try:
            download_file(url, dest)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
