import os
import uuid
from PIL import Image

TEMP_DIR = "public/temp/"
OUTPUT_DIR = "public/output/"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def save_temp_image(upload_file):
    file_ext = upload_file.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_ext}"
    temp_path = os.path.join(TEMP_DIR, file_name)
    with open(temp_path, "wb") as f:
        f.write(await upload_file.read())
    return temp_path

def get_output_path(original_filename):
    ext = original_filename.split(".")[-1]
    return os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.{ext}")
