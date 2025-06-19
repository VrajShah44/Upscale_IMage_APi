import os
from app.utils.image_utils import save_temp_image, get_output_path
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
from PIL import Image
import numpy as np
import shutil

async def upscale_image(image_file, factor, enhance_faces):
    temp_path = await save_temp_image(image_file)
    img = Image.open(temp_path)

    # Load Real-ESRGAN
    upsampler = RealESRGANer(scale=factor, model_path='weights/RealESRGAN_x4plus.pth', tile=0, tile_pad=10, pre_pad=0, half=False)
    output, _ = upsampler.enhance(img, outscale=factor)

    # Optional: Face Enhancement with GFPGAN
    if enhance_faces:
        face_enhancer = GFPGANer(model_path='weights/GFPGANv1.3.pth', upscale=factor)
        _, _, output = face_enhancer.enhance(np.array(output), has_aligned=False, only_center_face=False, paste_back=True)

    output_path = get_output_path(image_file.filename)

    if not isinstance(output, Image.Image):
        output = Image.fromarray(output)
    output.save(output_path)
    output.save(output_path)
    os.remove(temp_path)

    return f"/public/output/{os.path.basename(output_path)}"
