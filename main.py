from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.routes import router
from app.middleware.auth import verify_api_key
from app.middleware.rate_limit import limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from dotenv import load_dotenv
import os
import uuid
import torch
from PIL import Image
import io
import numpy as np

from realesrgan import RealESRGANer

# Load environment variables
load_dotenv()

app = FastAPI()

# ----- Model Setup -----
model_path = 'weights/RealESRGAN_x4plus.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Real-ESRGAN weights not found at {model_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the Real-ESRGAN model without custom RRDBNet
try:
    upscaler = RealESRGANer(
        scale=4,
        model_path=model_path,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize RealESRGAN: {str(e)}")

# ----- Static Files Setup -----
if not os.path.exists("public"):
    os.makedirs("public")
app.mount("/public", StaticFiles(directory="public"), name="public")

# ----- Middleware and Routes -----
app.include_router(router)
app.state.limiter = limiter
app.middleware('http')(verify_api_key)
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "Too many requests"})

# ----- Image Upscaling Endpoint -----
@app.post("/upscale-image")
async def upscale_image(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        image_bytes = await file.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(input_image)

        with torch.no_grad():
            output_np, _ = upscaler.enhance(img_np, outscale=4)

        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        output_path = f"public/{unique_filename}"
        Image.fromarray(output_np).save(output_path)

        return {"url": f"/public/{unique_filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upscale image: {str(e)}")
