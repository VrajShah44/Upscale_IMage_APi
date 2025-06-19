
from fastapi import APIRouter, File, UploadFile, Form, Request
from app.services.upscale import upscale_image
from fastapi.responses import JSONResponse
import os

router = APIRouter()

@router.post("/api/upscale")
async def upscale(
    request: Request,
    image: UploadFile = File(...),
    factor: int = Form(2),
    enhance_faces: bool = Form(False)
):
    try:
        output_url = await upscale_image(image, factor, enhance_faces)
        return {"status": "success", "url": output_url}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})