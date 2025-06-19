import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from realesrgan import RealESRGAN
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
import pandas as pd

factor = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=factor)
model.load_weights(f'weights/RealESRGAN_x{factor}.pth')

def upscale_image(input_pil_image):
    with torch.no_grad():
        output_image = model.predict(input_pil_image)
    return output_image

def calculate_psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * log10(255.0 / sqrt(mse))

# Example: process a single image and display
input_image_path = "input.jpg"  # Change to your image path
img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_pil_image = Image.fromarray(img_rgb)
enhanced_pil_image = upscale_image(input_pil_image)

# Display original and enhanced images
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(np.array(enhanced_pil_image))
plt.title("Enhanced")
plt.axis('off')
plt.show()

# --- Batch processing example (optional) ---
input_root = 'input_images/'
output_root = 'output_images/'
os.makedirs(output_root, exist_ok=True)
categories = ['sharpen']
for category in categories:
    input_folder = os.path.join(input_root, category)
    output_folder = os.path.join(output_root, category)
    os.makedirs(output_folder, exist_ok=True)
    results = []
    for idx, filename in enumerate(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_image_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_pil_image = Image.fromarray(img_rgb)
            enhanced_pil_image = upscale_image(input_pil_image)
            enhanced_image = np.array(enhanced_pil_image)
            enhanced_resized = cv2.resize(enhanced_image, (img_rgb.shape[1], img_rgb.shape[0]))
            output_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_image_path, enhanced_resized)
            img_rgb_float = img_rgb.astype('float32') / 255.0
            enhanced_resized_float = enhanced_resized.astype('float32') / 255.0
            ssim_value = ssim(img_rgb_float, enhanced_resized_float, channel_axis=-1, data_range=1.0)
            psnr_value = calculate_psnr(img_rgb, enhanced_resized)
            results.append({'Image Name': filename, 'SSIM': ssim_value, 'PSNR': psnr_value})
    csv_path = os.path.join(output_root, f'{category}_metrics.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)