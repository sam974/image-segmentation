import os
from fastapi import FastAPI, UploadFile, File, Response
from PIL import Image
import io
import numpy as np
import torch
import segmentation_models_pytorch as smp

# 1. Instantiate the FastAPI application
app = FastAPI()

# 2. Define the device for PyTorch operations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Redefinition de la même architecture de modèle
# Paramètres correspondent à ceux de l'entraînement
model_reconstruit = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1, # Correspond au monochrome utilisé
    classes=8      # Correspond à len(CLASS_NAMES)
)

# 4. Chargez les poids depuis le fichier '.pth'
# Assurez-vous que ce chemin est correct pour le déploiement Docker
model_path = os.path.join("/app", "tuned_model_epochs_3_vs_ref_winner.pth") # Adjust path for Docker
print(f"Loading model weights from: {model_path}")
model_reconstruit.load_state_dict(torch.load(model_path, map_location=device))
model_reconstruit.eval()

# 5. Move the model to the defined device
model_reconstruit.to(device);
print("✅ Model moved to device.")

# 6. Define CLASS_NAMES and COLOR_MAP
CLASS_NAMES = ['background', 'skin', 'nose', 'eye_g', 'hair', 'ear', 'mouth', 'lip']
COLOR_MAP = [
    [0, 0, 0],         # 0. Background (Black)
    [255, 204, 153],   # 1. Skin (Light Peach)
    [102, 51, 0],      # 2. Nose (Dark Brown)
    [153, 204, 255],   # 3. Eye_g (Light Blue)
    [153, 0, 0],       # 4. Hair (Dark Red)
    [204, 153, 255],   # 5. Ear (Light Purple)
    [0, 102, 0],       # 6. Mouth (Dark Green)
    [255, 102, 102]    # 7. Lip (Light Red)
]

IMG_SIZE = (256, 256)

async def preprocess_image(image_file: UploadFile):
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.convert('L')
    image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    image_np = np.array(image)
    image_np = image_np / 255.0
    image_tensor = torch.from_numpy(image_np).float()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    image_tensor = image_tensor.to(device)
    return image_tensor

@app.post("/predict_mask")
async def predict_mask(file: UploadFile = File(...)):
    preprocessed_image = await preprocess_image(file)

    with torch.no_grad():
        outputs = model_reconstruit(preprocessed_image)

    predicted_mask_tensor = torch.argmax(outputs, dim=1)
    predicted_mask_np = predicted_mask_tensor.cpu().numpy()
    predicted_mask_single_image = predicted_mask_np[0]

    colored_mask = np.zeros((*predicted_mask_single_image.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(COLOR_MAP):
        colored_mask[predicted_mask_single_image == class_idx] = color
    mask_pil_image = Image.fromarray(colored_mask, mode='RGB')

    img_byte_arr = io.BytesIO()
    mask_pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type="image/png")
