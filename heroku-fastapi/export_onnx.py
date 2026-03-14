import torch
import segmentation_models_pytorch as smp

# 1. Chargement du modèle avec PyTorch (Localement seulement)
model = smp.Unet(
    encoder_name="resnet34", 
    encoder_weights=None, 
    in_channels=1, 
    classes=8
)
model.load_state_dict(torch.load("tuned_model_epochs_3_vs_ref_winner.pth", map_location="cpu"))
model.eval()

# 2. Création d'un tenseur de test (Dummy input)
# Taille (Batch, Canaux, Hauteur, Largeur)
dummy_input = torch.randn(1, 1, 256, 256)

# 3. Exportation vers ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx", 
    export_params=True, 
    opset_version=15, # Version recommandée pour la compatibilité
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("✅ Export réussi : model.onnx est prêt.")