import os
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from dw_vit import DWViT

# 1. Configuración de Hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# 2. Dataset personalizado COCO JSON
class CocoPVDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        self.images = {img['id']: img for img in self.coco['images']}
        self.img_ids = list(self.images.keys())
        
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
            
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert('RGB')
        anns = self.annotations.get(img_id, [])
        num_boxes = len(anns)
        
        if self.transform:
            image = self.transform(image)
            
        return image, num_boxes

# Transformaciones (640x640)
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_root = "Satellite-PV-2"
train_dataset = CocoPVDataset(os.path.join(data_root, "train"), os.path.join(data_root, "train", "_annotations.coco.json"), transform=transform)
val_dataset = CocoPVDataset(os.path.join(data_root, "valid"), os.path.join(data_root, "valid", "_annotations.coco.json"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

print(f"✓ Train: {len(train_dataset)} parches | Valid: {len(val_dataset)} parches")

# 3. Instanciar DW-ViT con multiventana dinámica
model = DWViT(
    img_size=640,
    patch_size=4,
    in_chans=3,
    num_classes=1,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[4, 8, 16, 32],
    window_size=[7, 7]
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
criterion = nn.MSELoss()

# 4. Hiperparámetros de Early Stopping
max_epochs = 1000
patience = 50
best_val_mae = float('inf')
patience_counter = 0

csv_file = "dwvit_training_metrics.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_mae"])

print(f"\n🚀 Iniciando entrenamiento (Máx: {max_epochs} épocas, Paciencia Early Stop: {patience})...\n")

for epoch in range(1, max_epochs + 1):
    model.train()
    total_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_train_loss = total_loss / len(train_loader)

    # Validación y MAE
    model.eval()
    mae_errors = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.clamp(outputs, min=0).cpu().numpy().flatten()
            reals = targets.numpy().flatten()
            mae_errors.extend(np.abs(preds - reals))
            
    val_mae = float(np.mean(mae_errors))
    
    # Guardar métricas en CSV
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_train_loss, val_mae])
    
    # Comprobar Early Stopping y Mejor Checkpoint
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0
        torch.save(model.state_dict(), "dwvit_best_model.pth")
        status = "⭐ ¡Nuevo mejor modelo guardado!"
    else:
        patience_counter += 1
        status = f"Paciencia: {patience_counter}/{patience}"

    print(f"Época [{epoch:04d}/{max_epochs:04d}] | Train Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f} | {status}")

    if patience_counter >= patience:
        print(f"\n⏹️ Early Stopping activado en la época {epoch}. El modelo no mejoró en {patience} épocas consecutivas.")
        break

print(f"\n✓ Mejor Val MAE alcanzado: {best_val_mae:.4f}")
print("✓ Archivo de pesos: dwvit_best_model.pth")
print(f"✓ Archivo de métricas: {csv_file}")
