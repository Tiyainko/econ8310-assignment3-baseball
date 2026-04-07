import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import BaseballDataset

def collate_fn(batch):
    return tuple(zip(*batch))

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

dataset = BaseballDataset("data/frames", "data/annotations")


dataset = Subset(dataset, list(range(min(50, len(dataset)))))

loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

model.train()

for i, (images, targets) in enumerate(loader):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    print(f"Step {i+1}/{len(loader)}, Loss: {losses.item():.4f}")

    
    if i >= 20:
        break

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/baseball_detector.pth")

print("DONE — Model saved")