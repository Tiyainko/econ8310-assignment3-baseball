import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import BaseballDataset


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)


model.load_state_dict(torch.load("models/baseball_detector.pth", map_location=device))

model.to(device)
model.eval()

print("Model loaded successfully!")


dataset = BaseballDataset("data/frames", "data/annotations")

image, target = dataset[0]
image = image.to(device)


with torch.no_grad():
    prediction = model([image])

print("Prediction output:")
print(prediction)