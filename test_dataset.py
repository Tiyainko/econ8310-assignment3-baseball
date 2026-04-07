from dataset import BaseballDataset

dataset = BaseballDataset("data/frames", "data/annotations", annotated_only=True)

print("Dataset size:", len(dataset))

image, target = dataset[0]

print("Image type:", type(image))
print("Boxes:", target["boxes"])
print("Labels:", target["labels"])