import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision import transforms as T

class BaseballDataset(torch.utils.data.Dataset):
    def __init__(self, frames_dir, annotations_dir, transforms=None):
        self.frames_dir = frames_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms if transforms is not None else T.ToTensor()

        self.images = sorted(os.listdir(frames_dir))
        self.valid_images = []

        for img_name in self.images:
            video_name = img_name.split("_frame_")[0]
            xml_file = video_name + ".xml"
            xml_path = os.path.join(self.annotations_dir, xml_file)

            if not os.path.exists(xml_path):
                continue

            frame_str = img_name.split("_frame_")[-1].split(".jpg")[0]
            frame_id = int(frame_str)

            tree = ET.parse(xml_path)
            root = tree.getroot()

            found_box = False

            for track in root.findall("track"):
                if track.get("label") != "baseball":
                    continue

                for box in track.findall("box"):
                    if int(box.get("frame")) == frame_id and box.get("outside") == "0":
                        found_box = True
                        break

                if found_box:
                    break

            if found_box:
                self.valid_images.append(img_name)

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        img_path = os.path.join(self.frames_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        video_name = img_name.split("_frame_")[0]
        xml_file = video_name + ".xml"
        xml_path = os.path.join(self.annotations_dir, xml_file)

        frame_str = img_name.split("_frame_")[-1].split(".jpg")[0]
        frame_id = int(frame_str)

        boxes = []
        labels = []

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for track in root.findall("track"):
            if track.get("label") != "baseball":
                continue

            for box in track.findall("box"):
                if int(box.get("frame")) == frame_id and box.get("outside") == "0":
                    xtl = float(box.get("xtl"))
                    ytl = float(box.get("ytl"))
                    xbr = float(box.get("xbr"))
                    ybr = float(box.get("ybr"))
                    boxes.append([xtl, ytl, xbr, ybr])
                    labels.append(1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd
        }

        image = self.transforms(image)

        return image, target