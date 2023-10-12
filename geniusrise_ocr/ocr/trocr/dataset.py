# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple
import os
import json
import xml.etree.ElementTree as ET


class CustomOCRDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, dataset_format: str = "coco"):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_format = dataset_format
        self.image_paths, self.text_labels = self._load_data()

    def _load_data(self) -> Tuple[List[str], List[str]]:
        image_paths = []
        text_labels = []

        if self.dataset_format == "coco":
            annotations_path = os.path.join(self.root_dir, "annotations.json")
            with open(annotations_path, "r") as f:
                annotations = json.load(f)
            for item in annotations["images"]:
                image_paths.append(os.path.join(self.root_dir, item["file_name"]))
                text_labels.append(annotations["annotations"][item["id"]]["text"])

        elif self.dataset_format == "icdar":
            for xml_file in os.listdir(os.path.join(self.root_dir, "Annotations")):
                tree = ET.parse(os.path.join(self.root_dir, "Annotations", xml_file))
                root = tree.getroot()
                image_name = root.find("filename").text  # type: ignore
                image_paths.append(os.path.join(self.root_dir, "Images", image_name))  # type: ignore
                text = [obj.find("text").text for obj in root.findall("object")]  # type: ignore
                text_labels.append(" ".join(text))  # type: ignore

        elif self.dataset_format == "synthtext":
            # Assuming SynthText's ground truth is in the same folder as images
            for image_file in os.listdir(self.root_dir):
                if image_file.endswith(".jpg"):
                    image_paths.append(os.path.join(self.root_dir, image_file))
                    text_file = image_file.replace(".jpg", ".txt")
                    with open(os.path.join(self.root_dir, text_file), "r") as f:
                        text = f.read().strip()  # type: ignore
                    text_labels.append(text)

        return image_paths, text_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        text_label = self.text_labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, text_label
