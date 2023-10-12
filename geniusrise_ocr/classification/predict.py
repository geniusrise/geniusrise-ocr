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

import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict
from geniusrise.core.data import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise import Bolt


class ImageClassPredictor(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `ImageClassPredictor` class classifies images using a pre-trained PyTorch model.
        It assumes that the `input.input_folder` contains sub-folders of images to be classified.
        The classified images are saved in `output.output_folder`, organized by their predicted labels.

        Args:
            input (BatchInput): Instance of BatchInput for reading data.
            output (BatchOutput): Instance of BatchOutput for saving data.
            state (State): Instance of State for maintaining state.
            **kwargs: Additional keyword arguments.

        ## Command Line Invocation with geniusrise
        ```bash
        genius ImageClassPredictor rise \
            batch \
                --bucket my_bucket \
                --s3_folder s3/input \
            batch \
                --bucket my_bucket \
                --s3_folder s3/output \
            none \
            predict \
                --args classes='{"0": "cat", "1": "dog"}' model_path=/path/to/model.pth
        ```

        ## YAML Configuration with geniusrise
        ```yaml
        version: "1"
        spouts:
            image_classification:
                name: "ImageClassPredictor"
                method: "predict"
                args:
                    classes: '{"0": "cat", "1": "dog"}'
                    model_path: "/path/to/model.pth"
                input:
                    type: "batch"
                    args:
                        bucket: "my_bucket"
                        s3_folder: "s3/input"
                output:
                    type: "batch"
                    args:
                        bucket: "my_bucket"
                        s3_folder: "s3/output"
        ```
        """
        super().__init__(input, output, state, **kwargs)
        self.log = setup_logger(self.state)
        self.classes: Dict[int, str] = {}

    def predict(self, classes: str, model_path: str, use_cuda: bool = False) -> None:
        """
        📖 Classify images in the input sub-folders using a pre-trained PyTorch model.

        Args:
            classes (str): JSON string mapping class indices to labels.
            model_path (str): Path to the pre-trained PyTorch model.
            use_cuda (bool): Whether to use CUDA for model inference. Default is False.

        This method iterates through each image file in the specified sub-folders, applies the model,
        and classifies the image. The classified images are then saved in an output folder, organized by their predicted labels.
        """
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

        self.classes = json.loads(classes)
        input_folder = self.input.input_folder
        input_folders = os.listdir(input_folder)

        # Load the model
        model = torch.load(model_path).to(device)
        model.eval()

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        for folder in input_folders:
            for image_file in os.listdir(folder):
                if not image_file.lower().endswith(("png", "jpg", "jpeg")):
                    continue

                image_path = os.path.join(folder, image_file)
                image = Image.open(image_path).convert("RGB")
                image = transform(image).to(device)
                image = image.unsqueeze(0)

                with torch.no_grad():
                    output = model(image)
                    _, predicted = torch.max(output, 1)
                    label = self.get_label(predicted.item())

                output_folder = os.path.join(self.output.output_folder, label)
                os.makedirs(output_folder, exist_ok=True)

                output_path = os.path.join(output_folder, image_file)
                Image.open(image_path).save(output_path)

                self.log.info(f"Classified {image_file} as {label}")

    def get_label(self, class_idx: int) -> str:
        """
        📖 Get the label corresponding to the class index.

        Args:
            class_idx (int): The class index.

        Returns:
            str: The label corresponding to the class index.

        This method returns the label that corresponds to a given class index based on the `classes` dictionary.
        """
        return self.classes.get(class_idx, "Unknown")
