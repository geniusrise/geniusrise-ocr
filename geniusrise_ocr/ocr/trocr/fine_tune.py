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

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise import Bolt
from geniusrise_ocr.ocr.dataset import CustomOCRDataset


class FineTuneTROCR(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `FineTuneTROCR` class is designed to fine-tune the TROCR model on a custom OCR dataset.
        It supports three popular OCR dataset formats: COCO, ICDAR, and SynthText.

        Args:
            input (BatchInput): An instance of the BatchInput class for reading the data.
            output (BatchOutput): An instance of the BatchOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        Dataset Formats:
            - COCO: Assumes a folder structure with an 'annotations.json' file containing image and text annotations.
            - ICDAR: Assumes a folder structure with 'Images' and 'Annotations' folders containing image files and XML annotation files respectively.
            - SynthText: Assumes a folder with image files and corresponding '.txt' files containing ground truth text.

        ## Using geniusrise to invoke via command line
        ```bash
        genius FineTuneTROCR rise \
            batch \
                --bucket my_bucket \
                --s3_folder s3/input \
            batch \
                --bucket my_bucket \
                --s3_folder s3/output \
            none \
            process \
                --args epochs=3 batch_size=32 learning_rate=0.001 dataset_format=coco use_cuda=true
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            fine_tune_trocr:
                name: "FineTuneTROCR"
                method: "process"
                args:
                    epochs: 3
                    batch_size: 32
                    learning_rate: 0.001
                    dataset_format: coco
                    use_cuda: true
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

    def process(
        self, epochs: int, batch_size: int, learning_rate: float, dataset_format: str, use_cuda: bool = False
    ) -> None:
        """
        📖 Fine-tune the TROCR model on a custom OCR dataset.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            dataset_format (str): Format of the OCR dataset. Supported formats are "coco", "icdar", and "synthtext".
            use_cuda (bool): Whether to use CUDA for training. Default is False.

        This method fine-tunes the TROCR model using the images and annotations in the dataset specified by `dataset_format`.
        The fine-tuned model is saved to the specified output path.
        """
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        dataset = CustomOCRDataset(root_dir=self.input.input_folder, transform=transform, dataset_format=dataset_format)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed").to(device)
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            for batch_idx, (images, texts) in enumerate(dataloader):
                self.log.debug(f"Processing batch {batch_idx}")

                images = images.to(device)
                inputs = processor(texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

                optimizer.zero_grad()
                outputs = model(images, labels=inputs)
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), inputs.view(-1))
                loss.backward()
                optimizer.step()

            self.log.info(f"Epoch {epoch + 1}/{epochs} completed.")

        model.save_pretrained(self.output.output_folder)
        self.log.info(f"Fine-tuned model saved to {self.output.output_folder}")
