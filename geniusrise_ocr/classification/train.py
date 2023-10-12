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
import torch
import torch.nn as nn
import os
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from geniusrise.core.data import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise import Bolt


class TrainImageClassifier(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `TrainImageClassifier` class is designed to train an image classifier using a ResNet-152 model.
        It assumes that the `input.input_folder` contains sub-folders named 'train' and 'test'.
        Each of these sub-folders should contain class-specific folders with images.
        The trained model is saved to the specified output path.

        Args:
            input (BatchInput): An instance of the BatchInput class for reading the data.
            output (BatchOutput): An instance of the BatchOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius TrainImageClassifier rise \
            batch \
                --bucket my_bucket \
                --s3_folder s3/input \
            batch \
                --bucket my_bucket \
                --s3_folder s3/output \
            none \
            process \
                --args output_model_path=/path/to/model.pth num_classes=4 epochs=10 batch_size=32 learning_rate=0.001
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            image_training:
                name: "TrainImageClassifier"
                method: "process"
                args:
                    output_model_path: "/path/to/model.pth"
                    num_classes: 4
                    epochs: 10
                    batch_size: 32
                    learning_rate: 0.001
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
        self,
        output_model_path: str,
        num_classes: int = 4,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> None:
        """
        📖 Train an image classifier using a ResNet-152 model.

        Args:
            output_model_path (str): Path to save the trained model.
            num_classes (int): Number of classes of the images.
            epochs (Optional[int]): Number of training epochs. Default is 10.
            batch_size (Optional[int]): Batch size for training. Default is 32.
            learning_rate (Optional[float]): Learning rate for the optimizer. Default is 0.001.

        This method trains a ResNet-152 model using the images in the 'train' and 'test' sub-folders
        of `input.input_folder`. Each of these sub-folders should contain class-specific folders with images.
        The trained model is saved to the specified output path.
        """
        # Data transformations
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # Prepare datasets and dataloaders
        train_folder = os.path.join(self.input.input_folder, "train")
        test_folder = os.path.join(self.input.input_folder, "test")

        train_dataset = datasets.ImageFolder(train_folder, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = datasets.ImageFolder(test_folder, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model
        model = models.resnet152(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            model.train()
            for _, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            self.log.info(f"Epoch {epoch+1}/{epochs} completed.")

        # Save the trained model
        torch.save(model, output_model_path)
        self.log.info(f"Model saved to {output_model_path}")