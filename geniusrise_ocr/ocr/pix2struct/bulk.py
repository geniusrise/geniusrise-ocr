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

import json
import os

import torch
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor


class Pix2StructImageOCR(Bolt):
    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        model_name: str = "google/pix2struct-large",
        **kwargs,
    ) -> None:
        r"""
        The `Pix2StructImageOCR` class performs OCR on images using Google's Pix2Struct model.
        It expects the `input.input_folder` to contain the images for OCR and saves the OCR results as JSON files in `output.output_folder`.

        Args:
            input (BatchInput): Instance of BatchInput for reading data.
            output (BatchOutput): Instance of BatchOutput for saving data.
            state (State): Instance of State for maintaining state.
            model_name (str): The name of the Pix2Struct model to use. Default is "google/pix2struct-large".
            **kwargs: Additional keyword arguments.

        ## Command Line Invocation with geniusrise
        ```bash
        genius Pix2StructImageOCR rise \
            batch \
                --bucket my_bucket \
                --s3_folder s3/input \
            batch \
                --bucket my_bucket \
                --s3_folder s3/output \
            none \
            process
        ```

        ## YAML Configuration with geniusrise
        ```yaml
        version: "1"
        spouts:
            ocr_processing:
                name: "Pix2StructImageOCR"
                method: "process"
                input:
                    type: "batch"
                    args:
                        bucket: "my_bucket"
                        s3_folder: "s3/input"
                        use_cuda: true
                output:
                    type: "batch"
                    args:
                        bucket: "my_bucket"
                        s3_folder: "s3/output"
                        use_cuda: true
        ```
        """
        super().__init__(input, output, state, **kwargs)
        self.model_name = model_name
        self.log = setup_logger(self.state)

    def process(self, use_cuda: bool = True) -> None:
        """
        📖 Perform OCR on images in the input folder and save the OCR results as JSON files in the output folder.

        Args:
            use_cuda (bool): Whether to use CUDA for model inference. Default is True.
        """
        device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"

        self.processor = Pix2StructProcessor.from_pretrained(self.model_name)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(self.model_name).to(device)

        input_folder = self.input.input_folder

        for image_file in os.listdir(input_folder):
            if not image_file.lower().endswith(("png", "jpg", "jpeg")):
                continue

            image_path = os.path.join(input_folder, image_file)
            image = Image.open(image_path)

            # Perform OCR
            inputs = self.processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = self.model.generate(**inputs)
            ocr_result = self.processor.batch_decode(out, skip_special_tokens=True)[0]

            # Save OCR result
            json_file = f"{os.path.splitext(image_file)[0]}.json"
            json_path = os.path.join(self.output.output_folder, json_file)
            with open(json_path, "w") as f:
                json.dump({"ocr_text": ocr_result}, f)

            self.log.info(f"Processed OCR for {image_file}")
