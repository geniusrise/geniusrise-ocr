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
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise import Bolt
import torch


class TROCRImageOCR(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `TROCRImageOCR` class performs OCR (Optical Character Recognition) on images using Microsoft's TROCR model.
        It expects the `input.input_folder` to contain the images for OCR and saves the OCR results as JSON files in `output.output_folder`.

        Args:
            input (BatchInput): Instance of BatchInput for reading data.
            output (BatchOutput): Instance of BatchOutput for saving data.
            state (State): Instance of State for maintaining state.
            **kwargs: Additional keyword arguments.

        ## Command Line Invocation with geniusrise
        ```bash
        genius TROCRImageOCR rise \
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
                name: "TROCRImageOCR"
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
        self.log = setup_logger(self.state)

    def process(self, kind: str = "printed", use_cuda: bool = True) -> None:
        """
        📖 Perform OCR on images in the input folder and save the OCR results as JSON files in the output folder.

        This method iterates through each image file in `input.input_folder`, performs OCR using the TROCR model,
        and saves the OCR results as JSON files in `output.output_folder`.

        Args:
            kind (str): The kind of TROCR model to use. Default is "printed". Options are "printed" or "handwritten".
            use_cuda (bool): Whether to use CUDA for model inference. Default is True.
        """
        device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"

        self.processor = TrOCRProcessor.from_pretrained(f"microsoft/trocr-large-{kind}")
        self.model = VisionEncoderDecoderModel.from_pretrained(f"microsoft/trocr-large-{kind}").to(device)

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
