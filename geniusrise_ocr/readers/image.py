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

from PIL import Image
import os
from typing import Optional
from geniusrise.core.data import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise import Bolt


class ConvertImage(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `ConvertImage` class is designed to convert images from one format to another.
        It takes an input folder containing images and an output format as arguments.
        The class iterates through each image file in the specified folder and converts it to the desired format.
        Additional options like quality and subsampling can be specified for lossy formats like 'JPG'.

        Args:
            input (BatchInput): An instance of the BatchInput class for reading the data.
            output (BatchOutput): An instance of the BatchOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius ConvertImage rise \
            batch \
                --bucket my_bucket \
                --s3_folder s3/input \
            batch \
                --bucket my_bucket \
                --s3_folder s3/output \
            none \
            process \
                --args input_folder=/path/to/image/folder output_format=PNG quality=95 subsampling=0
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            convert_images:
                name: "ConvertImage"
                method: "process"
                args:
                    output_format: "PNG"
                    quality: 95
                    subsampling: 0
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

    def process(self, output_format: str, quality: Optional[int] = None, subsampling: Optional[int] = 0) -> None:
        """
        📖 Convert images in the given input folder to the specified output format.

        Args:
            output_format (str): The format to convert images to ('PNG' or 'JPG').
            quality (Optional[int]): The quality of the output image for lossy formats like 'JPG'. Defaults to None.
            subsampling (Optional[int]): The subsampling factor for JPEG compression. Defaults to 0.

        This method iterates through each image file in the specified folder, reads the image,
        and converts it to the specified output format. Additional parameters like quality and subsampling
        can be set for lossy formats.
        """
        input_folder = self.input.input_folder

        for image_file in os.listdir(input_folder):
            file_extension = image_file.lower().split(".")[-1]

            if file_extension not in ("png", "jpg", "jpeg"):
                continue

            image_path = os.path.join(input_folder, image_file)
            image = Image.open(image_path)

            output_file = f"{os.path.splitext(image_file)[0]}.{output_format.lower()}"
            output_path = os.path.join(self.output.output_folder, output_file)

            # If the image is already in the desired format, save it to the output folder
            if file_extension == output_format.lower():
                image.save(output_path)
                self.log.info(f"Saved {image_file} to {output_file} without conversion.")
                continue

            if output_format.upper() == "PNG":
                image.save(output_path, "PNG")
            elif output_format.upper() == "JPG":
                image.save(output_path, "JPEG", quality=quality if quality else 95, subsampling=subsampling)

            self.log.info(f"Converted {image_file} to {output_file}")
