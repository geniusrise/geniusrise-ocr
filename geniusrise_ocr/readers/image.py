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
        super().__init__(input, output, state, **kwargs)
        self.log = setup_logger(self.state)

    def process(
        self, input_folder: str, output_format: str, quality: Optional[int] = None, subsampling: Optional[int] = 0
    ) -> None:
        """
        Convert images in the given input folder to the specified output format.

        Args:
            input_folder (str): The folder containing images to convert.
            output_format (str): The format to convert images to ('PNG' or 'JPG').
            quality (Optional[int]): The quality of the output image for lossy formats like 'JPG'.
            subsampling (Optional[int]): The subsampling factor for JPEG compression.
        """
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
