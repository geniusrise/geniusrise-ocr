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
import fitz  # PyMuPDF
from typing import Optional
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise import Bolt


class ParseXPS(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `ParseXPS` class is designed to process XPS files.
        It takes an input folder containing XPS files as an argument and iterates through each file.
        For each file, it extracts the images and saves them in a designated output folder.

        Args:
            input (BatchInput): An instance of the BatchInput class for reading the data.
            output (BatchOutput): An instance of the BatchOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius ParseXPS rise \
            batch \
                --bucket my_bucket \
                --s3_folder s3/input \
            batch \
                --bucket my_bucket \
                --s3_folder s3/output \
            none \
            process
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            parse_xps:
                name: "ParseXPS"
                method: "process"
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

    def process(self, input_folder: Optional[str] = None) -> None:
        """
        📖 Process XPS files in the given input folder and extract images.

        Args:
            input_folder (str): The folder containing XPS files to process.

        This method iterates through each XPS file in the specified folder and extracts the images.
        """
        input_folder = input_folder if input_folder else self.input.input_folder

        for xps_file in os.listdir(input_folder):
            if xps_file.lower().endswith(".xps") or xps_file.lower().endswith(".oxps"):
                self._process_xps_file(os.path.join(input_folder, xps_file))

    def _process_xps_file(self, xps_path: str) -> None:
        """
        📖 Process an XPS file and save its images.

        Args:
            xps_path (str): The path to the XPS file.

        This method extracts each image from the XPS file and saves it in a folder within the output folder.
        """
        doc = fitz.open(xps_path)
        image_folder = os.path.join(
            self.output.output_folder, os.path.basename(xps_path).replace(".xps", "").replace(".oxps", "")
        )
        os.makedirs(image_folder, exist_ok=True)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                image_path = os.path.join(image_folder, image_filename)

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

        self.log.info(f"Processed XPS file: {os.path.basename(xps_path)}")
