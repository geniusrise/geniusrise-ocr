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
import subprocess
import random
from typing import List, Optional
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise import Bolt


class ParseDjvu(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `ParseDjvu` class is designed to process DJVU files and classify them as either text-based or image-based.
        It takes an input folder containing DJVU files as an argument and iterates through each file.
        For each DJVU, it samples a few pages to determine the type of content it primarily contains.
        If the DJVU is text-based, the class extracts the text from each page and saves it as a JSON file.
        If the DJVU is image-based, it converts each page to a PNG image and saves them in a designated output folder.

        Args:
            input (BatchInput): An instance of the BatchInput class for reading the data.
            output (BatchOutput): An instance of the BatchOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius ParseDjvu rise \
            batch \
                --bucket my_bucket \
                --s3_folder s3/input \
            batch \
                --bucket my_bucket \
                --s3_folder s3/output \
            none \
            process
        ```
        """
        super().__init__(input, output, state, **kwargs)
        self.log = setup_logger(self.state)

    def process(self, input_folder: Optional[str] = None) -> None:
        """
        📖 Process DJVU files in the given input folder and classify them as text-based or image-based.

        Args:
            input_folder (str): The folder containing DJVU files to process.

        This method iterates through each DJVU file in the specified folder, reads a sample of pages,
        and determines whether the DJVU is text-based or image-based. It then delegates further processing
        to `_process_text_djvu` or `_process_image_djvu` based on this determination.
        """
        input_folder = input_folder if input_folder else self.input.input_folder

        for djvu_file in os.listdir(input_folder):
            if not djvu_file.endswith(".djvu"):
                continue

            djvu_path = os.path.join(input_folder, djvu_file)

            # Count the total number of pages in the DJVU file
            page_count_command = ["djvused", "-e", "n", djvu_path]
            total_pages = int(subprocess.check_output(page_count_command).decode().strip())

            # Randomly sample 3 pages to determine DJVU type
            sample_pages = random.sample(range(1, total_pages + 1), min(3, total_pages))

            text_count = 0
            image_count = 0

            for page_num in sample_pages:
                # Extract text content from the sampled page
                text_command = ["djvutxt", "--page", str(page_num), djvu_path]
                text_content = subprocess.check_output(text_command).decode().strip()

                if text_content:
                    text_count += 1
                else:
                    image_count += 1

            if text_count > image_count:
                self._process_text_djvu(djvu_path, djvu_file)
            else:
                self._process_image_djvu(djvu_path, djvu_file)

    def _process_text_djvu(self, djvu_path: str, djvu_file: str) -> None:
        """
        📖 Process a text-based DJVU file and save its content as a JSON file.

        Args:
            djvu_path (str): The path to the DJVU file.
            djvu_file (str): The name of the DJVU file.

        This method reads each page of the text-based DJVU, extracts the text content, and saves it
        as a JSON file in the output folder.
        """
        text_data: List[str] = []
        text_command = ["djvutxt", djvu_path]
        text_content = subprocess.check_output(text_command).decode().strip()
        text_data.append(text_content)

        json_file = djvu_file.replace(".djvu", ".json")
        json_path = os.path.join(self.output.output_folder, json_file)
        with open(json_path, "w") as f:
            json.dump(text_data, f)

        self.log.info(f"Processed text DJVU: {djvu_file}")

    def _process_image_djvu(self, djvu_path: str, djvu_file: str) -> None:
        """
        📖 Process an image-based DJVU file and save its pages as PNG images.

        Args:
            djvu_path (str): The path to the DJVU file.
            djvu_file (str): The name of the DJVU file.

        This method converts each page of the image-based DJVU to a PNG image and saves it
        in a folder within the output folder.
        """
        image_folder = os.path.join(self.output.output_folder, djvu_file.replace(".djvu", ""))
        os.makedirs(image_folder, exist_ok=True)

        # Count the total number of pages in the DJVU file
        page_count_command = ["djvused", "-e", "n", djvu_path]
        total_pages = int(subprocess.check_output(page_count_command).decode().strip())

        for i in range(1, total_pages + 1):
            image_path = os.path.join(image_folder, f"page_{i}.png")
            image_command = ["ddjvu", "-format=png", "-page", str(i), djvu_path, image_path]
            subprocess.run(image_command)

        self.log.info(f"Processed image DJVU: {djvu_file}")
