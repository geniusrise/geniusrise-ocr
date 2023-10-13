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
import mobi
from typing import Optional
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise import Bolt


class ParseMOBI(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `ParseMOBI` class is designed to process MOBI files.
        It takes an input folder containing MOBI files as an argument and iterates through each file.
        For each file, it extracts the images and saves them in a designated output folder.

        Args:
            input (BatchInput): An instance of the BatchInput class for reading the data.
            output (BatchOutput): An instance of the BatchOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius ParseMOBI rise \
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
            parse_mobi:
                name: "ParseMOBI"
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
        📖 Process MOBI files in the given input folder and extract images.

        Args:
            input_folder (str): The folder containing MOBI files to process.

        This method iterates through each MOBI file in the specified folder and extracts the images.
        """
        input_folder = input_folder if input_folder else self.input.input_folder

        for mobi_file in os.listdir(input_folder):
            if mobi_file.lower().endswith(".mobi"):
                self._process_mobi_file(os.path.join(input_folder, mobi_file))

    def _process_mobi_file(self, mobi_path: str) -> None:
        """
        📖 Process a MOBI file and save its images.

        Args:
            mobi_path (str): The path to the MOBI file.

        This method extracts each image from the MOBI file and saves it in a folder within the output folder.
        """
        with open(mobi_path, "rb") as f:
            reader = mobi.Mobi(f)
            book = reader.read()

        image_folder = os.path.join(self.output.output_folder, os.path.basename(mobi_path).replace(".mobi", ""))
        os.makedirs(image_folder, exist_ok=True)

        for img_index, img_data in enumerate(book.images):
            image_filename = f"img_{img_index + 1}.png"
            image_path = os.path.join(image_folder, image_filename)

            with open(image_path, "wb") as img_file:
                img_file.write(img_data)

        self.log.info(f"Processed MOBI file: {os.path.basename(mobi_path)}")
