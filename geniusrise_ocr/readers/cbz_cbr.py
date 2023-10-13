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
import zipfile
import rarfile
from typing import Optional
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise import Bolt


class ParseCBZCBR(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `ParseCBZCBR` class is designed to process CBZ and CBR files, which are commonly used for comic books.
        It takes an input folder containing CBZ/CBR files as an argument and iterates through each file.
        For each file, it extracts the images and saves them in a designated output folder.

        Args:
            input (BatchInput): An instance of the BatchInput class for reading the data.
            output (BatchOutput): An instance of the BatchOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius ParseCBZCBR rise \
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
            parse_cbzcbr:
                name: "ParseCBZCBR"
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
        📖 Process CBZ and CBR files in the given input folder and extract images.

        Args:
            input_folder (str): The folder containing CBZ/CBR files to process.

        This method iterates through each CBZ/CBR file in the specified folder and extracts the images.
        """
        input_folder = input_folder if input_folder else self.input.input_folder

        for comic_file in os.listdir(input_folder):
            if comic_file.lower().endswith(".cbz"):
                self._process_cbz_file(os.path.join(input_folder, comic_file))
            elif comic_file.lower().endswith(".cbr"):
                self._process_cbr_file(os.path.join(input_folder, comic_file))

    def _process_cbz_file(self, cbz_path: str) -> None:
        """
        📖 Process a CBZ file and save its images.

        Args:
            cbz_path (str): The path to the CBZ file.

        This method extracts each image from the CBZ file and saves it in a folder within the output folder.
        """
        with zipfile.ZipFile(cbz_path, "r") as zip_ref:
            image_folder = os.path.join(self.output.output_folder, os.path.basename(cbz_path).replace(".cbz", ""))
            os.makedirs(image_folder, exist_ok=True)

            for file_info in zip_ref.infolist():
                if file_info.filename.lower().endswith(("jpg", "jpeg", "png", "gif", "bmp")):
                    zip_ref.extract(file_info, image_folder)

        self.log.info(f"Processed CBZ file: {os.path.basename(cbz_path)}")

    def _process_cbr_file(self, cbr_path: str) -> None:
        """
        📖 Process a CBR file and save its images.

        Args:
            cbr_path (str): The path to the CBR file.

        This method extracts each image from the CBR file and saves it in a folder within the output folder.
        """
        with rarfile.RarFile(cbr_path, "r") as rar_ref:
            image_folder = os.path.join(self.output.output_folder, os.path.basename(cbr_path).replace(".cbr", ""))
            os.makedirs(image_folder, exist_ok=True)

            for file_info in rar_ref.infolist():
                if file_info.filename.lower().endswith(("jpg", "jpeg", "png", "gif", "bmp")):
                    rar_ref.extract(file_info, image_folder)

        self.log.info(f"Processed CBR file: {os.path.basename(cbr_path)}")
