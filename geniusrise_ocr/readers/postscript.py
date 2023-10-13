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
import random
import subprocess
from typing import List, Optional
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise import Bolt


class ParsePostScript(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `ParsePostScript` class is designed to process PostScript files and classify them as either text-based or image-based.
        It takes an input folder containing PostScript files as an argument and iterates through each file.
        For each PostScript file, it converts it to PDF and samples a few pages to determine the type of content it primarily contains.
        If the PostScript is text-based, the class extracts the text from each page and saves it as a JSON file.
        If the PostScript is image-based, it converts each page to a PNG image and saves them in a designated output folder.

        Args:
            input (BatchInput): An instance of the BatchInput class for reading the data.
            output (BatchOutput): An instance of the BatchOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius ParsePostScript rise \
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
        📖 Process PostScript files in the given input folder and classify them as text-based or image-based.

        Args:
            input_folder (str): The folder containing PostScript files to process.

        This method iterates through each PostScript file in the specified folder, converts it to PDF,
        reads a sample of pages, and determines whether the PostScript is text-based or image-based.
        It then delegates further processing to `_process_text_ps` or `_process_image_ps` based on this determination.
        """
        input_folder = input_folder if input_folder else self.input.input_folder

        for ps_file in os.listdir(input_folder):
            if not ps_file.endswith(".ps"):
                continue

            ps_path = os.path.join(input_folder, ps_file)
            pdf_path = os.path.join(input_folder, ps_file.replace(".ps", ".pdf"))

            # Convert PostScript to PDF
            subprocess.run(["ps2pdf", ps_path, pdf_path])

            pdf_reader = PdfFileReader(open(pdf_path, "rb"))
            total_pages = pdf_reader.getNumPages()

            # Randomly sample 3 pages to determine PostScript type
            sample_pages = random.sample(range(total_pages), min(3, total_pages))

            text_count = 0
            image_count = 0

            for page_num in sample_pages:
                page = pdf_reader.getPage(page_num)
                text_content = page.extract_text()

                if text_content.strip():
                    text_count += 1
                else:
                    image_count += 1

            if text_count > image_count:
                self._process_text_ps(pdf_path, ps_file)
            else:
                self._process_image_ps(pdf_path, ps_file)

    def _process_text_ps(self, pdf_path: str, ps_file: str) -> None:
        """
        📖 Process a text-based PostScript file and save its content as a JSON file.

        Args:
            pdf_path (str): The path to the converted PDF file.
            ps_file (str): The name of the PostScript file.

        This method reads each page of the text-based PostScript, extracts the text content, and saves it
        as a JSON file in the output folder.
        """
        pdf_reader = PdfFileReader(open(pdf_path, "rb"))
        total_pages = pdf_reader.getNumPages()
        text_data: List[str] = []

        for page_num in range(total_pages):
            page = pdf_reader.getPage(page_num)
            text_content = page.extract_text()
            text_data.append(text_content.strip())

        json_file = ps_file.replace(".ps", ".json")
        json_path = os.path.join(self.output.output_folder, json_file)
        with open(json_path, "w") as f:
            json.dump(text_data, f)

        self.log.info(f"Processed text PostScript: {ps_file}")

    def _process_image_ps(self, pdf_path: str, ps_file: str) -> None:
        """
        📖 Process an image-based PostScript file and save its pages as PNG images.

        Args:
            pdf_path (str): The path to the converted PDF file.
            ps_file (str): The name of the PostScript file.

        This method converts each page of the image-based PostScript to a PNG image and saves it
        in a folder within the output folder.
        """
        images = convert_from_path(pdf_path)
        image_folder = os.path.join(self.output.output_folder, ps_file.replace(".ps", ""))
        os.makedirs(image_folder, exist_ok=True)

        for i, image in enumerate(images):
            image_path = os.path.join(image_folder, f"page_{i + 1}.png")
            image.save(image_path, "PNG")

        self.log.info(f"Processed image PostScript: {ps_file}")
