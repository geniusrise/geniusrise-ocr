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
from typing import List, Optional
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise import Bolt


class ParsePdf(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `ParsePdf` class is designed to process PDF files and classify them as either text-based or image-based.
        It takes an input folder containing PDF files as an argument and iterates through each file.
        For each PDF, it samples a few pages to determine the type of content it primarily contains.
        If the PDF is text-based, the class extracts the text from each page and saves it as a JSON file.
        If the PDF is image-based, it converts each page to a PNG image and saves them in a designated output folder.

            Args:
                input (BatchInput): An instance of the BatchInput class for reading the data.
                output (BatchOutput): An instance of the BatchOutput class for saving the data.
                state (State): An instance of the State class for maintaining the state.
                **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius ParsePdf rise \
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
            parse_pdfs:
                name: "ParsePdf"
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
                        s3_folder: "s3/outupt"
        ```
        """
        super().__init__(input, output, state, **kwargs)
        self.log = setup_logger(self.state)

    def process(self, input_folder: Optional[str] = None) -> None:
        """
        📖 Process PDF files in the given input folder and classify them as text-based or image-based.

        Args:
            input_folder (str): The folder containing PDF files to process.

        This method iterates through each PDF file in the specified folder, reads a sample of pages,
        and determines whether the PDF is text-based or image-based. It then delegates further processing
        to `_process_text_pdf` or `_process_image_pdf` based on this determination.
        """
        input_folder = input_folder if input_folder else os.path.join(self.input.input_folder, self.input.s3_folder)

        for pdf_file in os.listdir(input_folder):
            if not pdf_file.endswith(".pdf"):
                continue

            pdf_path = os.path.join(input_folder, pdf_file)
            pdf_reader = PdfReader(open(pdf_path, "rb"))
            total_pages = len(pdf_reader.pages)

            # Randomly sample 3 pages to determine PDF type
            sample_pages = random.sample(range(total_pages), min(3, total_pages))

            text_count = 0
            image_count = 0

            for page_num in sample_pages:
                page = pdf_reader.pages[page_num]
                text_content = page.extract_text()

                if text_content.strip():
                    text_count += 1
                else:
                    image_count += 1

            # Determine PDF type based on sampled pages
            if text_count > image_count:
                self._process_text_pdf(pdf_path, pdf_file)
            else:
                self._process_image_pdf(pdf_path, pdf_file)

    def _process_text_pdf(self, pdf_path: str, pdf_file: str) -> None:
        """
        📖 Process a text-based PDF file and save its content as a JSON file.

        Args:
            pdf_path (str): The path to the PDF file.
            pdf_file (str): The name of the PDF file.

        This method reads each page of the text-based PDF, extracts the text content, and saves it
        as a JSON file in the output folder.
        """
        pdf_reader = PdfReader(open(pdf_path, "rb"))
        total_pages = len(pdf_reader.pages)
        text_data: List[str] = []

        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            text_content = page.extract_text()
            text_data.append(text_content.strip())

        json_file = pdf_file.replace(".pdf", ".json")
        json_path = os.path.join(self.output.output_folder, json_file)
        with open(json_path, "w") as f:
            json.dump(text_data, f)

        self.log.info(f"Processed text PDF: {pdf_file}")

    def _process_image_pdf(self, pdf_path: str, pdf_file: str) -> None:
        """
        📖 Process an image-based PDF file and save its pages as PNG images.

        Args:
            pdf_path (str): The path to the PDF file.
            pdf_file (str): The name of the PDF file.

        This method converts each page of the image-based PDF to a PNG image and saves it
        in a folder within the output folder.
        """
        images = convert_from_path(pdf_path)
        image_folder = os.path.join(self.output.output_folder, pdf_file.replace(".pdf", ""))
        os.makedirs(image_folder, exist_ok=True)

        for i, image in enumerate(images):
            image_path = os.path.join(image_folder, f"page_{i + 1}.png")
            image.save(image_path, "PNG")

        self.log.info(f"Processed image PDF: {pdf_file}")
