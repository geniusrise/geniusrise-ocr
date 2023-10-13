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
from typing import List, Optional

import ebooklib
from ebooklib import epub
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger


class ParseEpub(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `ParseEpub` class is designed to process EPUB files and classify them as either text-based or image-based.
        It takes an input folder containing EPUB files as an argument and iterates through each file.
        For each EPUB, it samples a few items to determine the type of content it primarily contains.
        If the EPUB is text-based, the class extracts the text from each item and saves it as a JSON file.
        If the EPUB is image-based, it saves the images in a designated output folder.

        Args:
            input (BatchInput): An instance of the BatchInput class for reading the data.
            output (BatchOutput): An instance of the BatchOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius ParseEpub rise \
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
        📖 Process EPUB files in the given input folder and classify them as text-based or image-based.

        Args:
            input_folder (str): The folder containing EPUB files to process.

        This method iterates through each EPUB file in the specified folder, reads a sample of items,
        and determines whether the EPUB is text-based or image-based. It then delegates further processing
        to `_process_text_epub` or `_process_image_epub` based on this determination.
        """
        input_folder = input_folder if input_folder else self.input.input_folder

        for epub_file in os.listdir(input_folder):
            if not epub_file.endswith(".epub"):
                continue

            epub_path = os.path.join(input_folder, epub_file)
            epub_book = epub.read_epub(epub_path)

            text_count = 0
            image_count = 0

            for item in list(epub_book.get_items())[:10]:  # Sample first 10 items
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    text_count += 1
                elif item.get_type() == ebooklib.ITEM_IMAGE:
                    image_count += 1

            if text_count > image_count:
                self._process_text_epub(epub_path, epub_file)
            else:
                self._process_image_epub(epub_path, epub_file)

    def _process_text_epub(self, epub_path: str, epub_file: str) -> None:
        """
        📖 Process a text-based EPUB file and save its content as a JSON file.

        Args:
            epub_path (str): The path to the EPUB file.
            epub_file (str): The name of the EPUB file.

        This method reads each item of the text-based EPUB, extracts the text content, and saves it
        as a JSON file in the output folder.
        """
        epub_book = epub.read_epub(epub_path)
        text_data: List[str] = []

        for item in epub_book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            text_data.append(item.content.decode("utf-8").strip())

        json_file = epub_file.replace(".epub", ".json")
        json_path = os.path.join(self.output.output_folder, json_file)
        with open(json_path, "w") as f:
            json.dump(text_data, f)

        self.log.info(f"Processed text EPUB: {epub_file}")

    def _process_image_epub(self, epub_path: str, epub_file: str) -> None:
        """
        📖 Process an image-based EPUB file and save its images.

        Args:
            epub_path (str): The path to the EPUB file.
            epub_file (str): The name of the EPUB file.

        This method saves each image of the image-based EPUB in a folder within the output folder.
        """
        epub_book = epub.read_epub(epub_path)
        image_folder = os.path.join(self.output.output_folder, epub_file.replace(".epub", ""))
        os.makedirs(image_folder, exist_ok=True)

        for i, item in enumerate(epub_book.get_items_of_type(ebooklib.ITEM_IMAGE)):
            image_path = os.path.join(image_folder, f"image_{i + 1}.png")
            with open(image_path, "wb") as f:
                f.write(item.content)

        self.log.info(f"Processed image EPUB: {epub_file}")
