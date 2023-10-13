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
import tempfile
import json
from geniusrise import BatchInput, BatchOutput, InMemoryState
from geniusrise_ocr import ParseEpub


def test_process_text_epub():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input_batch = BatchInput(bucket="geniusrise-test", s3_folder="epub/epub_input", input_folder=input_dir)
    output_batch = BatchOutput(bucket="geniusrise-test", s3_folder="epub/text_output", output_folder=output_dir)
    state = InMemoryState()
    parse_epub = ParseEpub(input=input_batch, output=output_batch, state=state)

    parse_epub.input.copy_from_remote()

    # Process a known text-based epub
    parse_epub.process()
    output_folder = parse_epub.output.output_folder

    # Check if the output JSON file exists and contains expected data
    json_file = "epub_text.json"
    json_path = os.path.join(output_folder, json_file)
    assert os.path.exists(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert all(isinstance(page, str) for page in data)
    assert data[0] == ""


def test_process_image_epub():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input_batch = BatchInput(bucket="geniusrise-test", s3_folder="epub/epub_input", input_folder=input_dir)
    output_batch = BatchOutput(bucket="geniusrise-test", s3_folder="epub/image_output", output_folder=output_dir)
    state = InMemoryState()
    parse_epub = ParseEpub(input=input_batch, output=output_batch, state=state)

    parse_epub.input.copy_from_remote()

    # Process a known image-based epub
    parse_epub.process()
    output_folder = parse_epub.output.output_folder

    # Check if the output image folder exists and contains PNG files
    image_folder = "epub_images"
    image_folder_path = os.path.join(output_folder, image_folder)
    assert os.path.exists(image_folder_path)

    image_files = os.listdir(image_folder_path)
    assert all(file.endswith(".png") for file in image_files)
