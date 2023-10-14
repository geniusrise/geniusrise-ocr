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

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import cherrypy
import cv2
import numpy as np
import torch
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TROCRImageOCRAPI(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `TROCRImageOCR` class performs OCR (Optical Character Recognition) on images using Microsoft's TROCR model.
        The class exposes an API endpoint for OCR on single images. The endpoint is accessible at `/api/v1/ocr`.
        The API takes a POST request with a JSON payload containing a base64 encoded image under the key `image_base64`.
        It returns a JSON response containing the OCR result under the key `ocr_text`.

        Args:
            input (BatchInput): Instance of BatchInput for reading data.
            output (BatchOutput): Instance of BatchOutput for saving data.
            state (State): Instance of State for maintaining state.
            **kwargs: Additional keyword arguments.

        ## Command Line Invocation with geniusrise
        ```bash
        genius TROCRImageOCR rise \
            batch \
                --bucket my_bucket \
                --s3_folder s3/input \
            batch \
                --bucket my_bucket \
                --s3_folder s3/output \
            none \
            listen \
                --args endpoint=* port=3000 cors_domain=* kind=handwriting use_cuda=True
        ```

        ## YAML Configuration with geniusrise
        ```yaml
        version: "1"
        spouts:
            ocr_processing:
                name: "TROCRImageOCR"
                method: "listen"
                args:
                    endpoint: *
                    port: 3000
                    cors_domain: *
                    kind: handwriting
                    use_cuda: true
                input:
                    type: "batch"
                    args:
                        bucket: "my_bucket"
                        s3_folder: "s3/input"
                        use_cuda: true
                output:
                    type: "batch"
                    args:
                        bucket: "my_bucket"
                        s3_folder: "s3/output"
                        use_cuda: true
        ```


        ### API Example
        ```bash
        curl -X POST "http://localhost:3000/api/v1/ocr" -H "Content-Type: application/json" -d '{"image_base64": "your_base64_encoded_image_here"}'
        ```
        """
        super().__init__(input, output, state, **kwargs)
        self.log = setup_logger(self.state)

    def preprocess_and_detect_boxes(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Preprocess the image and detect text bounding boxes using the EAST model.

        Args:
            image (Image.Image): PIL Image object.

        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes (x, y, w, h).
        """
        # Get image dimensions and adjust to nearest multiple of 32
        h, w = image.size
        h = (h // 32) * 32
        w = (w // 32) * 32

        # Convert image to array
        img_array = np.array(image)

        # Load the EAST model
        net = cv2.dnn.readNet("models/frozen_east_text_detection.pb")
        blob = cv2.dnn.blobFromImage(
            img_array,
            1.0,
            (w, h),
            (123.68, 116.78, 103.94),  # This is R-G-B for Imagenet
            True,
            False,
        )
        net.setInput(blob)
        scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

        # Post-processing to get bounding boxes
        bounding_boxes = []
        min_score = 0.99  # Minimum confidence score
        min_area = 100  # Minimum bounding box area

        for i in range(scores.shape[2]):
            for j in range(scores.shape[3]):
                if scores[0, 0, i, j] < min_score:
                    continue
                offset_x, offset_y = j * 4.0, i * 4.0
                angle = geometry[0, 4, i, j]
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                h = geometry[0, 0, i, j] + geometry[0, 2, i, j]
                w = geometry[0, 1, i, j] + geometry[0, 3, i, j]
                x = int(offset_x - cos_a * w - sin_a * h)
                y = int(offset_y - sin_a * w + cos_a * h)

                # Filter based on size
                if w * h < min_area:
                    continue

                bounding_boxes.append((x, y, int(w), int(h)))

        return bounding_boxes

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def ocr(self, username: Optional[str] = None, password: Optional[str] = None) -> Dict[str, Any]:
        if username and password:
            self._check_auth(username=username, password=password)
        data = cherrypy.request.json
        image_base64 = data.get("image_base64", "")
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Preprocess and detect text bounding boxes
        bounding_boxes = self.preprocess_and_detect_boxes(image)

        # Draw bounding boxes on the original image
        for x, y, w, h in bounding_boxes:
            cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert the OpenCV image with bounding boxes back to PIL format
        image_with_boxes = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

        # Convert the PIL image to base64
        buffered = io.BytesIO()
        image_with_boxes.save(buffered, format="JPEG")
        image_with_boxes_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        ocr_results = []
        for box in bounding_boxes:
            x, y, w, h = box
            cropped_image = ImageOps.crop(image, (x, y, image.width - (x + w), image.height - (y + h)))

            inputs = self.processor(images=cropped_image, return_tensors="pt").to(device).pixel_values
            with torch.no_grad():
                out = self.model.generate(inputs)
            ocr_result = self.processor.batch_decode(out, skip_special_tokens=True)[0]
            ocr_results.append(ocr_result)

        return {"ocr_text": ocr_results, "image_with_boxes": image_with_boxes_base64}

    def _check_auth(self, username: str, password: str) -> None:
        auth_header = cherrypy.request.headers.get("Authorization")
        if auth_header:
            auth_decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            provided_username, provided_password = auth_decoded.split(":", 1)
            if provided_username != username or provided_password != password:
                raise cherrypy.HTTPError(401, "Unauthorized")
        else:
            raise cherrypy.HTTPError(401, "Unauthorized")

    def listen(
        self,
        endpoint: str = "*",
        port: int = 3000,
        cors_domain: str = "http://localhost:3000",
        kind: str = "printed",
        use_cuda: bool = True,
        **kwargs,
    ) -> None:
        device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"

        self.processor = TrOCRProcessor.from_pretrained(f"microsoft/trocr-large-{kind}")
        self.model = VisionEncoderDecoderModel.from_pretrained(f"microsoft/trocr-large-{kind}").to(device)

        def CORS():
            cherrypy.response.headers["Access-Control-Allow-Origin"] = cors_domain
            cherrypy.response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            cherrypy.response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            cherrypy.response.headers["Access-Control-Allow-Credentials"] = "true"

            if cherrypy.request.method == "OPTIONS":
                cherrypy.response.status = 200
                return True

        cherrypy.config.update(
            {
                "server.socket_host": "0.0.0.0",
                "server.socket_port": port,
                "log.screen": False,
                "tools.CORS.on": True,
            }
        )

        cherrypy.tools.CORS = cherrypy.Tool("before_handler", CORS)
        cherrypy.tree.mount(self, "/api/v1/trocr/", {"/": {"tools.CORS.on": True}})
        cherrypy.tools.CORS = cherrypy.Tool("before_finalize", CORS)
        cherrypy.engine.start()
        cherrypy.engine.block()
