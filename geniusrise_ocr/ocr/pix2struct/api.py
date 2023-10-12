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

import io
import base64
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise import Bolt
import torch
import cherrypy
from typing import Dict, Any, Optional


class Pix2StructImageOCRAPI(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        r"""
        The `Pix2StructImageOCRAPI` class performs OCR on images using Google's Pix2Struct model.
        The class exposes an API endpoint for OCR on single images. The endpoint is accessible at `/api/v1/ocr`.
        The API takes a POST request with a JSON payload containing a base64 encoded image under the key `image_base64`.
        It returns a JSON response containing the OCR result under the key `ocr_text`.

        Args:
            input (BatchInput): Instance of BatchInput for reading data.
            output (BatchOutput): Instance of BatchOutput for saving data.
            state (State): Instance of State for maintaining state.
            model_name (str): The name of the Pix2Struct model to use. Default is "google/pix2struct-large".
            **kwargs: Additional keyword arguments.

        ## Command Line Invocation with geniusrise
        ```bash
        genius Pix2StructImageOCRAPI rise \
            batch \
                --bucket my_bucket \
                --s3_folder s3/input \
            batch \
                --bucket my_bucket \
                --s3_folder s3/output \
            none \
            listen \
                --args endpoint=* port=3000 cors_domain=* use_cuda=True
        ```

        ## YAML Configuration with geniusrise
        ```yaml
        version: "1"
        spouts:
            ocr_processing:
                name: "Pix2StructImageOCRAPI"
                method: "listen"
                args:
                    endpoint: *
                    port: 3000
                    cors_domain: *
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
        """
        super().__init__(input, output, state, **kwargs)
        self.log = setup_logger(self.state)

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
        image = Image.open(io.BytesIO(image_bytes))

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = Pix2StructProcessor.from_pretrained(self.model_name)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(self.model_name).to(device)

        inputs = self.processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model.generate(**inputs)
        ocr_result = self.processor.batch_decode(out, skip_special_tokens=True)[0]

        return {"ocr_text": ocr_result}

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
        model_name: str = "google/pix2struct-large",
        use_cuda: bool = True,
    ) -> None:
        device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.processor = Pix2StructProcessor.from_pretrained(self.model_name)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(self.model_name).to(device)

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
        cherrypy.tree.mount(self, "/api/v1/", {"/": {"tools.CORS.on": True}})
        cherrypy.tools.CORS = cherrypy.Tool("before_finalize", CORS)
        cherrypy.engine.start()
        cherrypy.engine.block()
