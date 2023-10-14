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
from typing import Any, Dict, Optional
import tempfile

import cherrypy
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger


class DoctrImageOCRAPI(Bolt):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        """
        The `DoctrImageOCRAPI` class performs OCR (Optical Character Recognition) on images using Doctr.
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
        genius DoctrImageOCR rise \
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
                name: "DoctrImageOCR"
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

        # Create a temporary file to load the image
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            temp_file.write(image_bytes)
            temp_file.flush()
            single_img_doc = DocumentFile.from_images(temp_file.name)

        # Perform OCR using Doctr
        ocr_result = self.model(single_img_doc)

        # Extract text from OCR result
        _res = ocr_result.export()
        ocr_text = [
            word["value"]
            for page in _res["pages"]
            for block in page["blocks"]
            for line in block["lines"]
            for word in line["words"]
        ]

        return {"ocr_text": " ".join(ocr_text), "bound": _res}

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
        use_cuda: bool = True,
        detection_model="db_resnet50",
        recognition_model="crnn_vgg16_bn",
        **kwargs,
    ) -> None:
        self.model = ocr_predictor(
            pretrained=True,
            det_arch=detection_model,
            reco_arch=recognition_model,
        )

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
