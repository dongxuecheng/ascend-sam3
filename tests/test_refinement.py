"""Protocol tests with a fake native model; no NPU or OM models required.

Run: python3 -m unittest discover -s tests -v
"""
import base64
import importlib
import io
import contextlib
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError

from service.refinement import CropConfig, RefineRequest


class FakeModel:
    calls = []

    def detect_obj_refine(self, *args):
        self.calls.append(args)
        if args[0] == b"invalid":
            raise ValueError("Failed to decode image")
        if args[0] == b"failure":
            raise RuntimeError("obj-refine decoder failed")
        _, png = cv2.imencode(".png", np.array([[255, 0], [0, 255]], dtype=np.uint8))
        return {
            "results": [{"class_name": "helmet", "score": 0.9,
                         "box": {"left": 10, "top": 20, "right": 12, "bottom": 22},
                         "mask_png": png.tobytes()}],
            "refinement": {"pre_detections": 1, "candidate_crops": 1,
                           "crops_processed": 1, "limited": False,
                           "timings_ms": {"total": 10}},
        }

    def detect(self, *args):
        return []


class RefinementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_extension = types.ModuleType("ascendsam3")
        fake_extension.Sam3Model = FakeModel
        fake_extension.CropConfig = type("NativeCropConfig", (), {})
        cls.modules_patch = patch.dict(sys.modules, {"ascendsam3": fake_extension})
        cls.modules_patch.start()
        cls.service = importlib.import_module("service.main")
        cls.service._model = FakeModel()
        cls.client = TestClient(cls.service.app)  # do not run OM-loading lifespan

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.modules_patch.stop()
        sys.modules.pop("service.main", None)

    def setUp(self):
        FakeModel.calls.clear()

    def payload(self, **extra):
        value = {"image_base64": base64.b64encode(b"image").decode(),
                 "prompts": [{"text": " Person "}, {"text": "helmet"}, {"text": "HELMET"}]}
        return {**value, **extra}

    def test_label_filtering_and_defaults(self):
        req = RefineRequest(**self.payload())
        self.assertEqual(req.refine_labels(), ["helmet"])
        self.assertEqual(req.pre_detect_labels, ["person"])
        self.assertTrue(req.merge_results)
        self.assertFalse(req.return_mask)
        self.assertEqual(RefineRequest(**self.payload(pre_detect_labels=[])).pre_detect_labels, ["person"])

    def test_validation(self):
        for changes in [{"confidence_threshold": 2}, {"confidence_threshold": float("nan")},
                        {"prompts": [{"text": "helmet", "boxes": [{"bbox": [1, 2, 3, 4]}]}]}]:
            with self.assertRaises(ValidationError):
                RefineRequest(**self.payload(**changes))
        for changes in [{"padding": -1}, {"target_ar": 0}, {"max_crops": 1000}, {"w_diou": float("inf")},
                        {"unknown_parameter": True}]:
            with self.assertRaises(ValidationError):
                CropConfig(**changes)

    def test_base64_protocol_and_mask(self):
        response = self.client.post("/predict-obj-refine", json=self.payload(return_mask=True))
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["results"][0]
        self.assertEqual(result["box"], [10, 20, 12, 22])
        self.assertEqual(result["mask"], [1, 1, 4, 1])
        self.assertEqual(result["mask_width"], 2)
        self.assertEqual(FakeModel.calls[0][1:3], (["person"], ["helmet"]))
        self.assertIn("X-SAM3-Worker-PID", response.headers)

    def test_data_url_and_no_mask(self):
        payload = self.payload()
        payload["image_base64"] = "data:image/jpeg;base64," + payload["image_base64"]
        response = self.client.post("/predict-obj-refine", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("mask", response.json()["results"][0])

    def test_invalid_image_and_runtime_error(self):
        response = self.client.post("/predict-obj-refine", json=self.payload(image_base64="%%%"))
        self.assertEqual(response.status_code, 400)
        for image, status in [(b"invalid", 400), (b"failure", 500)]:
            response = self.client.post("/predict-obj-refine", json=self.payload(image_base64=base64.b64encode(image).decode()))
            self.assertEqual(response.status_code, status)

    def test_file_protocol(self):
        response = self.client.post(
            "/predict-obj-refine/file", files={"image": ("test.jpg", b"image", "image/jpeg")},
            data={"class_names": "helmet, vest", "pre_detect_labels": "person, car",
                  "merge_results": "false", "return_mask": "false", "pre_detect_confidence": "0.2",
                  "crop_config_json": '{"max_crops":1,"max_size":512}'},
        )
        self.assertEqual(response.status_code, 200, response.text)
        call = FakeModel.calls[0]
        self.assertEqual(call[1:3], (["person", "car"], ["helmet", "vest"]))
        self.assertFalse(call[5])
        self.assertEqual(call[6].max_crops, 1)
        self.assertAlmostEqual(call[7], 0.2)

    def test_file_repeated_fields(self):
        response = self.client.post(
            "/predict-obj-refine/file", files={"image": ("test.jpg", b"image")},
            data={"class_names": ["helmet", "vest"], "pre_detect_labels": ["person", "car"]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(FakeModel.calls[0][1:3], (["person", "car"], ["helmet", "vest"]))

    def test_file_invalid_config(self):
        for cfg in ['{"max_crops":0}', "not-json", '{"max_crops":1000}']:
            response = self.client.post("/predict-obj-refine/file", files={"image": ("test.jpg", b"image")},
                                        data={"crop_config_json": cfg})
            self.assertEqual(response.status_code, 422)
        self.assertEqual(FakeModel.calls, [])

    def test_existing_routes_unchanged(self):
        response = self.client.post("/predict", json={"image": base64.b64encode(b"image").decode()})
        self.assertEqual(response.json(), {"results": []})
        response = self.client.post("/predict/file", files={"image": ("x.jpg", b"image")})
        self.assertEqual(response.json(), {"results": []})

    def test_frontend_static_assets_and_capabilities(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Ascend SAM3", response.text)
        self.assertIn("/static/style.css?v=", response.text)
        self.assertEqual(response.headers["cache-control"], "no-cache")
        for path in ("/static/style.css", "/static/script.js"):
            self.assertEqual(self.client.get(path).status_code, 200)
        data = self.client.get("/ui-config").json()
        self.assertEqual(data["supported_modes"], ["multi-class", "obj-refine"])
        self.assertGreater(data["refinement_limits"]["max_crops"], 0)


class BenchmarkTest(unittest.TestCase):
    def test_cli_and_concurrent_http_report(self):
        from scripts import benchmark_service as benchmark
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                seen.append((self.path, body))
                payload = json.dumps({"results": [], "refinement": {
                    "crops_processed": 2, "limited": True, "timings_ms": {"total": 10}
                }}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-SAM3-Worker-PID", "100")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                (directory / "a.jpg").write_bytes(b"image-a")
                (directory / "b.png").write_bytes(b"image-b")
                report = directory / "report.json"
                args = ["benchmark", "--url", f"http://127.0.0.1:{server.server_port}/predict/file",
                        "--images", tmp, "--mode", "obj-refine", "--rounds", "2", "--warmup", "1",
                        "--concurrency", "2", "--max-crops", "2", "--json-output", str(report)]
                with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(benchmark.main(), 0)
                data = json.loads(report.read_text(encoding="utf-8"))
                self.assertEqual(data["summary"]["successful_requests"], 4)
                self.assertEqual(data["summary"]["refinement"]["limited_requests"], 4)
                self.assertEqual(data["summary"]["refinement"]["mean_crops_processed"], 2)
                self.assertEqual(len(seen), 5) # excludes warmup from report
                self.assertTrue(all(path == "/predict-obj-refine/file" for path, _ in seen))
                self.assertTrue(all(b'name="crop_config_json"' in body for _, body in seen))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_refine_multipart_and_stats(self):
        from scripts.benchmark_service import build_multipart_request, execute_request
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "图像,空 格.jpg"
            image.write_bytes(b"image")
            prepared = build_multipart_request(image, ["helmet"], 0.3, False,
                                               [("pre_detect_labels", "person"), ("merge_results", "false")])
        self.assertIn(b'name="pre_detect_labels"', prepared.body)
        self.assertIn(b'name="image"', prepared.body)
        class Response(io.BytesIO):
            status = 200
            headers = {"X-SAM3-Worker-PID": "123"}
        with patch("urllib.request.urlopen", return_value=Response(json.dumps({
            "results": [], "refinement": {"crops_processed": 2, "limited": False}
        }).encode())):
            result = execute_request("http://test/predict-obj-refine/file", prepared, 5, True)
        self.assertTrue(result.success)
        self.assertEqual(result.refinement["crops_processed"], 2)
        self.assertEqual(result.detections, [])


if __name__ == "__main__":
    unittest.main()
