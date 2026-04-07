import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def install_handler_stubs():
    runpod_module = types.ModuleType("runpod")
    serverless_module = types.ModuleType("runpod.serverless")
    utils_module = types.ModuleType("runpod.serverless.utils")

    utils_module.rp_upload = types.SimpleNamespace(upload_file_to_bucket=None)
    serverless_module.start = lambda config: config
    serverless_module.utils = utils_module
    runpod_module.serverless = serverless_module

    websocket_module = types.ModuleType("websocket")

    class WebSocketTimeoutException(Exception):
        pass

    class WebSocketConnectionClosedException(Exception):
        pass

    class WebSocket:
        def __init__(self):
            self.timeout = None

        def settimeout(self, timeout):
            self.timeout = timeout

        def connect(self, _url):
            return None

        def recv(self):
            raise NotImplementedError

        def close(self):
            return None

    websocket_module.WebSocket = WebSocket
    websocket_module.WebSocketTimeoutException = WebSocketTimeoutException
    websocket_module.WebSocketConnectionClosedException = WebSocketConnectionClosedException

    sys.modules["runpod"] = runpod_module
    sys.modules["runpod.serverless"] = serverless_module
    sys.modules["runpod.serverless.utils"] = utils_module
    sys.modules["websocket"] = websocket_module


install_handler_stubs()
handler = importlib.import_module("handler")


class FakeWebSocket:
    def __init__(self, messages):
        self._messages = list(messages)

    def recv(self):
        if not self._messages:
            raise AssertionError("No more websocket messages available")
        return self._messages.pop(0)


class HandlerTests(unittest.TestCase):
    def test_normalize_base64_data_strips_data_uri_and_whitespace(self):
        encoded = "data:image/png;base64, aGVs bG8= \n"
        self.assertEqual(handler.normalize_base64_data(encoded), "aGVsbG8=")

    def test_prepare_job_config_uses_defaults(self):
        with tempfile.TemporaryDirectory() as temp_root:
            example_image = Path(temp_root) / "example.png"
            example_image.write_bytes(b"not-a-real-image")

            with mock.patch.object(handler, "TEMP_ROOT", temp_root), mock.patch.object(handler, "DEFAULT_IMAGE_PATH", str(example_image)):
                config = handler.prepare_job_config({"prompt": "test prompt"}, "job/123")

            self.assertEqual(config["seed"], handler.DEFAULT_SEED)
            self.assertEqual(config["width"], handler.DEFAULT_WIDTH)
            self.assertEqual(config["height"], handler.DEFAULT_HEIGHT)
            self.assertEqual(config["steps"], handler.DEFAULT_STEPS)
            self.assertEqual(config["negative_prompt"], handler.DEFAULT_NEGATIVE_PROMPT)
            self.assertTrue(config["temp_dir"].endswith("job_123"))

            handler.cleanup_temp_dir(config["temp_dir"])

    def test_prepare_job_config_rejects_multiple_image_sources(self):
        with tempfile.TemporaryDirectory() as temp_root:
            example_image = Path(temp_root) / "example.png"
            example_image.write_bytes(b"not-a-real-image")

            with mock.patch.object(handler, "TEMP_ROOT", temp_root), mock.patch.object(handler, "DEFAULT_IMAGE_PATH", str(example_image)):
                with self.assertRaises(Exception) as context:
                    handler.prepare_job_config(
                        {
                            "prompt": "test prompt",
                            "image_url": "https://example.com/image.png",
                            "image_base64": "aGVsbG8=",
                        },
                        "job-1",
                    )

            self.assertIn("image 입력은 하나만 지정할 수 있습니다", str(context.exception))

    def test_validate_prompt_node_types_bypasses_optional_blockswap(self):
        prompt = {
            "122": {"class_type": "WanVideoModelLoader", "inputs": {}},
            "279": {"class_type": "WanVideoLoraSelectMulti", "inputs": {}},
            "525": {"class_type": "WanVideoEnhancedBlockSwap", "inputs": {}},
            "555": {
                "class_type": "WanVideoSetBlockSwap",
                "inputs": {"model": ["122", 0], "block_swap_args": ["525", 0]},
            },
            "556": {
                "class_type": "WanVideoSetLoRAs",
                "inputs": {"model": ["555", 0], "lora": ["279", 0]},
            },
        }

        available_node_types = {
            "WanVideoModelLoader",
            "WanVideoLoraSelectMulti",
            "WanVideoSetLoRAs",
        }

        with mock.patch.object(handler, "get_available_node_types", return_value=available_node_types):
            handler.validate_prompt_node_types(prompt)

        self.assertNotIn("525", prompt)
        self.assertNotIn("555", prompt)
        self.assertEqual(prompt["556"]["inputs"]["model"], ["122", 0])

    def test_get_videos_raises_execution_error(self):
        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "execution_error",
                        "data": {
                            "node_id": "220",
                            "exception_type": "RuntimeError",
                            "exception_message": "sampler exploded",
                        },
                    }
                )
            ]
        )

        with mock.patch.object(handler, "queue_prompt", return_value={"prompt_id": "prompt-1"}):
            with self.assertRaises(Exception) as context:
                handler.get_videos(ws, {"220": {}}, "job-1", "client-1")

        self.assertIn("sampler exploded", str(context.exception))


if __name__ == "__main__":
    unittest.main()
