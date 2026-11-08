"""Behaviour tests for the offline ONNX worker's two explicit modes.

The worker needs numpy/PIL, so run it with the worker interpreter:

    cd prototype/workers && /root/workspace/countthis/pilot/.venv/bin/python -m unittest test_onnx_worker -v
"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import onnx_worker

ONNX_JSON_PREFIX = "ANYCOUNT_JSON:"


class FakeSession:
    """Stands in for onnxruntime.InferenceSession and counts real inferences."""

    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [types.SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

    def run(self, _outputs, _feeds):
        self.calls += 1
        rows = []
        for i in range(10):
            # 2 rows x 5 columns of well-separated 8px boxes near canvas centre.
            rows.append([280.0 + (i % 5) * 20.0, 300.0 + (i // 5) * 20.0, 8.0, 8.0, 0.9])
        return [np.asarray([rows], dtype=np.float32)]


class FakeOrt:
    def __init__(self):
        self.session = FakeSession()

    def SessionOptions(self):
        return types.SimpleNamespace(intra_op_num_threads=0)

    def InferenceSession(self, *_args, **_kwargs):
        return self.session


class InferModes(unittest.TestCase):
    def setUp(self):
        self._original_ort = sys.modules.get("onnxruntime")
        self.fake_ort = FakeOrt()
        sys.modules["onnxruntime"] = self.fake_ort
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.image = root / "work.jpg"
        Image.new("RGB", (400, 300), (90, 120, 150)).save(self.image)
        self.onnx = root / "pack_int8.onnx"
        self.onnx.write_bytes(b"not-a-real-model")
        self.work = root / "work"

    def tearDown(self):
        if self._original_ort is None:
            sys.modules.pop("onnxruntime", None)
        else:
            sys.modules["onnxruntime"] = self._original_ort
        self.tmp.cleanup()

    def _infer(self, mode):
        args = argparse.Namespace(
            onnx=str(self.onnx),
            image=str(self.image),
            orig_width=400,
            orig_height=300,
            mode=mode,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            onnx_worker.cmd_infer(args)
        lines = [l for l in buf.getvalue().splitlines() if l.startswith(ONNX_JSON_PREFIX)]
        self.assertEqual(len(lines), 1, buf.getvalue())
        return json.loads(lines[0][len(ONNX_JSON_PREFIX):])

    def test_single_mode_is_one_pass_and_never_probes_the_tiles(self):
        payload = self._infer("single")
        self.assertEqual(self.fake_ort.session.calls, 1, "single must run exactly one inference")
        self.assertEqual(payload["mode"], "single_640")
        self.assertIsNone(payload["grid_boxes"])
        self.assertEqual(payload["count"], 10)

    def test_tiled_mode_always_runs_four_crops_and_reports_point_nms(self):
        payload = self._infer("tiled")
        self.assertEqual(self.fake_ort.session.calls, 4, "tiled must not add a single-pass probe")
        self.assertEqual(payload["mode"], "tiled_2x2_640+point_nms")
        self.assertEqual(len(payload["grid_boxes"]), 4)
        self.assertEqual(payload["count"], 40)

    def test_mode_flag_defaults_to_single_and_rejects_unknown_values(self):
        parser = onnx_worker.build_parser()
        base = ["infer", "--onnx", "a.onnx", "--image", "a.jpg", "--orig-width", "1", "--orig-height", "1"]
        self.assertEqual(parser.parse_args(base).mode, "single")
        self.assertEqual(parser.parse_args(base + ["--mode", "tiled"]).mode, "tiled")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(base + ["--mode", "auto"])


if __name__ == "__main__":
    unittest.main()
