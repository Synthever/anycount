"""Regression tests for the M2 tiled-rescue accept band.

Run with the ONNX worker interpreter (it needs numpy/PIL to import the module):

    cd prototype/workers && /root/workspace/countthis/pilot/.venv/bin/python -m unittest test_onnx_worker -v
"""
import unittest

import onnx_worker


class TiledAcceptBand(unittest.TestCase):
    def test_single_pass_miss_accepts_a_non_empty_tiled_rescue(self):
        # Large sparse image: single pass returns 0, tiled crops still find it.
        self.assertTrue(onnx_worker.accept_tiled_result(0, 12))

    def test_single_pass_miss_with_empty_tiled_stays_single(self):
        self.assertFalse(onnx_worker.accept_tiled_result(0, 0))

    def test_band_edges_and_collapse(self):
        self.assertTrue(onnx_worker.accept_tiled_result(100, 85))
        self.assertTrue(onnx_worker.accept_tiled_result(100, 135))
        self.assertFalse(onnx_worker.accept_tiled_result(100, 84))
        self.assertFalse(onnx_worker.accept_tiled_result(100, 136))
        self.assertFalse(onnx_worker.accept_tiled_result(100, 0))


if __name__ == "__main__":
    unittest.main()
