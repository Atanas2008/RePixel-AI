import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import enhance
from core.image import enhance_pixels


class FakeUpsampler:
    def enhance(self, image, alpha_upsampler):
        return cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), None


class FakeFaceHelper:
    def set_upscale_factor(self, factor):
        self.factor = factor


class FakeRestorer:
    def __init__(self):
        self.face_helper = FakeFaceHelper()
        self.bg_upsampler = None
        self.images = []

    def enhance(self, image, **kwargs):
        self.images.append(image.shape)
        background, _ = self.bg_upsampler.enhance(image, self.upscale)
        result = background.copy()
        result[:, :, 0] = 100
        return [], [], result


class FaceTests(unittest.TestCase):
    def test_faces_use_bounded_proxy_and_keep_alpha(self):
        image = Image.new("RGBA", (1400, 20), (10, 20, 30, 80))
        restorer = FakeRestorer()
        result = enhance_pixels(image, FakeUpsampler(), 2, 256, face_restorer=restorer)
        self.assertEqual(restorer.images[0][1], 1280)
        self.assertEqual(result.size, (2800, 40))
        self.assertEqual(result.mode, "RGBA")
        np.testing.assert_allclose(result.getpixel((10, 10)), (10, 20, 100, 80), atol=1)
        self.assertIsNone(restorer.bg_upsampler)

    def test_cli_enables_faces_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.png"
            Image.new("RGB", (4, 4), "red").save(path)
            device = type("Device", (), {"type": "cpu", "__str__": lambda self: "cpu"})()
            restorer = FakeRestorer()
            with patch("core.device.select_device", return_value=device), \
                 patch("enhance.create_upsampler", return_value=FakeUpsampler()), \
                 patch("enhance.create_face_restorer", return_value=restorer) as create, \
                 patch("enhance.Path", side_effect=lambda p: Path(directory) / p), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(enhance.main([str(path), "--scale", "2"]), 0)
                create.assert_not_called()
                self.assertEqual(enhance.main([str(path), "--scale", "2", "--faces"]), 0)
                create.assert_called_once_with(device)
            self.assertEqual(len(restorer.images), 1)


if __name__ == "__main__":
    unittest.main()
