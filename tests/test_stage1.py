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
from core.image import enhance_image, output_dimensions
from core import models


class FakeUpsampler:
    def __init__(self):
        self.shapes = []

    def enhance(self, tile, alpha_upsampler):
        self.shapes.append(tile.shape)
        return cv2.resize(tile, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), None


class ImageTests(unittest.TestCase):
    def test_dimensions_respect_cap_and_aspect_ratio(self):
        self.assertEqual(output_dimensions(200, 100, 2), (400, 200))
        self.assertEqual(output_dimensions(1920, 1080, 4), (3840, 2160))
        self.assertEqual(output_dimensions(6000, 4000, 4), (3240, 2160))

    def test_tiled_alpha_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "photo.png"
            original = np.full((13, 19, 4), (30, 60, 90, 120), dtype=np.uint8)
            Image.fromarray(original).save(source)
            before = source.read_bytes()
            upsampler = FakeUpsampler()
            first = enhance_image(source, root, upsampler, 2, 5)
            second = enhance_image(source, root, upsampler, 2, 5)
            self.assertEqual(first.name, "photo_enhanced.png")
            self.assertEqual(second.name, "photo_enhanced_1.png")
            self.assertEqual(source.read_bytes(), before)
            with Image.open(first) as result:
                self.assertEqual(result.size, (38, 26))
                self.assertEqual(result.mode, "RGBA")
                np.testing.assert_allclose(np.asarray(result)[10, 10], original[0, 0], atol=1)
            self.assertGreater(len(upsampler.shapes), 1)

    def test_batch_continues_after_bad_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.jpg").write_bytes(b"not an image")
            Image.new("RGB", (4, 3), "red").save(root / "good.webp")
            device = type("Device", (), {"type": "cpu", "__str__": lambda self: "cpu"})()
            with patch("core.device.select_device", return_value=device), \
                 patch("enhance.create_upsampler", return_value=FakeUpsampler()), \
                 patch("enhance.Path", side_effect=lambda p: root / p), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(enhance.main([str(root), "--scale", "2"]), 1)
            self.assertTrue((root / "output" / "good_enhanced.webp").is_file())

    def test_invalid_tile_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            enhance.main(["missing.png", "--tile", "0"])


class DownloadTests(unittest.TestCase):
    def test_interrupted_download_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(models, "MODEL_DIR", Path(directory)), \
                 patch("urllib.request.urlretrieve", side_effect=KeyboardInterrupt), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    models.download_weights("RealESRGAN_x4plus")
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
