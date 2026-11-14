import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import gradio as gr
from PIL import Image

import app
from test_faces import FakeRestorer


class FakeUpsampler:
    def enhance(self, image, alpha_upsampler):
        return cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), None


class Progress:
    def __init__(self):
        self.updates = []

    def __call__(self, value, desc):
        self.updates.append((value, desc))


class AppTests(unittest.TestCase):
    def test_batch_and_first_photo_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "photo.png"
            Image.new("RGB", (4, 3), "red").save(source)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            device = type("Device", (), {"type": "cpu", "__str__": lambda self: "cpu"})()
            progress = Progress()

            def fake_video(path, output_dir, upsampler, scale, tile, face_restorer, progress_callback):
                progress_callback(1, 2)
                progress_callback(2, 2)
                return output_dir / "clip_enhanced.mp4"

            with patch.object(app, "ROOT", root), patch.object(app, "select_device", return_value=device), \
                 patch.object(app, "create_upsampler", return_value=FakeUpsampler()), \
                 patch.object(app, "create_face_restorer", return_value=FakeRestorer()) as face_factory, \
                 patch.object(app, "enhance_video", side_effect=fake_video):
                status, results, before, after = app.process_files(
                    [str(source)], str(root), 2, "realesr-general-x4v3", True, 128, progress)
            self.assertEqual(len(results), 2)
            self.assertEqual(before.size, (4, 3))
            self.assertEqual(after.size, (8, 6))
            self.assertTrue((root / "output" / "photo_enhanced.png").exists())
            self.assertIn("2 от 2", status)
            self.assertTrue(any("кадър 1/2" in desc for _, desc in progress.updates))
            face_factory.assert_called_once_with(device)

    def test_missing_and_unsupported_files_show_error(self):
        with self.assertRaises(gr.Error):
            app.process_files([], "", 2, "realesr-general-x4v3", False, 256, Progress())
        with self.assertRaises(gr.Error):
            app.process_files(["missing.txt"], "", 2, "realesr-general-x4v3", False, 256, Progress())

    def test_build_app(self):
        self.assertIsInstance(app.build_app(), gr.Blocks)


if __name__ == "__main__":
    unittest.main()
