import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import enhance
from core.video import check_ffmpeg, enhance_video


class FakeUpsampler:
    def enhance(self, tile, alpha_upsampler):
        return cv2.resize(tile, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), None


def ffprobe(path, arguments):
    result = subprocess.run(
        ["ffprobe", "-v", "error", *arguments, "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class VideoTests(unittest.TestCase):
    def make_video(self, path, with_audio=True, audio_offset="0.3"):
        command = [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=32x24:rate=10:duration=1",
        ]
        if with_audio:
            command += ["-itsoffset", audio_offset, "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5"]
        command += [
            "-vf", "setpts=if(lt(N\\,5)\\,N\\,5+(N-5)*2)/(10*TB)",
            "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        ]
        if with_audio:
            command += ["-c:a", "aac"]
        command += ["-y", str(path)]
        subprocess.run(command, check=True, capture_output=True)

    def frame_times(self, path):
        data = ffprobe(path, ["-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-show_frames"])
        return [float(frame["best_effort_timestamp_time"]) for frame in data["frames"]]

    def audio_start(self, path):
        data = ffprobe(path, ["-select_streams", "a:0", "-show_entries", "stream=start_time"])
        return float(data["streams"][0]["start_time"])

    def last_frame_duration(self, path):
        data = ffprobe(path, ["-select_streams", "v:0", "-show_entries", "frame=duration_time", "-show_frames"])
        return float(data["frames"][-1]["duration_time"])

    def test_vfr_audio_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mp4"
            self.make_video(source)
            original = source.read_bytes()
            output = root / "output"
            output.mkdir()
            (output / "input_enhanced.mp4").write_bytes(b"existing")
            with contextlib.redirect_stdout(io.StringIO()) as progress:
                result = enhance_video(source, output, FakeUpsampler(), 2, 16)
            self.assertEqual(result.name, "input_1_enhanced.mp4")
            self.assertEqual((output / "input_enhanced.mp4").read_bytes(), b"existing")
            self.assertEqual(source.read_bytes(), original)
            times_in, times_out = self.frame_times(source), self.frame_times(result)
            self.assertEqual(len(times_in), len(times_out))
            self.assertGreater(times_in[-1] - times_in[-2], times_in[1] - times_in[0])
            for before, after in zip(times_in, times_out):
                self.assertAlmostEqual(before - times_in[0], after - times_out[0], delta=0.001)
            self.assertAlmostEqual(self.last_frame_duration(source), self.last_frame_duration(result), delta=0.001)
            self.assertAlmostEqual(self.audio_start(source) - times_in[0],
                                   self.audio_start(result) - times_out[0], delta=0.025)
            info = ffprobe(result, ["-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height"])
            self.assertEqual(info["streams"][0], {"codec_name": "h264", "width": 64, "height": 48})
            self.assertIn("Кадър 10/10", progress.getvalue())

    def test_no_audio_and_cleanup_on_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mp4"
            self.make_video(source, with_audio=False)
            output = root / "output"
            with patch("core.video.enhance_pixels", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    enhance_video(source, output, FakeUpsampler(), 2, 16)
            self.assertEqual(list(output.iterdir()), [])
            with contextlib.redirect_stdout(io.StringIO()):
                result = enhance_video(source, output, FakeUpsampler(), 2, 16)
            self.assertEqual(ffprobe(result, ["-select_streams", "a", "-show_entries", "stream=index"])["streams"], [])

    def test_audio_starts_before_video_and_is_transcoded_when_needed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "delayed.mkv"
            subprocess.run([
                "ffmpeg", "-v", "error", "-itsoffset", "0.5",
                "-f", "lavfi", "-i", "testsrc=size=32x24:rate=5:duration=1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-y", str(source),
            ], check=True, capture_output=True)
            with contextlib.redirect_stdout(io.StringIO()):
                result = enhance_video(source, root / "output", FakeUpsampler(), 2, 16)
            source_offset = self.frame_times(source)[0] - self.audio_start(source)
            result_offset = self.frame_times(result)[0] - self.audio_start(result)
            self.assertAlmostEqual(source_offset, result_offset, delta=0.025)
            audio = ffprobe(result, ["-select_streams", "a:0", "-show_entries", "stream=codec_name"])
            self.assertEqual(audio["streams"][0]["codec_name"], "aac")

    def test_batch_continues_when_ffmpeg_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 4), "red").save(root / "photo.png")
            (root / "video.mp4").write_bytes(b"video")
            device = type("Device", (), {"type": "cpu", "__str__": lambda self: "cpu"})()
            with patch("core.device.select_device", return_value=device), \
                 patch("enhance.create_upsampler", return_value=FakeUpsampler()), \
                 patch("enhance.Path", side_effect=lambda p: root / p), \
                 patch("core.video.check_ffmpeg", side_effect=RuntimeError("FFmpeg missing")), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(enhance.main([str(root), "--scale", "2"]), 1)
            self.assertTrue((root / "output" / "photo_enhanced.png").is_file())

    def test_phone_rotation_matches_ffmpeg(self):
        help_text = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"], capture_output=True, text=True).stdout
        if "-display_rotation" not in help_text:
            self.skipTest("FFmpeg cannot set display rotation for a test input")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            rotated = root / "rotated.mp4"
            self.make_video(source, with_audio=False)
            subprocess.run([
                "ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(source),
                "-c", "copy", "-y", str(rotated),
            ], check=True, capture_output=True)
            with contextlib.redirect_stdout(io.StringIO()):
                result = enhance_video(rotated, root / "output", FakeUpsampler(), 2, 16)

            def pixels(path, resize=False):
                command = ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1"]
                if resize:
                    command += ["-vf", "scale=48:64"]
                command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
                raw = subprocess.run(command, check=True, capture_output=True).stdout
                return np.frombuffer(raw, dtype=np.uint8).astype(np.int16)

            expected, actual = pixels(rotated, True), pixels(result)
            self.assertEqual(len(expected), len(actual))
            self.assertLess(np.abs(expected - actual).mean(), 25)


class MissingFFmpegTests(unittest.TestCase):
    def test_missing_ffmpeg_has_install_guidance(self):
        with patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "инсталирайте FFmpeg"):
                check_ffmpeg()


if __name__ == "__main__":
    unittest.main()
