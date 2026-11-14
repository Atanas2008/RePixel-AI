import json
import os
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path

from core.image import enhance_pixels, output_dimensions

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
VIDEO_TIME_BASE = Fraction(1, 90000)


def check_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "FFmpeg или ffprobe липсва — инсталирайте FFmpeg и го добавете в PATH. "
            "Вижте раздела за видео в README.md."
        )


def probe_video(source):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames:stream_tags=rotate:stream_side_data=rotation",
        "-of", "json", str(source),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"FFprobe не може да прочете видеото: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("Файлът няма видео поток.")
    stream = streams[0]
    try:
        count = int(stream["nb_read_frames"])
        if count <= 0:
            raise ValueError
    except (ValueError, KeyError, TypeError):
        raise ValueError("Броят на кадрите не може да се определи; проверете видео файла.") from None
    rotation = stream.get("tags", {}).get("rotate", 0)
    for data in stream.get("side_data_list", []):
        if "rotation" in data:
            rotation = data["rotation"]
            break
    rotation = int(rotation) % 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Неподдържано завъртане на видеото; първо го завъртете с FFmpeg.")
    return count, rotation


def even_dimensions(width, height, scale):
    width, height = output_dimensions(width, height, scale)
    return max(2, width - width % 2), max(2, height - height % 2)


def format_time(seconds):
    seconds = max(0, round(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def enhance_video(source, output_dir, upsampler, scale, tile_size):
    import av

    check_ffmpeg()
    source = Path(source)
    count, rotation = probe_video(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = None
    completed = False
    try:
        for index in range(1000000):
            extra = f"_{index}" if index else ""
            candidate = output_dir / f"{source.stem}{extra}_enhanced.mp4"
            try:
                with candidate.open("xb"):
                    pass
                destination = candidate
                break
            except FileExistsError:
                continue
        if destination is None:
            raise RuntimeError("Не може да се избере свободно име за резултата.")

        with tempfile.TemporaryDirectory(prefix=".repixel-", dir=output_dir) as temp_dir:
            video_only = Path(temp_dir) / "video.mp4"
            merged = Path(temp_dir) / "merged.mp4"
            with av.open(str(source)) as input_file:
                input_stream = input_file.streams.video[0]
                origin = Fraction(input_file.start_time or 0, av.time_base)
                frames = input_file.decode(input_stream)
                first = next(frames, None)
                if first is None:
                    raise ValueError("Видеото няма декодируеми кадри.")
                if first.pts is None or first.time_base is None:
                    raise ValueError("Липсват времеви отметки на кадрите; преобразувайте видеото с FFmpeg.")
                origin = min(origin, first.pts * first.time_base)
                image = first.to_image()
                if rotation:
                    image = image.rotate(rotation, expand=True)
                dimensions = even_dimensions(image.width, image.height, scale)
                aspect_ratio = input_stream.sample_aspect_ratio or Fraction(1, 1)
                if rotation in (90, 270):
                    aspect_ratio = 1 / aspect_ratio
                aspect_ratio *= Fraction(image.width * dimensions[1], image.height * dimensions[0])

                started = time.monotonic()
                enhanced = enhance_pixels(image, upsampler, scale, tile_size, dimensions)
                first_seconds = max(time.monotonic() - started, 0.001)
                print(f"Очаквано време за обработка: около {format_time(first_seconds * count)} "
                      "(оценката е по първия кадър).", flush=True)
                with av.open(str(video_only), mode="w") as output_file:
                    rate = input_stream.average_rate or Fraction(30, 1)
                    stream = output_file.add_stream("libx264", rate=rate)
                    stream.width, stream.height = dimensions
                    stream.pix_fmt = "yuv420p"
                    stream.codec_context.sample_aspect_ratio = aspect_ratio
                    stream.time_base = VIDEO_TIME_BASE
                    stream.codec_context.time_base = VIDEO_TIME_BASE
                    stream.options = {"crf": "18", "bf": "0"}
                    previous_pts = None
                    durations = {}

                    def mux_packets(packets):
                        for packet in packets:
                            duration = durations.pop(packet.pts * packet.time_base, None)
                            if duration:
                                packet.duration = max(1, round(duration / packet.time_base))
                            output_file.mux(packet)

                    def write_frame(frame, result):
                        nonlocal previous_pts
                        if frame.pts is None or frame.time_base is None:
                            raise ValueError("Липсват времеви отметки на кадрите.")
                        pts = round((frame.pts * frame.time_base - origin) / VIDEO_TIME_BASE)
                        if previous_pts is not None and pts <= previous_pts:
                            raise ValueError("Времевите отметки на кадрите не нарастват; поправете видео файла с FFmpeg.")
                        previous_pts = pts
                        durations[pts * VIDEO_TIME_BASE] = frame.duration * frame.time_base if frame.duration else None
                        new_frame = av.VideoFrame.from_image(result)
                        new_frame.pts = pts
                        new_frame.time_base = VIDEO_TIME_BASE
                        mux_packets(stream.encode(new_frame))

                    write_frame(first, enhanced)
                    processed = 1
                    print(f"Кадър {processed}/{count} ({processed / count * 100:.1f}%), "
                          f"остават около {format_time(first_seconds * (count - processed))}", flush=True)
                    for frame in frames:
                        if (frame.width, frame.height) != (first.width, first.height):
                            raise ValueError("Променливият размер на кадрите не се поддържа.")
                        image = frame.to_image()
                        if rotation:
                            image = image.rotate(rotation, expand=True)
                        result = enhance_pixels(image, upsampler, scale, tile_size, dimensions)
                        write_frame(frame, result)
                        processed += 1
                        elapsed = time.monotonic() - started
                        remaining = elapsed / processed * max(count - processed, 0)
                        print(f"Кадър {processed}/{count} ({processed / count * 100:.1f}%), "
                              f"остават около {format_time(remaining)}", flush=True)
                    if processed != count:
                        raise ValueError(f"Прочетени са {processed} от {count} кадъра; проверете видео файла.")
                    mux_packets(stream.encode())
                audio_codecs = [s.codec_context.name for s in input_file.streams.audio]

            audio_codec = "copy" if all(c in {"aac", "mp3", "alac", "ac3", "eac3"} for c in audio_codecs) else "aac"
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-copyts",
                "-i", str(video_only), "-itsoffset", str(float(-origin)), "-i", str(source),
                "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "-1",
                "-c:v", "copy", "-c:a", audio_codec, "-movflags", "+faststart", "-y", str(merged),
            ]
            mux = subprocess.run(command, capture_output=True, text=True, check=False)
            if mux.returncode:
                raise RuntimeError(f"FFmpeg не успя да добави звука: {mux.stderr.strip()}")
            os.replace(merged, destination)
            completed = True
            return destination
    finally:
        if destination is not None and not completed:
            destination.unlink(missing_ok=True)
