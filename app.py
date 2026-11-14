import os
import secrets
import tempfile
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
if __name__ == "__main__":
    cache = tempfile.TemporaryDirectory(prefix="repixel-gradio-")
    os.environ["GRADIO_TEMP_DIR"] = cache.name

import gradio as gr

from core.device import select_device
from core.image import SUPPORTED_EXTENSIONS, enhance_image
from core.models import MODEL_URLS, create_face_restorer, create_upsampler
from core.video import SUPPORTED_VIDEO_EXTENSIONS, enhance_video

ROOT = Path(__file__).resolve().parent
SUPPORTED = SUPPORTED_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS


def process_files(files, folder, scale, model, faces, tile, progress=gr.Progress()):
    paths = [Path(path) for path in (files or [])]
    if folder and folder.strip():
        directory = Path(folder.strip()).expanduser()
        if not directory.is_dir():
            raise gr.Error("Папката не съществува. Проверете пътя.")
        paths.extend(sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise gr.Error("Изберете файлове или въведете папка с поддържани файлове.")
    if any(not p.is_file() or p.suffix.lower() not in SUPPORTED for p in paths):
        raise gr.Error("Използвайте само съществуващи JPG, PNG, WEBP, MP4, MOV, MKV или AVI файлове.")
    if model not in MODEL_URLS or int(scale) not in (2, 4) or not tile or int(tile) < 1:
        raise gr.Error("Проверете модела, увеличението и размера на блоковете.")

    progress(0, desc="Зареждане на моделите...")
    try:
        device = select_device()
        upsampler = create_upsampler(model, device)
        restorer = create_face_restorer(device) if faces else None
    except Exception as error:
        raise gr.Error(f"Моделът не може да се зареди: {error}. Проверете инсталацията и интернет връзката.") from error

    results = []
    errors = []
    before = after = None
    output_dir = ROOT / "output"
    for index, path in enumerate(paths):
        progress(index / len(paths), desc=f"Обработка на файл {index + 1} от {len(paths)}: {path.name}")
        try:
            if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                def frame_progress(frame, total):
                    progress((index + frame / total) / len(paths),
                             desc=f"Файл {index + 1}/{len(paths)} — кадър {frame}/{total}")

                result = enhance_video(path, output_dir, upsampler, int(scale), int(tile),
                                       face_restorer=restorer, progress_callback=frame_progress)
            else:
                result = enhance_image(path, output_dir, upsampler, int(scale), int(tile),
                                       face_restorer=restorer)
                if before is None:
                    from PIL import Image, ImageOps
                    with Image.open(path) as original:
                        before = ImageOps.exif_transpose(original).copy()
                    with Image.open(result) as enhanced:
                        after = enhanced.copy()
            results.append(str(result))
        except Exception as error:
            errors.append(f"{path.name}: {error}")
    progress(1, desc="Готово")
    device_message = f"Устройство: {device}."
    if device.type == "cpu":
        device_message += " За AMD/Intel GPU разгледайте отделния realesrgan-ncnn-vulkan."
    message = f"{device_message} Готови файлове: {len(results)} от {len(paths)}."
    if errors:
        message += "\nГрешки:\n" + "\n".join(errors)
    if not results:
        raise gr.Error(message)
    return message, results, before, after


def build_app():
    with gr.Blocks(title="RePixel-AI") as demo:
        gr.Markdown("# RePixel-AI\nПодобряване на снимки и видео с готови AI модели.")
        files = gr.UploadButton("Изберете снимки или видео", file_count="multiple", type="filepath",
                                file_types=[".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".mkv", ".avi"])
        selected = gr.Textbox(label="Избрани файлове", interactive=False)
        files.upload(lambda paths: ", ".join(Path(path).name for path in (paths or [])),
                     inputs=files, outputs=selected)
        folder = gr.Textbox(label="Или въведете път към папка", placeholder="Път към папка на този компютър")
        with gr.Row():
            scale = gr.Radio([2, 4], value=2, label="Увеличение")
            model = gr.Dropdown(list(MODEL_URLS), value="realesr-general-x4v3", label="Модел")
        with gr.Row():
            faces = gr.Checkbox(label="Възстановяване на лица с GFPGAN", value=False)
            tile = gr.Number(label="Размер на блоковете", value=256, minimum=1, precision=0)
        start = gr.Button("Старт", variant="primary")
        status = gr.Textbox(label="Състояние", interactive=False, lines=3)
        output = gr.File(label="Готови файлове", file_count="multiple")
        with gr.Row():
            before = gr.Image(label="Преди", type="filepath", interactive=False, buttons=[])
            after = gr.Image(label="След", type="filepath", interactive=False, buttons=[])
        gr.Markdown("Сравнението показва първата успешно обработена снимка. "
                    "Резултатите се запазват в папката `output`; оригиналите не се променят.")
        start.click(process_files, inputs=[files, folder, scale, model, faces, tile],
                    outputs=[status, output, before, after], concurrency_limit=1)
    return demo


if __name__ == "__main__":
    try:
        (ROOT / "output").mkdir(exist_ok=True)
        password = secrets.token_urlsafe(12)
        print(f"Потребител: repixel\nКод за достъп: {password}", flush=True)
        build_app().queue(default_concurrency_limit=1, api_open=False).launch(
            server_name="127.0.0.1", inbrowser=True, allowed_paths=[str(ROOT / "output")], show_error=True,
            footer_links=[], auth=("repixel", password),
            auth_message="Username: потребител; Password: код за достъп от терминала; Login: вход.",
        )
    finally:
        cache.cleanup()
