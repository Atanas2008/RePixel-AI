from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_WIDTH, MAX_HEIGHT = 3840, 2160
TILE_PADDING = 16


def output_dimensions(width, height, scale):
    factor = min(scale, MAX_WIDTH / width, MAX_HEIGHT / height)
    return max(1, int(width * factor)), max(1, int(height * factor))


def enhance_pixels(image, upsampler, scale, tile_size, dimensions=None, face_restorer=None):
    import cv2
    import numpy as np
    from PIL import Image

    width, height = image.size
    out_width, out_height = dimensions or output_dimensions(width, height, scale)
    result = Image.new(image.mode, (out_width, out_height))
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            right, bottom = min(left + tile_size, width), min(top + tile_size, height)
            x0, y0 = max(0, left - TILE_PADDING), max(0, top - TILE_PADDING)
            x1, y1 = min(width, right + TILE_PADDING), min(height, bottom + TILE_PADDING)
            tile = np.asarray(image.crop((x0, y0, x1, y1)))
            color = cv2.COLOR_RGBA2BGRA if image.mode == "RGBA" else cv2.COLOR_RGB2BGR
            enhanced, _ = upsampler.enhance(cv2.cvtColor(tile, color), alpha_upsampler="bicubic")
            enhanced = Image.fromarray(cv2.cvtColor(enhanced, color))
            enhanced = enhanced.crop(((left - x0) * 4, (top - y0) * 4, (right - x0) * 4, (bottom - y0) * 4))
            box = (left * out_width // width, top * out_height // height,
                   right * out_width // width, bottom * out_height // height)
            if box[2] > box[0] and box[3] > box[1]:
                result.paste(enhanced.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS), box)
    if face_restorer is not None:
        result = restore_faces(image, result, face_restorer)
    return result


def restore_faces(image, background, restorer):
    import cv2
    import numpy as np
    from PIL import Image

    proxy = image.convert("RGB")
    proxy.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    original = cv2.cvtColor(np.asarray(proxy), cv2.COLOR_RGB2BGR)
    background_bgr = cv2.cvtColor(np.asarray(background.convert("RGB")), cv2.COLOR_RGB2BGR)

    class Background:
        def enhance(self, _image, outscale):
            return background_bgr, None

    restorer.upscale = background.width / proxy.width
    restorer.face_helper.set_upscale_factor(restorer.upscale)
    restorer.bg_upsampler = Background()
    try:
        _, _, restored = restorer.enhance(original, has_aligned=False, only_center_face=False, paste_back=True)
    finally:
        restorer.bg_upsampler = None
    restored = cv2.resize(restored, background.size, interpolation=cv2.INTER_LANCZOS4)
    result = Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
    if background.mode == "RGBA":
        result.putalpha(background.getchannel("A"))
    return result


def enhance_image(source, output_dir, upsampler, scale, tile_size, face_restorer=None):
    from PIL import Image, ImageOps

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        image = image.convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB")
    result = enhance_pixels(image, upsampler, scale, tile_size, face_restorer=face_restorer)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    suffix = source.suffix.lower()
    format_name = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}[suffix]
    if format_name == "JPEG" and result.mode == "RGBA":
        result = result.convert("RGB")
    index = 0
    while True:
        extra = f"_{index}" if index else ""
        destination = output_dir / f"{source.stem}{extra}_enhanced{suffix}"
        try:
            with destination.open("xb") as file:
                try:
                    options = {"quality": 95} if format_name in {"JPEG", "WEBP"} else {}
                    result.save(file, format=format_name, **options)
                except BaseException:
                    file.close()
                    destination.unlink(missing_ok=True)
                    raise
            return destination
        except FileExistsError:
            index += 1
