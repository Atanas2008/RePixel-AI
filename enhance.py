import argparse
import sys
from pathlib import Path

from core.image import SUPPORTED_EXTENSIONS, enhance_image
from core.models import MODEL_URLS, create_upsampler


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Размерът трябва да е положително цяло число.")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Подобряване на снимки с Real-ESRGAN.")
    parser.add_argument("input", type=Path, help="Снимка или папка със снимки")
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4, help="Увеличение: 2 или 4")
    parser.add_argument("--model", choices=MODEL_URLS, default="RealESRGAN_x4plus", help="AI модел")
    parser.add_argument("--tile", type=positive_int, default=256, help="Размер на блоковете (по подразбиране: 256)")
    args = parser.parse_args(argv)

    if args.input.is_dir():
        files = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    elif args.input.is_file() and args.input.suffix.lower() in SUPPORTED_EXTENSIONS:
        files = [args.input]
    else:
        parser.error("Посочете съществуваща снимка JPG, PNG или WEBP или папка със снимки.")
    if not files:
        parser.error("В папката няма снимки JPG, PNG или WEBP.")

    try:
        from core.device import select_device
        device = select_device()
        print(f"Устройство: {device}")
        if device.type == "cpu":
            print("За AMD/Intel GPU разгледайте realesrgan-ncnn-vulkan; този проект използва CPU.")
        upsampler = create_upsampler(args.model, device)
        failed = False
        for path in files:
            try:
                print(f"Обработка: {path}")
                destination = enhance_image(path, Path("output"), upsampler, args.scale, args.tile)
                print(f"Запазено: {destination}")
            except Exception as error:
                failed = True
                print(f"Грешка при {path}: {error}. Проверете файла или опитайте с по-малък --tile.", file=sys.stderr)
        return 1 if failed else 0
    except Exception as error:
        print(f"Грешка: {error}. Проверете инсталацията и интернет връзката; вижте README.md.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nПрекъснато.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
