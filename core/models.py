import os
import sys
import tempfile
import urllib.request
from pathlib import Path

MODEL_URLS = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
}
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def download_weights(name):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / f"{name}.pth"
    if target.is_file():
        return target
    print(f"Изтегляне на модела {name}...")
    fd, temporary = tempfile.mkstemp(dir=MODEL_DIR, suffix=".download")
    os.close(fd)
    try:
        urllib.request.urlretrieve(MODEL_URLS[name], temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def create_upsampler(name, device):
    import torchvision.transforms.functional as functional

    # BasicSR imports a torchvision module removed in newer releases.
    sys.modules.setdefault("torchvision.transforms.functional_tensor", functional)
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    if name == "RealESRGAN_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    else:
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=16 if name == "realesr-animevideov3" else 32,
            upscale=4, act_type="prelu",
        )
    return RealESRGANer(
        scale=4, model_path=str(download_weights(name)), model=model,
        tile=0, pre_pad=0, half=device.type == "cuda", device=device,
    )
