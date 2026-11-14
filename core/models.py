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
FACE_MODEL_URLS = {
    "GFPGANv1.3": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    "detection_Resnet50_Final": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing_parsenet": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
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
        urllib.request.urlretrieve((MODEL_URLS | FACE_MODEL_URLS)[name], temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def create_upsampler(name, device):
    prepare_basicsr()
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


def prepare_basicsr():
    import torchvision.transforms.functional as functional

    # BasicSR imports a torchvision module removed in newer releases.
    sys.modules.setdefault("torchvision.transforms.functional_tensor", functional)


def create_face_restorer(device):
    prepare_basicsr()
    import torch
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
    from gfpgan.utils import GFPGANer

    for name in FACE_MODEL_URLS:
        download_weights(name)
    restorer = GFPGANer.__new__(GFPGANer)
    restorer.device = device
    restorer.upscale = 1
    restorer.bg_upsampler = None
    restorer.gfpgan = GFPGANv1Clean(
        out_size=512, num_style_feat=512, channel_multiplier=2,
        decoder_load_path=None, fix_decoder=False, num_mlp=8,
        input_is_latent=True, different_w=True, narrow=1, sft_half=True,
    )
    weights = torch.load(str(MODEL_DIR / "GFPGANv1.3.pth"), map_location="cpu", weights_only=True)
    restorer.gfpgan.load_state_dict(weights.get("params_ema", weights.get("params")), strict=True)
    restorer.gfpgan.eval().to(device)
    restorer.face_helper = FaceRestoreHelper(
        1, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50",
        save_ext="png", use_parse=True, device=device, model_rootpath=str(MODEL_DIR),
    )
    return restorer
