from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
THIRD_PARTY_RVM = ROOT / "third_party" / "rvm"
THIRD_PARTY_VDA = ROOT / "third_party" / "video_depth_anything"

DEFAULT_RVM_WEIGHTS = MODELS_DIR / "rvm_mobilenetv3.pth"
DEFAULT_DEPTH_WEIGHTS = MODELS_DIR / "video_depth_anything_vits.pth"


def add_vendor_to_syspath() -> None:
    for path in (THIRD_PARTY_RVM, THIRD_PARTY_VDA):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
