import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from .emotions_detection import EmotionsRecognition

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CMD_OMZ = "uv run omz_downloader --name emotions-recognition-retail-0003"

MODEL_DIR = PROJECT_ROOT / "emotions_detection" / "intel"
if not MODEL_DIR.exists():
    subprocess.call(CMD_OMZ.split(" "), cwd=str(PROJECT_ROOT / "emotions_detection"))
MODEL_PATH = MODEL_DIR / "emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003"
CAMERA_PORT = int(os.getenv("CAMERA_PORT", "0"))
logger.debug(
    {"PROJECT_ROOT": PROJECT_ROOT, "MODEL_PATH": MODEL_PATH, "CAMERA_PORT": CAMERA_PORT}
)

emotions_recognition = EmotionsRecognition(model_path=str(MODEL_PATH))
