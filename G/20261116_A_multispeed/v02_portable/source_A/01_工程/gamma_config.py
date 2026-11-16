"""Explicit experiment configuration; other controller/scoring values inherited."""
from pathlib import Path
import json,math
CONFIG_PATH=Path(__file__).resolve().parent/'gamma_config.json'
SETTINGS=json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
DEFAULT_W_GAMMA=float(SETTINGS['default_w_gamma'])
if not math.isfinite(DEFAULT_W_GAMMA) or DEFAULT_W_GAMMA<=0:raise ValueError('Positive finite default w_gamma required')
