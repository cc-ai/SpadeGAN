import os
from pathlib import Path
import argparse
import torch
import sys

sys.path.append(str((Path(__file__).parent.parent / 'scripts').resolve()))
root = Path(__file__).parent.parent.resolve()

from scripts.utils import load_opts




opts = load_opts(path=root / "configs/patchgan.yaml", default=root / "configs/patchgan.yaml")