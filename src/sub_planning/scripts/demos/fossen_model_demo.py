#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from models.fossen_model_demo import main


if __name__ == '__main__':
    main()
