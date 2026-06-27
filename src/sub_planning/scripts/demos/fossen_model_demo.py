#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：作为脚本入口转发到模型演示函数，便于直接启动 Fossen 三自由度模型示例。
输入：通过脚本入口参数或内部示例参数触发模型演示流程。
输出：输出示例日志或调用下层模型函数完成一次演示。
"""
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from models.fossen_model_demo import main


if __name__ == '__main__':
    main()
