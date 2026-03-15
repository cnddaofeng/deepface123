"""
main.py — 入口脚本

用法：
  python main.py --target src   # 处理 workspace/data_src → aligned
  python main.py --target dst   # 处理 workspace/data_dst → aligned
"""

import argparse
import sys
from pathlib import Path

from config import ALIGN_SIZE   # 统一从 config 读取，不再硬编码
from pipeline import FaceExtractionPipeline


def find_workspace() -> Path:
    current = Path(__file__).resolve().parent
    for p in [current] + list(current.parents):
        ws = p / "workspace"
        if ws.exists() and ws.is_dir():
            return ws
    raise FileNotFoundError(
        "找不到 workspace 目录，请确认脚本位于 DeepFaceLab-Torch/plugins/ 下"
    )


def main():
    parser = argparse.ArgumentParser(description="DFL 标准脸提取工具")
    parser.add_argument(
        "--target", choices=["src", "dst"], required=True,
        help="处理目标：src = data_src，dst = data_dst"
    )
    args = parser.parse_args()

    workspace = find_workspace()
    key       = "data_src" if args.target == "src" else "data_dst"
    input_dir  = workspace / key
    output_dir = workspace / key / "aligned"

    print(f"[INFO] target    : {args.target}")
    print(f"[INFO] input_dir : {input_dir}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] align_size: {ALIGN_SIZE}")

    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        sys.exit(1)

    pipeline = FaceExtractionPipeline(size=ALIGN_SIZE)
    pipeline.process_folder(str(input_dir), str(output_dir))


if __name__ == "__main__":
    main()
