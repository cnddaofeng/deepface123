"""
config.py — 全局配置（修复版）

修复记录：
  BUG5 ALIGN_SIZE=256 与 pipeline.py 实际使用的 512 不一致。
       config.py 从未被任何模块 import，配置形同虚设。
  修复：统一为 512，并在 pipeline.py 中引用此常量。
"""

import os

# ── 路径 ──────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DFL_ROOT  = os.path.abspath(os.path.join(BASE, "../.."))          # D:/DeepFaceLab-Torch
WORKSPACE = os.path.join(DFL_ROOT, "workspace")

DATA_SRC  = os.path.join(WORKSPACE, "data_src")
DATA_DST  = os.path.join(WORKSPACE, "data_dst")
ALIGNED_SRC = os.path.join(DATA_SRC, "aligned")
ALIGNED_DST = os.path.join(DATA_DST, "aligned")

# ── 模型 ──────────────────────────────────────────────────────────
# InsightFace 模型由 insightface 库自动管理，无需指定路径

# ── 对齐参数 ──────────────────────────────────────────────────────
ALIGN_SIZE  = 512    # 输出脸图尺寸（正方形边长）。BUG5修复：原为256，与pipeline不符

# ── 检测参数 ──────────────────────────────────────────────────────
DET_SIZE    = 640    # InsightFace 检测输入分辨率
CONF_THRES  = 0.5    # 人脸置信度阈值
MIN_FACE    = 80     # 最小人脸像素宽度（过滤过小的脸）

# ── 运行参数 ──────────────────────────────────────────────────────
THREADS     = 8      # 预留，当前流水线为单线程
