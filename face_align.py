"""
face_align.py — 脸部对齐模块（修复版）

修复记录：
  原版接收 landmarks 参数但未说明点数要求。
  LandmarksProcessor.get_transform_mat 只接受 68点。
  传入 InsightFace 106点会产生索引错位。

  修复：改为接收 kps_5pt（5关键点），使用相似变换，与 DFL WHOLE_FACE 等效。
  如仍需 68点路径（例如对接 dlib），保留原版接口作为备用。
"""

import cv2
import numpy as np


# DFL WHOLE_FACE 参考模板（5关键点 @ size=512）
# 与 pipeline.py 保持一致，两处共用同一常量
_DFL_WHOLE_FACE_REF_512 = np.float32([
    [159.0, 185.0],   # 左眼中心
    [353.0, 185.0],   # 右眼中心
    [256.0, 268.0],   # 鼻尖
    [179.0, 371.0],   # 左嘴角
    [333.0, 371.0],   # 右嘴角
])


class FaceAligner:
    def __init__(self, size: int = 512):
        self.size = size

    # ── 主接口：5关键点 → 相似变换（推荐，无索引错位风险）──────────
    def align_from_kps(self, image: np.ndarray, kps_5pt: np.ndarray):
        """
        参数：
            image   : BGR 图像
            kps_5pt : shape (5,2) float32，InsightFace face.kps
                      顺序 [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]
        返回：
            aligned : 对齐后的脸图 (size × size)
            mat     : (2,3) 变换矩阵
        """
        ref = _DFL_WHOLE_FACE_REF_512 * (self.size / 512.0)
        mat, _ = cv2.estimateAffinePartial2D(
            kps_5pt.reshape(5, 1, 2),
            ref.reshape(5, 1, 2),
            method=cv2.LMEDS,
        )
        if mat is None:
            raise RuntimeError("相似变换估计失败，关键点异常")

        aligned = cv2.warpAffine(
            image, mat, (self.size, self.size),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return aligned, mat.astype(np.float64)

    # ── 备用接口：68点 → DFL 原版变换（仅当已有标准68点时使用）──────
    def align_from_lmk68(self, image: np.ndarray, landmarks68: np.ndarray):
        """
        仅适用于标准 DFL/dlib 68点关键点。
        切勿传入 InsightFace 的 106点，会产生索引错位。
        """
        try:
            from facelib import LandmarksProcessor, FaceType
        except ImportError:
            raise ImportError("align_from_lmk68 需要 DFL facelib，请确认 DFL_ROOT 在 sys.path 中")

        if len(landmarks68) != 68:
            raise ValueError(
                f"align_from_lmk68 只接受 68点，收到 {len(landmarks68)} 点。"
                f"InsightFace 106点请用 align_from_kps。"
            )

        mat = LandmarksProcessor.get_transform_mat(
            landmarks68.astype(np.float32),
            self.size,
            FaceType.WHOLE_FACE,
        )
        aligned = cv2.warpAffine(
            image, mat, (self.size, self.size),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return aligned, mat
