"""
dflimg_writer.py — DFL 标准 JPEG 元数据写入器（BUG2 修复版）

修复记录：
  BUG2 原 footer 格式：pickle | 8字节len | b"DFLIMG"(6字节)
       加载器读 data[-8:] 作为 uint64 length
       但最后8字节 = b"DFLIMG"前6字节 + len高2字节 = 乱码 → 永远读不出来

  正确格式：JPEG | pickle | 8字节len（最后8字节就是 uint64 length）
  与 load_dfljpg 的 legacy footer 路径完全兼容。
"""

import pickle
import struct

import cv2


class DFLWriter:
    def write(self, image: "np.ndarray", dfl_dict: dict, output_path: str) -> None:
        """
        将图像 + 元数据写为 DFL 标准 JPEG。

        文件结构（DFL legacy footer，与 load_dfljpg 完全兼容）：
          [JPEG 字节] [pickle(dfl_dict)] [uint64 : len(pickle)]

        加载验证：
          data[-8:]            → meta_len (uint64)
          data[-8-meta_len:-8] → pickle.loads → dict  ✓
        """
        # 编码 JPEG
        ok, buf = cv2.imencode(
            ".jpg", image,
            [cv2.IMWRITE_JPEG_QUALITY, 100]
        )
        if not ok:
            raise RuntimeError(f"JPEG 编码失败: {output_path}")

        jpeg_bytes   = buf.tobytes()
        pickle_bytes = pickle.dumps(dfl_dict)

        with open(output_path, "wb") as f:
            f.write(jpeg_bytes)                          # JPEG 图像
            f.write(pickle_bytes)                        # pickle 元数据
            f.write(struct.pack("<Q", len(pickle_bytes)))  # 8字节 footer length
