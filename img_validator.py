"""
DFL Aligned Image Validator (Enhanced)
=======================================
工作目录：D:\\DeepFaceLab-Torch\\plugins\\  (脚本所在位置)
workspace：自动向上查找 D:\\DeepFaceLab-Torch\\workspace\\

用法示例（最简单）：
  python dfl_validator.py                        # 默认验证 workspace/data_src/aligned
  python dfl_validator.py --aligned src          # 同上，简写
  python dfl_validator.py --aligned dst          # 验证 data_dst/aligned
  python dfl_validator.py --aligned D:/path/to/aligned   # 绝对路径

带可视化 & 报告：
  python dfl_validator.py --aligned src --vis-dir vis_out --report result.json

简写映射（自动转换为 workspace 内的路径）：
  src  → workspace/data_src/aligned
  dst  → workspace/data_dst/aligned
"""

import argparse
import json
import pickle
import struct
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────
# 1. 智能路径解析（来自 validator_main.py）
# ─────────────────────────────────────────────────────────────────

ALIAS = {
    "src":         "data_src/aligned",
    "dst":         "data_dst/aligned",
    "source":      "data_src/frames",
    "destination": "data_dst/frames",
}


def find_workspace(start: Path) -> Optional[Path]:
    """从 start 向上最多 5 层，查找 workspace 目录"""
    current = start.resolve()
    for _ in range(5):
        if (current / "workspace").exists():
            return current / "workspace"
        if current.name == "workspace":
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def resolve_path(path_str: str, plugin_dir: Path, workspace: Optional[Path] = None) -> Path:
    """
    路径解析优先级：
      1. 绝对路径 → 直接返回
      2. 简写（src/dst/…） → 展开为 workspace 内路径
      3. 相对路径 → 先尝试 workspace/，再 cwd/
    """
    p = Path(path_str)
    if p.is_absolute():
        return p

    # 简写展开
    if path_str in ALIAS:
        path_str = ALIAS[path_str]

    if workspace:
        full = workspace / path_str
        if full.exists() or full.parent.exists():
            return full

    return Path.cwd() / path_str


# ─────────────────────────────────────────────────────────────────
# 2. DFLJPG 元数据解析（APP segment + legacy footer 双路兼容）
# ─────────────────────────────────────────────────────────────────

JPEG_SOI = b'\xFF\xD8'


def load_dfljpg(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except (IOError, OSError):
        return None

    if not data.startswith(JPEG_SOI):
        return None

    size = len(data)

    # ── 策略1：APP segment ──────────────────────────────────────
    i = 2
    while i + 3 < size:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        i += 2
        if marker in (0xD9, 0xDA):
            break
        if i + 2 > size:
            break
        seg_len   = struct.unpack('>H', data[i:i + 2])[0]
        seg_start = i + 2
        seg_end   = seg_start + seg_len - 2
        if seg_end > size:
            break
        seg = data[seg_start:seg_end]
        try:
            meta = pickle.loads(seg)
            if isinstance(meta, dict) and 'image_to_face_mat' in meta:
                return meta
        except Exception:
            pass
        i = seg_end

    # ── 策略2：Legacy footer pickle ─────────────────────────────
    for footer_size, fmt in ((4, '<I'), (8, '<Q')):
        if len(data) < footer_size:
            continue
        try:
            (meta_len,) = struct.unpack(fmt, data[-footer_size:])
            if meta_len <= 0 or meta_len > len(data) - footer_size:
                continue
            blob = data[-footer_size - meta_len:-footer_size]
            meta = pickle.loads(blob)
            if isinstance(meta, dict):
                return meta
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────────────────────────
# 3. 关键点 & 面部遮罩绘制
# ─────────────────────────────────────────────────────────────────

LANDMARK_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),
    (7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),
    (17,18),(18,19),(19,20),(20,21),
    (22,23),(23,24),(24,25),(25,26),
    (27,28),(28,29),(29,30),
    (30,31),(31,32),(32,33),(33,34),(34,35),
    (36,37),(37,38),(38,39),(39,40),(40,41),(41,36),
    (42,43),(43,44),(44,45),(45,46),(46,47),(47,42),
    (48,49),(49,50),(50,51),(51,52),(52,53),(53,54),
    (54,55),(55,56),(56,57),(57,58),(58,59),(59,48),
]

FACE_TYPE_ELLIPSE = {
    'whole_face': (0.45, 0.55),
    'full_face':  (0.42, 0.50),
    'half_face':  (0.38, 0.45),
}


def draw_landmarks(img: np.ndarray, lms: np.ndarray) -> np.ndarray:
    n = len(lms)
    for (x, y) in lms:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    for i, j in LANDMARK_EDGES:
        if i < n and j < n:
            cv2.line(img,
                     (int(lms[i][0]), int(lms[i][1])),
                     (int(lms[j][0]), int(lms[j][1])),
                     (255, 0, 255), 1)
    return img


def draw_face_mask(img: np.ndarray, face_type: str, alpha: float = 0.30) -> np.ndarray:
    h, w = img.shape[:2]
    rx_r, ry_r = FACE_TYPE_ELLIPSE.get(face_type, (0.42, 0.50))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2),
                (int(w * rx_r), int(h * ry_r)), 0, 0, 360, 255, -1)
    out = img.copy()
    out[mask == 255] = (out[mask == 255] * (1 - alpha)).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────
# 4. 核心验证逻辑
# ─────────────────────────────────────────────────────────────────

REQUIRED_KEYS       = ["face_type", "landmarks", "image_to_face_mat", "source_filename"]
VALID_FACE_TYPES    = {'half_face', 'full_face', 'whole_face', 'head', 'head_no_align'}
VALID_LANDMARK_COUNTS = {68, 98, 106}
COMMON_SIZES        = {256, 384, 512, 768, 1024, 1536, 2048}


def validate_aligned(path: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": path, "errors": [], "warnings": [],
        "meta": None, "shape": None, "face_type": None,
        "landmark_count": None, "lm_in_bounds": None,
        "mat_shape": None, "mat_det": None,
    }
    errors, warnings = result["errors"], result["warnings"]

    img = cv2.imread(path)
    if img is None:
        errors.append("IMAGE_UNREADABLE: 无法读取图像文件")
        return result

    h, w = img.shape[:2]
    result["shape"] = (h, w)
    if h != w:
        errors.append(f"NOT_SQUARE: 图像非正方形 {w}×{h}")
    if w not in COMMON_SIZES:
        warnings.append(f"UNUSUAL_SIZE: 非常见对齐尺寸 {w}px")

    meta = load_dfljpg(path)
    if meta is None:
        errors.append("META_MISSING: 无法解析 DFLJPG 元数据")
        return result
    result["meta"] = meta

    for k in REQUIRED_KEYS:
        if k not in meta:
            errors.append(f"KEY_MISSING: 缺少字段 '{k}'")

    face_type = meta.get("face_type")
    result["face_type"] = face_type
    if face_type is not None and face_type not in VALID_FACE_TYPES:
        warnings.append(f"UNKNOWN_FACE_TYPE: 未知类型 '{face_type}'")

    lms_raw = meta.get("landmarks", [])
    n_lms   = len(lms_raw)
    result["landmark_count"] = n_lms
    if n_lms == 0:
        errors.append("LM_EMPTY: 关键点列表为空")
    else:
        lms = np.array(lms_raw, dtype=np.float32)
        if n_lms not in VALID_LANDMARK_COUNTS:
            warnings.append(f"LM_COUNT_UNUSUAL: 关键点数量异常 {n_lms}（预期 68/98/106）")
        if not np.all(np.isfinite(lms)):
            errors.append("LM_NAN_INF: 关键点含 NaN/Inf")
        else:
            in_bounds = bool(np.all(
                (lms[:, 0] >= 0) & (lms[:, 0] < w) &
                (lms[:, 1] >= 0) & (lms[:, 1] < h)
            ))
            result["lm_in_bounds"] = in_bounds
            if not in_bounds:
                oob = int(np.sum(
                    (lms[:, 0] < 0) | (lms[:, 0] >= w) |
                    (lms[:, 1] < 0) | (lms[:, 1] >= h)
                ))
                warnings.append(f"LM_OUT_OF_BOUNDS: {oob} 个关键点超出图像范围")

    mat = np.array(meta.get("image_to_face_mat", []), dtype=np.float32)
    if mat.shape == (3, 3):
        mat = mat[:2]
    result["mat_shape"] = str(mat.shape)
    if mat.shape != (2, 3):
        errors.append(f"MAT_SHAPE_INVALID: 矩阵形状错误 {mat.shape}（应为 (2,3)）")
    else:
        if not np.all(np.isfinite(mat)):
            errors.append("MAT_NAN_INF: 仿射矩阵含 NaN/Inf")
        else:
            det = float(np.linalg.det(mat[:, :2]))
            result["mat_det"] = round(det, 6)
            if abs(det) < 1e-4:
                errors.append(f"MAT_DEGENERATE: 矩阵退化 det={det:.6f}")
            elif abs(det) < 0.01:
                warnings.append(f"MAT_NEAR_DEGENERATE: 行列式过小 det={det:.6f}")
            scale = abs(det) ** 0.5
            if not (0.05 < scale < 20):
                warnings.append(f"MAT_SCALE_SUSPECT: 缩放比异常 scale≈{scale:.4f}")

    return result


# ─────────────────────────────────────────────────────────────────
# 5. 可视化
# ─────────────────────────────────────────────────────────────────

def render_debug_image(path: str, result: Dict[str, Any]) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None
    meta = result.get("meta")
    if meta:
        lms_raw = meta.get("landmarks", [])
        if len(lms_raw) > 0:
            img = draw_landmarks(img, np.array(lms_raw, dtype=np.float32))
        ft = meta.get("face_type", "")
        if ft in FACE_TYPE_ELLIPSE:
            img = draw_face_mask(img, ft)
    has_err = bool(result["errors"])
    status  = "OK" if not has_err else f"ERR x{len(result['errors'])}"
    color   = (0, 200, 0) if not has_err else (30, 30, 220)
    cv2.putText(img, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if result.get("face_type"):
        cv2.putText(img, result["face_type"], (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    if result.get("landmark_count"):
        cv2.putText(img, f"LM:{result['landmark_count']}", (8, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 220, 160), 1, cv2.LINE_AA)
    return img


# ─────────────────────────────────────────────────────────────────
# 6. 批量扫描 & 统计
# ─────────────────────────────────────────────────────────────────

def scan_directory(aligned_dir: str, vis_dir: Optional[str] = None) -> List[Dict]:
    aligned_path = Path(aligned_dir)
    jpgs = sorted(
        list(aligned_path.glob("*.jpg")) +
        list(aligned_path.glob("*.jpeg"))
    )
    if not jpgs:
        print(f"[警告] 未找到 .jpg/.jpeg 文件: {aligned_dir}")
        return []

    print(f"\n[扫描] 共 {len(jpgs)} 张图像  →  {aligned_dir}")
    if vis_dir:
        Path(vis_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for idx, jpg in enumerate(jpgs, 1):
        res = validate_aligned(str(jpg))
        results.append(res)
        tag      = "✓" if not res["errors"] else "✗"
        err_hint = f"  [{res['errors'][0]}]" if res["errors"] else ""
        print(f"  {idx:>5}/{len(jpgs)}  {tag}  {jpg.name}{err_hint}")
        if vis_dir:
            dbg = render_debug_image(str(jpg), res)
            if dbg is not None:
                cv2.imwrite(str(Path(vis_dir) / jpg.name), dbg)
    return results


def print_summary(results: List[Dict]) -> None:
    total = len(results)
    ok    = sum(1 for r in results if not r["errors"])
    err   = total - ok
    warn  = sum(1 for r in results if r["warnings"])
    print("\n" + "═" * 60)
    print(f"  验证完成   总计 {total}   ✓ 通过 {ok}   ✗ 错误 {err}   ⚠ 警告 {warn}")
    print("═" * 60)

    err_codes: Counter = Counter()
    for r in results:
        for e in r["errors"]:
            err_codes[e.split(":")[0]] += 1
    if err_codes:
        print("\n  错误分布：")
        for code, cnt in err_codes.most_common():
            print(f"    {code:<32} × {cnt}")

    ft_cnt: Counter = Counter(r["face_type"] for r in results if r["face_type"])
    if ft_cnt:
        print("\n  face_type 分布：")
        for ft, cnt in ft_cnt.most_common():
            print(f"    {ft:<22} × {cnt}")

    lm_cnt: Counter = Counter(r["landmark_count"] for r in results
                               if r["landmark_count"] is not None)
    if lm_cnt:
        print("\n  关键点数量分布：")
        for n, cnt in lm_cnt.most_common():
            print(f"    {n} 点   × {cnt}")
    print()


# ─────────────────────────────────────────────────────────────────
# 7. CLI（智能路径 + 友好提示）
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DFL Aligned Image Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python dfl_validator.py                            # 默认 src（workspace/data_src/aligned）
  python dfl_validator.py --aligned src              # 同上
  python dfl_validator.py --aligned dst              # workspace/data_dst/aligned
  python dfl_validator.py --aligned D:/path/aligned  # 绝对路径
  python dfl_validator.py --aligned src --vis-dir vis_out --report result.json

简写：src=data_src/aligned  dst=data_dst/aligned
        """
    )
    p.add_argument("--aligned",  default="src",
                   help="aligned 图像目录（默认: src）")
    p.add_argument("--source",   default=None,
                   help="原始帧目录（可选）")
    p.add_argument("--vis-dir",  default=None,
                   help="可视化调试图像输出目录（可选）")
    p.add_argument("--report",   default=None,
                   help="结果保存为 JSON 文件（可选，默认不保存）")
    return p.parse_args()


def _serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    return obj


def main():
    try:
        args     = parse_args()
        plugin_dir = Path(__file__).parent.resolve()
        workspace  = find_workspace(plugin_dir)

        # 提示发现的 workspace
        if workspace:
            print(f"[工作目录] workspace → {workspace}")
        else:
            print(f"[工作目录] 未找到 workspace，使用当前目录 → {Path.cwd()}")

        # ── 解析 aligned 路径 ───────────────────────────────────
        aligned_path = resolve_path(args.aligned, plugin_dir, workspace)
        if not aligned_path.exists():
            print(f"\n[错误] aligned 目录不存在: {aligned_path}")
            print(f"\n  可用简写：")
            if workspace:
                for alias, rel in [("src", "data_src/aligned"),
                                    ("dst", "data_dst/aligned")]:
                    full = workspace / rel
                    exists_tag = "✓" if full.exists() else "✗ (不存在)"
                    print(f"    --aligned {alias:<6}  →  {full}  {exists_tag}")
            print(f"\n  或使用绝对路径，例如：")
            print(f"    --aligned D:/DeepFaceLab-Torch/workspace/data_src/aligned")
            input("\n按回车退出...")
            sys.exit(1)

        print(f"[aligned]  {aligned_path}")

        # ── 解析 vis-dir 路径（如果提供）──────────────────────
        vis_dir = None
        if args.vis_dir:
            vis_path = resolve_path(args.vis_dir, plugin_dir, workspace)
            vis_dir  = str(vis_path)
            print(f"[vis-dir]  {vis_path}")

        # ── 运行验证 ────────────────────────────────────────────
        results = scan_directory(str(aligned_path), vis_dir=vis_dir)
        if not results:
            input("按回车退出...")
            sys.exit(0)

        print_summary(results)

        # ── 保存报告 ────────────────────────────────────────────
        if args.report:
            report_path = resolve_path(args.report, plugin_dir, workspace)
            exportable  = []
            for r in results:
                row = {k: v for k, v in r.items() if k != "meta"}
                row["meta_keys"] = list(r["meta"].keys()) if r["meta"] else []
                exportable.append(_serialize(row))
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(exportable, f, ensure_ascii=False, indent=2)
            print(f"[报告] 已保存 → {report_path}")

        input("按回车退出...")

    except Exception:
        print("\n[致命错误]")
        traceback.print_exc()
        input("按回车退出...")


if __name__ == "__main__":
    main()
