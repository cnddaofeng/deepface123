"""
dfl_inspect.py — DFL 图像全参数直读工具
用法：
  python dfl_inspect.py D:/workspace/data_src/aligned
  python dfl_inspect.py D:/workspace/data_dst/aligned
  python dfl_inspect.py D:/workspace/data_src/aligned --vis-dir out_vis
"""
import argparse, json, math, pickle, struct, sys, traceback
from pathlib import Path
import cv2, numpy as np

# ── DFLJPG 解析 ──────────────────────────────────────────────────
def load_dfljpg(path):
    with open(path, 'rb') as f: data = f.read()
    if not data.startswith(b'\xFF\xD8'): return None
    i, size = 2, len(data)
    while i + 3 < size:
        if data[i] != 0xFF: break
        marker = data[i+1]; i += 2
        if marker in (0xD9, 0xDA): break
        if i+2 > size: break
        slen = struct.unpack('>H', data[i:i+2])[0]
        ss, se = i+2, i+2+slen-2
        if se > size: break
        try:
            m = pickle.loads(data[ss:se])
            if isinstance(m, dict) and 'image_to_face_mat' in m: return m
        except: pass
        i = se
    for fsz, fmt in ((4,'<I'),(8,'<Q')):
        if len(data) < fsz: continue
        try:
            (ml,) = struct.unpack(fmt, data[-fsz:])
            if 0 < ml <= len(data)-fsz:
                m = pickle.loads(data[-fsz-ml:-fsz])
                if isinstance(m, dict): return m
        except: pass
    return None

# ── 矩阵分解 ─────────────────────────────────────────────────────
def decompose(mat):
    a,b,c,d = mat[0,0],mat[0,1],mat[1,0],mat[1,1]
    scale = math.sqrt(a*a + b*b)
    angle = math.degrees(math.atan2(b, a))
    det   = a*d - b*c
    shear = abs(math.sqrt(c*c+d*d) - scale) / (scale+1e-9) > 0.01
    return dict(scale=scale, angle=angle,
                tx=mat[0,2], ty=mat[1,2],
                det=det, has_shear=shear)

# ── 凸包面积比 ───────────────────────────────────────────────────
def hull_ratio(lms, w, h):
    if len(lms) < 3: return 0.0
    hull = cv2.convexHull(lms.astype(np.float32).reshape(-1,1,2))
    return cv2.contourArea(hull) / (w*h)

# ── 关键点绘制（68点连线，其他只画点）───────────────────────────
EDGES68 = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),
    (10,11),(11,12),(12,13),(13,14),(14,15),(15,16),
    (17,18),(18,19),(19,20),(20,21),(22,23),(23,24),(24,25),(25,26),
    (27,28),(28,29),(29,30),(30,31),(31,32),(32,33),(33,34),(34,35),
    (36,37),(37,38),(38,39),(39,40),(40,41),(41,36),
    (42,43),(43,44),(44,45),(45,46),(46,47),(47,42),
    (48,49),(49,50),(50,51),(51,52),(52,53),(53,54),
    (54,55),(55,56),(56,57),(57,58),(58,59),(59,48),
    (60,61),(61,62),(62,63),(63,64),(64,65),(65,66),(66,67),(67,60),
]
def draw_lms(img, lms):
    n = len(lms)
    for x,y in lms: cv2.circle(img,(int(x),int(y)),2,(0,255,0),-1)
    if n == 68:
        for i,j in EDGES68:
            cv2.line(img,(int(lms[i][0]),int(lms[i][1])),
                        (int(lms[j][0]),int(lms[j][1])),(255,0,255),1)
    return img

# ── 单张图全参数读取 ─────────────────────────────────────────────
def inspect_one(path):
    img = cv2.imread(path)
    if img is None: return None, None, "无法读取图像"

    h, w = img.shape[:2]
    meta = load_dfljpg(path)
    if meta is None: return img, None, "无法解析DFLJPG元数据"

    mat_raw = np.array(meta.get('image_to_face_mat',[]), dtype=np.float64)
    if mat_raw.shape == (3,3): mat_raw = mat_raw[:2]
    mat_ok = mat_raw.shape == (2,3)

    lms_raw = meta.get('landmarks', [])
    lms = np.array(lms_raw, dtype=np.float32) if len(lms_raw) else None

    dec = decompose(mat_raw) if mat_ok else None
    fr  = hull_ratio(lms, w, h) if lms is not None else None

    info = {
        # ── 图像 ──
        'img_size':        f"{w}×{h}",
        'img_square':      w == h,
        # ── 元数据所有字段 ──
        'meta_keys':       list(meta.keys()),
        'face_type':       meta.get('face_type'),
        'source_filename': meta.get('source_filename'),
        'lm_count':        len(lms_raw),
        # ── 仿射矩阵原始值 ──
        'mat_raw':         mat_raw.tolist() if mat_ok else '解析失败',
        'mat_shape':       str(mat_raw.shape),
        # ── 矩阵分解 ──
        'scale':           round(dec['scale'], 6) if dec else None,
        'angle_deg':       round(dec['angle'], 4)  if dec else None,
        'tx':              round(dec['tx'], 4)      if dec else None,
        'ty':              round(dec['ty'], 4)      if dec else None,
        'det':             round(dec['det'], 6)     if dec else None,
        'has_shear':       dec['has_shear']         if dec else None,
        # ── 脸部占比 ──
        'face_hull_ratio': round(fr, 4) if fr is not None else None,
        'face_hull_pct':   f"{fr*100:.2f}%" if fr is not None else None,
        # ── 关键点范围 ──
        'lm_x_range':      [round(float(lms[:,0].min()),2),
                             round(float(lms[:,0].max()),2)] if lms is not None else None,
        'lm_y_range':      [round(float(lms[:,1].min()),2),
                             round(float(lms[:,1].max()),2)] if lms is not None else None,
        'lm_all_in_image': (bool(np.all((lms[:,0]>=0)&(lms[:,0]<w)&
                                        (lms[:,1]>=0)&(lms[:,1]<h)))
                            if lms is not None else None),
    }

    # 其他 meta 字段原样输出
    skip = {'landmarks','image_to_face_mat','face_type','source_filename'}
    for k,v in meta.items():
        if k not in skip:
            try:    info[f'meta_{k}'] = v.tolist() if hasattr(v,'tolist') else v
            except: info[f'meta_{k}'] = str(v)

    return img, lms, info

# ── 可视化标注图 ─────────────────────────────────────────────────
def render_vis(img, lms, info):
    out = img.copy()
    if lms is not None: draw_lms(out, lms)

    # 凸包轮廓
    if lms is not None and len(lms) >= 3:
        hull = cv2.convexHull(lms.astype(np.float32).reshape(-1,1,2))
        cv2.polylines(out,[hull.astype(np.int32)],True,(0,200,255),1)

    # 文字覆盖
    lines = [
        f"face_type : {info.get('face_type')}",
        f"size      : {info.get('img_size')}  square={info.get('img_square')}",
        f"lm_count  : {info.get('lm_count')}",
        f"scale     : {info.get('scale')}",
        f"angle     : {info.get('angle_deg')}°",
        f"tx/ty     : {info.get('tx')} / {info.get('ty')}",
        f"det       : {info.get('det')}",
        f"shear     : {info.get('has_shear')}",
        f"hull_ratio: {info.get('face_hull_pct')}",
        f"lm_in_img : {info.get('lm_all_in_image')}",
    ]
    pad = 8
    lh  = 18
    bh  = pad*2 + lh*len(lines)
    bar = np.zeros((bh, out.shape[1], 3), np.uint8)
    for i,l in enumerate(lines):
        cv2.putText(bar, l, (pad, pad+lh*(i+1)-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,230,200), 1, cv2.LINE_AA)
    return np.vstack([out, bar])

# ── 打印单张结果 ─────────────────────────────────────────────────
def print_info(name, info):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    for k,v in info.items():
        print(f"  {k:<22} = {v}")

# ── 批量扫描 ─────────────────────────────────────────────────────
def scan(aligned_dir, vis_dir=None, report=None):
    p = Path(aligned_dir)
    jpgs = sorted(list(p.glob("*.jpg")) + list(p.glob("*.jpeg")))
    if not jpgs:
        print(f"[错误] 目录中无 .jpg 文件: {p}"); return

    print(f"\n[扫描] {len(jpgs)} 张图像  →  {p}")
    if vis_dir: Path(vis_dir).mkdir(parents=True, exist_ok=True)

    all_info = []
    for jpg in jpgs:
        img, lms, info = inspect_one(str(jpg))
        if info is None: print(f"  [跳过] {jpg.name}"); continue

        print_info(jpg.name, info)
        all_info.append({"file": jpg.name, **info})

        if vis_dir and img is not None:
            vis = render_vis(img, lms, info)
            cv2.imwrite(str(Path(vis_dir)/jpg.name), vis)

    # ── 横向汇总 ─────────────────────────────────────────────────
    if len(all_info) > 1:
        scales = [x['scale'] for x in all_info if x.get('scale')]
        ratios = [x['face_hull_ratio'] for x in all_info if x.get('face_hull_ratio')]
        angles = [x['angle_deg'] for x in all_info if x.get('angle_deg') is not None]
        print(f"\n{'═'*60}")
        print(f"  汇总  {len(all_info)} 张")
        print(f"{'═'*60}")
        if scales:
            print(f"  scale     均值={np.mean(scales):.5f}  "
                  f"最大={max(scales):.5f}  最小={min(scales):.5f}")
        if ratios:
            print(f"  hull_ratio 均值={np.mean(ratios):.4f}  "
                  f"({np.mean(ratios)*100:.2f}%)  "
                  f"最大={max(ratios):.4f}  最小={min(ratios):.4f}")
        if angles:
            print(f"  angle     均值={np.mean(angles):.3f}°  "
                  f"最大={max(angles):.3f}°  最小={min(angles):.3f}°")
        ft = [x['face_type'] for x in all_info if x.get('face_type')]
        from collections import Counter
        if ft: print(f"  face_type  {dict(Counter(ft))}")
        lmc = [x['lm_count'] for x in all_info if x.get('lm_count')]
        if lmc: print(f"  lm_count   {dict(Counter(lmc))}")

    if report:
        with open(report,'w',encoding='utf-8') as f:
            json.dump(all_info, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[报告] → {report}")

# ── CLI ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('aligned', help='aligned 图像目录')
    ap.add_argument('--vis-dir', default=None)
    ap.add_argument('--report',  default=None)
    args = ap.parse_args()
    try:
        scan(args.aligned, args.vis_dir, args.report)
    except Exception:
        traceback.print_exc()
    input("\n按回车退出...")
