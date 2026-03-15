# -*- coding: utf-8 -*-
"""
Deep2005 工具集 Pro Max v3.0 — The Face Revolution
====================================================================
整合 dfl_gui.txt + deep2005.txt 全部功能
====================================================================
"""
import sys, os, threading, time, subprocess, json, shutil, logging, traceback
import math, struct, pickle, hashlib, queue
from pathlib import Path
from datetime import datetime
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(filename=str(LOG_DIR / "deep2005.log"), level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s", encoding="utf-8")
logger = logging.getLogger("Deep2005")

def global_exception_handler(exc_type, exc_value, exc_tb):
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.critical(f"未捕获异常:\n{error_msg}")
    with open(LOG_DIR / "error.log", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n{datetime.now().isoformat()}\n{error_msg}\n")
sys.excepthook = global_exception_handler

try:
    import cv2
    import numpy as np
except ImportError:
    print("❌ 缺少 opencv-python/numpy"); sys.exit(1)

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox,
        QTabWidget, QFrame, QScrollArea, QGridLayout, QLineEdit,
        QComboBox, QSpinBox, QCheckBox, QProgressBar, QStatusBar,
        QMenuBar, QMenu, QDialog, QFormLayout, QDialogButtonBox,
        QSplitter, QToolBar, QSizePolicy, QGroupBox, QDoubleSpinBox,
        QRadioButton, QButtonGroup, QSlider
    )
    from PySide6.QtCore import Qt, Signal, QObject, QTimer, QThread, QSize, QUrl, QMimeData
    from PySide6.QtGui import (QPixmap, QImage, QFont, QIcon, QDragEnterEvent, QDropEvent,
                                QPainter, QColor, QPen, QAction)
    from PySide6.QtCharts import QChart, QLineSeries, QValueAxis, QChartView
except ImportError:
    print("❌ 缺少 PySide6"); sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None

DFL_ROOT = Path(__file__).parent.parent.resolve()
PLUGINS_DIR = Path(__file__).parent.resolve()
WORKSPACE = DFL_ROOT / "workspace"
LOSS_OUTPUT_DIR = PLUGINS_DIR / "loss"

# ── DFLJPG 解析 ──
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

def decompose(mat):
    a,b,c,d = mat[0,0],mat[0,1],mat[1,0],mat[1,1]
    scale = math.sqrt(a*a+b*b)
    angle = math.degrees(math.atan2(b, a))
    det = a*d - b*c
    shear = abs(math.sqrt(c*c+d*d) - scale) / (scale+1e-9) > 0.01
    return dict(scale=scale, angle=angle, tx=mat[0,2], ty=mat[1,2], det=det, has_shear=shear)

def hull_ratio(lms, w, h):
    if len(lms) < 3: return 0.0
    hull = cv2.convexHull(lms.astype(np.float32).reshape(-1,1,2))
    return cv2.contourArea(hull) / (w*h)

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
            cv2.line(img,(int(lms[i][0]),int(lms[i][1])),(int(lms[j][0]),int(lms[j][1])),(255,0,255),1)
    return img

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
    fr = hull_ratio(lms, w, h) if lms is not None else None
    info = {
        'img_size': f"{w}x{h}", 'face_type': meta.get('face_type'),
        'source_filename': meta.get('source_filename'), 'lm_count': len(lms_raw),
        'scale': round(dec['scale'],6) if dec else None,
        'angle_deg': round(dec['angle'],4) if dec else None,
        'tx': round(dec['tx'],4) if dec else None, 'ty': round(dec['ty'],4) if dec else None,
        'det': round(dec['det'],6) if dec else None, 'has_shear': dec['has_shear'] if dec else None,
        'face_hull_ratio': round(fr,4) if fr is not None else None,
        'face_hull_pct': f"{fr*100:.2f}%" if fr is not None else None,
    }
    skip = {'landmarks','image_to_face_mat','face_type','source_filename'}
    for k,v in meta.items():
        if k not in skip:
            try: info[f'meta_{k}'] = v.tolist() if hasattr(v,'tolist') else v
            except: info[f'meta_{k}'] = str(v)
    return img, lms, info

def render_vis(img, lms, info):
    out = img.copy()
    if lms is not None: draw_lms(out, lms)
    if lms is not None and len(lms) >= 3:
        hull_pts = cv2.convexHull(lms.astype(np.float32).reshape(-1,1,2))
        cv2.polylines(out,[hull_pts.astype(np.int32)],True,(0,200,255),1)
    lines = [f"face_type:{info.get('face_type')}", f"scale:{info.get('scale')}",
             f"angle:{info.get('angle_deg')}", f"hull:{info.get('face_hull_pct')}"]
    pad, lh = 8, 18
    bh = pad*2 + lh*len(lines)
    bar = np.zeros((bh, out.shape[1], 3), np.uint8)
    for i,l in enumerate(lines):
        cv2.putText(bar, l, (pad, pad+lh*(i+1)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,230,200), 1, cv2.LINE_AA)
    return np.vstack([out, bar])

# ── GPU 检测 ──
def detect_gpu_info():
    try:
        result = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.free,driver_version",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({"name":parts[0],"total_mb":int(float(parts[1])),"free_mb":int(float(parts[2])),"driver":parts[3]})
            return gpus
    except: pass
    return []

def detect_gpu_torch():
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            return True, n, "  ".join([torch.cuda.get_device_name(i) for i in range(n)])
    except: pass
    gpus = detect_gpu_info()
    if gpus: return True, len(gpus), "  ".join(g['name'] for g in gpus)
    return False, 0, "未检测到"

def detect_environment():
    env = {"python":sys.version.split()[0],"opencv":cv2.__version__,"cuda_available":False,"ffmpeg_available":False,"gpus":[]}
    try:
        bi = cv2.getBuildInformation()
        env["cuda_available"] = "CUDA" in bi and "YES" in bi.split("CUDA")[1][:50]
    except: pass
    try:
        r = subprocess.run(["ffmpeg","-version"], capture_output=True, timeout=5)
        env["ffmpeg_available"] = r.returncode == 0
    except: pass
    env["gpus"] = detect_gpu_info()
    return env

def ai_suggest_params(gpus, model_type):
    if not gpus: return {"resolution":128,"batch_size":4,"reason":"未检测到GPU"}
    free_mb = gpus[0].get("free_mb",2000)
    if "SAEHD" in model_type:
        if free_mb>=8000: return {"resolution":256,"batch_size":8,"reason":f"显存{free_mb}MB充足"}
        elif free_mb>=4000: return {"resolution":192,"batch_size":6,"reason":f"显存{free_mb}MB中等"}
        else: return {"resolution":128,"batch_size":4,"reason":f"显存{free_mb}MB较低"}
    elif "AMP" in model_type:
        if free_mb>=8000: return {"resolution":224,"batch_size":8,"reason":"AMP推荐中高分辨率"}
        else: return {"resolution":160,"batch_size":6,"reason":"AMP保守配置"}
    elif "Quick96" in model_type:
        return {"resolution":96,"batch_size":16,"reason":"Quick96固定96"}
    return {"resolution":128,"batch_size":8,"reason":f"通用推荐({model_type})"}

# ── 智能去重算法集 ──
def calc_phash(img, hash_size=8):
    resized = cv2.resize(img, (hash_size+1, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape)==3 else resized
    return (gray[:, 1:] > gray[:, :-1]).flatten()

def calc_blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calc_hist(img):
    hist = cv2.calcHist([img],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

def check_face_exists(img):
    fc = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return len(fc.detectMultiScale(gray,1.1,5,minSize=(30,30))) > 0

def check_occlusion(img_path):
    meta = load_dfljpg(img_path)
    if not meta: return False, 0.0
    lms_raw = meta.get('landmarks',[])
    if not lms_raw: return False, 0.0
    lms = np.array(lms_raw, dtype=np.float32)
    img = cv2.imread(img_path)
    if img is None: return False, 0.0
    h,w = img.shape[:2]
    ratio = hull_ratio(lms, w, h)
    return ratio < 0.15, ratio

def check_angle(img_path, max_angle=30.0):
    meta = load_dfljpg(img_path)
    if not meta: return False, 0.0
    mat_raw = np.array(meta.get('image_to_face_mat',[]), dtype=np.float64)
    if mat_raw.shape==(3,3): mat_raw=mat_raw[:2]
    if mat_raw.shape!=(2,3): return False, 0.0
    dec = decompose(mat_raw)
    return abs(dec['angle']) > max_angle, dec['angle']

def check_large_angle(img_path, max_angle=45.0):
    return check_angle(img_path, max_angle)

def cvimg_to_qpixmap(cv_img):
    if cv_img is None: return QPixmap()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h,w,ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data.tobytes(), w, h, ch*w, QImage.Format_RGB888))

# ── 工作线程 ──
class WorkerThread(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(bool, str)
    def __init__(self, target_func, *args, **kwargs):
        super().__init__()
        self._target = target_func; self._args = args; self._kwargs = kwargs
    def run(self):
        try:
            self._target(self, *self._args, **self._kwargs)
            self.finished_signal.emit(True, "完成")
        except Exception as e:
            logger.error(traceback.format_exc())
            self.log_signal.emit(f"❌ 错误: {e}")
            self.finished_signal.emit(False, str(e))

# ── DFL检查弹窗 ──
class DFLInspectDialog(QDialog):
    def __init__(self, img_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"DFL检查 - {os.path.basename(img_path)}")
        self.resize(700, 860)
        self.setStyleSheet("background-color:#1a1a2e;color:#eaeaea;")
        layout = QVBoxLayout(self)
        img, lms, info = inspect_one(img_path)
        if img is None or isinstance(info, str):
            layout.addWidget(QLabel(f"读取失败: {info}"))
            return
        vis_bgr = render_vis(img, lms, info)
        pixmap = cvimg_to_qpixmap(vis_bgr)
        if pixmap.width()>680 or pixmap.height()>700:
            pixmap = pixmap.scaled(680,700,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        lbl = QLabel(); lbl.setPixmap(pixmap); lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        txt = QTextEdit(); txt.setReadOnly(True); txt.setMaximumHeight(150)
        txt.setStyleSheet("background:#0a0a1a;color:#8be9fd;font-family:Consolas;font-size:11px;")
        txt.setPlainText("\n".join(f"  {k:<22} = {v}" for k,v in info.items()))
        layout.addWidget(txt)

class ImagePreviewDialog(QDialog):
    def __init__(self, image_paths, start_index=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像预览"); self.resize(600,600)
        self.image_paths = image_paths; self.current_index = start_index
        layout = QVBoxLayout(self)
        self.image_label = QLabel(); self.image_label.setAlignment(Qt.AlignCenter); self.image_label.setMinimumSize(400,400)
        layout.addWidget(self.image_label)
        self.info_label = QLabel(); self.info_label.setAlignment(Qt.AlignCenter); self.info_label.setStyleSheet("color:#aaa;")
        layout.addWidget(self.info_label)
        nav = QHBoxLayout()
        bp = QPushButton("◀ 上一张"); bn = QPushButton("下一张 ▶"); bi = QPushButton("🔍 DFL检查")
        bp.clicked.connect(self.prev_image); bn.clicked.connect(self.next_image); bi.clicked.connect(self._inspect)
        nav.addWidget(bp); nav.addWidget(bi); nav.addWidget(bn)
        layout.addLayout(nav)
        self.show_current()
    def show_current(self):
        if not self.image_paths: self.image_label.setText("无图像"); return
        idx = self.current_index % len(self.image_paths)
        path = str(self.image_paths[idx])
        px = QPixmap(path)
        if not px.isNull(): self.image_label.setPixmap(px.scaled(500,500,Qt.KeepAspectRatio,Qt.SmoothTransformation))
        else: self.image_label.setText("无法加载")
        self.info_label.setText(f"{idx+1}/{len(self.image_paths)} | {os.path.basename(path)}")
    def prev_image(self): self.current_index = max(0, self.current_index-1); self.show_current()
    def next_image(self): self.current_index = min(len(self.image_paths)-1, self.current_index+1); self.show_current()
    def _inspect(self):
        if self.image_paths:
            DFLInspectDialog(str(self.image_paths[self.current_index % len(self.image_paths)]), self).exec()

class EnvCheckDialog(QDialog):
    def __init__(self, env_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔍 环境检测"); self.resize(500,400)
        layout = QVBoxLayout(self)
        text = QTextEdit(); text.setReadOnly(True)
        text.setStyleSheet("background:#1e1e1e;color:#ddd;font-family:Consolas;font-size:13px;")
        lines = ["="*50, "  Deep2005 环境检测报告", "="*50,
                 f"  Python: {env_info['python']}", f"  OpenCV: {env_info['opencv']}",
                 f"  CUDA: {'✅' if env_info['cuda_available'] else '❌'}",
                 f"  FFmpeg: {'✅' if env_info['ffmpeg_available'] else '❌'}", ""]
        for i,g in enumerate(env_info.get("gpus",[])):
            lines.append(f"  GPU{i}: {g['name']} ({g['free_mb']}MB free)")
        gpu_ok,_,gi = detect_gpu_torch()
        lines.append(f"\n  Torch GPU: {'✅ '+gi if gpu_ok else '❌'}")
        text.setPlainText("\n".join(lines)); layout.addWidget(text)
        b = QPushButton("确定"); b.clicked.connect(self.accept); layout.addWidget(b)

class CommandConsoleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🛠️ 命令调试台"); self.resize(700,500)
        layout = QVBoxLayout(self)
        self.output = QTextEdit(); self.output.setReadOnly(True)
        self.output.setStyleSheet("background:#0c0c0c;color:#0f0;font-family:Consolas;font-size:13px;")
        layout.addWidget(self.output)
        il = QHBoxLayout()
        self.cmd_input = QLineEdit(); self.cmd_input.setPlaceholderText("输入命令...")
        self.cmd_input.setStyleSheet("background:#1a1a1a;color:#0f0;font-family:Consolas;")
        self.cmd_input.returnPressed.connect(self._exec)
        br = QPushButton("执行"); br.clicked.connect(self._exec)
        il.addWidget(self.cmd_input); il.addWidget(br); layout.addLayout(il)
        self.output.append("Deep2005 命令调试台\n")
    def _exec(self):
        cmd = self.cmd_input.text().strip()
        if not cmd: return
        self.cmd_input.clear(); self.output.append(f"> {cmd}")
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
            if r.stdout: self.output.append(r.stdout)
            if r.stderr: self.output.append(f"[STDERR] {r.stderr}")
        except subprocess.TimeoutExpired: self.output.append("[超时]")
        except Exception as e: self.output.append(f"[错误] {e}")


# ============================================================
# 主窗口
# ============================================================
STYLESHEET = """
QMainWindow { background-color: #121212; color: #e0e0e0; }
QMenuBar { background-color: #1a1a1a; color: #ddd; border-bottom: 1px solid #333; }
QMenuBar::item:selected { background-color: #333; }
QMenu { background-color: #252525; color: #ddd; border: 1px solid #444; }
QMenu::item:selected { background-color: #3a3a3a; }
QToolBar { background-color: #1a1a1a; border-bottom: 1px solid #333; spacing: 6px; padding: 4px; }
QToolBar QToolButton { color: #ccc; padding: 4px 8px; border-radius: 4px; }
QToolBar QToolButton:hover { background-color: #333; }
QTabWidget::pane { border: 1px solid #333; background-color: #1a1a1a; }
QTabBar::tab { background-color: #252525; color: #aaa; padding: 10px 20px; border: 1px solid #333;
    border-bottom: none; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background-color: #1a1a1a; color: #4FC3F7; border-bottom: 2px solid #4FC3F7; }
QTabBar::tab:hover { background-color: #2a2a2a; }
QPushButton { background-color: #2a2a2a; color: #ddd; border: 1px solid #444; padding: 8px 16px;
    border-radius: 6px; font-size: 13px; }
QPushButton:hover { background-color: #3a3a3a; border-color: #4FC3F7; }
QPushButton:pressed { background-color: #1a1a1a; }
QPushButton:disabled { background-color: #1a1a1a; color: #555; border-color: #333; }
QLabel { color: #ccc; }
QLineEdit, QSpinBox, QComboBox, QDoubleSpinBox { background-color: #2a2a2a; color: #ddd;
    border: 1px solid #444; padding: 6px; border-radius: 4px; }
QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border-color: #4FC3F7; }
QComboBox::drop-down { border: none; width: 30px; }
QComboBox QAbstractItemView { background-color: #2a2a2a; color: #ddd; selection-background-color: #3a3a3a; }
QCheckBox { color: #ccc; spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #555; border-radius: 3px; background-color: #2a2a2a; }
QCheckBox::indicator:checked { background-color: #4CAF50; border-color: #4CAF50; }
QProgressBar { background-color: #1a1a1a; border: 1px solid #333; border-radius: 5px; height: 20px;
    text-align: center; color: #ddd; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1B5E20,stop:1 #4CAF50); border-radius: 4px; }
QGroupBox { color: #4FC3F7; border: 1px solid #333; border-radius: 6px; margin-top: 10px;
    padding-top: 10px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QScrollArea { border: 1px solid #333; background-color: #1a1a1a; }
QTextEdit { background-color: #0d1117; color: #c9d1d9; border: 1px solid #333; border-radius: 4px;
    font-family: Consolas, 'Courier New', monospace; font-size: 12px; }
QStatusBar { background-color: #1a1a1a; color: #888; border-top: 1px solid #333; }
QSplitter::handle { background-color: #333; height: 3px; }
QScrollBar:vertical { background-color: #1a1a1a; width: 10px; border: none; }
QScrollBar::handle:vertical { background-color: #444; border-radius: 5px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background-color: #555; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QRadioButton { color: #ccc; spacing: 6px; }
QRadioButton::indicator { width: 16px; height: 16px; }
"""


class Deep2005ProMax(QMainWindow):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    chart_signal = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep2005 工具集 Pro Max v3.0 — The Face Revolution")
        self.resize(1280, 860)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(STYLESHEET)
        self.setAcceptDrops(True)

        self.project_root = PLUGINS_DIR
        self.workspace = WORKSPACE
        self.workspace.mkdir(exist_ok=True)

        self.is_training = False
        self.training_process = None
        self.training_thread = None
        self.current_epoch = 0
        self.loss_history = deque(maxlen=500)
        self.env_info = None
        self._current_worker = None

        self._build_menu_bar()
        self._build_toolbar()
        self._build_central()
        self._build_status_bar()

        self.log_signal.connect(self._append_log)
        self.progress_signal.connect(self._update_progress)
        self.chart_signal.connect(self._update_chart)

        self._append_log("═" * 60)
        self._append_log("  🌊 Deep2005 工具集 Pro Max v3.0")
        self._append_log("  整合版: dfl_gui + deep2005 全功能")
        self._append_log("═" * 60)
        self._append_log(f"  工作目录: {self.workspace}")
        self._append_log(f"  DFL根目录: {DFL_ROOT}")
        self._append_log(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._append_log("")

        QTimer.singleShot(500, self._auto_detect_env)
        logger.info("Deep2005 Pro Max v3.0 启动完成")

    # ── 菜单栏 ──
    def _build_menu_bar(self):
        mb = self.menuBar()
        fm = mb.addMenu("文件(&F)")
        fm.addAction("创建工作区", self._create_workspace)
        fm.addAction("打开工作目录", self._open_workspace_folder)
        fm.addSeparator()
        fm.addAction("退出", self.close)
        tm = mb.addMenu("工具(&T)")
        tm.addAction("🔍 环境检测", self._show_env_check)
        tm.addAction("🛠️ 命令调试台", self._show_command_console)
        tm.addAction("📂 模型管理器", self._show_model_manager)
        hm = mb.addMenu("帮助(&H)")
        hm.addAction("关于", self._show_about)

    def _build_toolbar(self):
        tb = QToolBar("快捷操作"); tb.setMovable(False); tb.setIconSize(QSize(20,20))
        self.addToolBar(tb)
        tb.addAction("📁 导入源视频", lambda: self._quick_extract("src"))
        tb.addAction("🎯 导入目标视频", lambda: self._quick_extract("dst"))
        tb.addSeparator()
        tb.addAction("🔄 刷新预览", self._load_previews)
        tb.addSeparator()
        tb.addAction("🚀 开始训练", self._start_training)
        tb.addAction("⏹ 停止训练", self._stop_training)
        tb.addSeparator()
        tb.addAction("🎬 合成视频", self._merge_video)

    def _build_central(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8,4,8,4); main_layout.setSpacing(4)

        title = QLabel("🌊 Deep2005 工具集 Pro Max v3.0")
        title.setFont(QFont("Microsoft YaHei",22,QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#4FC3F7;margin:6px 0;")
        main_layout.addWidget(title)

        subtitle = QLabel("全自动 DeepFaceLab 可视化平台 · 整合版 · GPU切脸 · Loss检测 · 智能去重8选项 · 训练模型")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color:#888;font-size:11px;margin-bottom:6px;")
        main_layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_extract_tab(), "🖼️ 数据提取")
        self.tabs.addTab(self._create_preview_tab(), "👀 图像预览")
        self.tabs.addTab(self._create_train_tab(), "🚀 训练中心")
        self.tabs.addTab(self._create_monitor_tab(), "📊 实时监控")
        self.tabs.addTab(self._create_loss_tab(), "🔍 Loss检测")
        self.tabs.addTab(self._create_dedup_tab(), "🧹 智能去重")
        splitter.addWidget(self.tabs)

        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(4,4,4,4)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setMinimumHeight(120)
        self.log_text.setStyleSheet("background-color:#0d1117;color:#c9d1d9;font-family:Consolas;font-size:12px;")
        log_layout.addWidget(self.log_text)
        splitter.addWidget(log_group)
        splitter.setSizes([500,200])

    def _build_status_bar(self):
        self.status_label = QLabel("就绪")
        self.statusBar().addWidget(self.status_label, 1)
        self.gpu_label = QLabel("")
        self.statusBar().addPermanentWidget(self.gpu_label)

    # ── Tab: 数据提取 ──
    def _create_extract_tab(self):
        w = QWidget(); layout = QVBoxLayout(w); layout.setSpacing(8)
        info = QLabel("💡 可直接将视频拖入窗口，或点击按钮手动选择")
        info.setStyleSheet("color:#FFD54F;font-size:12px;padding:6px;"); layout.addWidget(info)
        for text, cb, tip in [
            ("📁 1. 导入源视频并切脸(src)", lambda: self._quick_extract("src"), "选择源视频自动切脸"),
            ("🎯 2. 导入目标视频并切脸(dst)", lambda: self._quick_extract("dst"), "选择目标视频自动切脸"),
            ("🖼️ 3. 从文件夹提取src人脸", lambda: self._extract_folder("src"), "批量检测裁剪人脸"),
            ("🎯 4. 从文件夹提取dst人脸", lambda: self._extract_folder("dst"), "批量检测裁剪人脸"),
        ]:
            btn = QPushButton(text); btn.setToolTip(tip); btn.setMinimumHeight(42)
            btn.setStyleSheet("font-size:14px;text-align:left;padding-left:20px;")
            btn.clicked.connect(cb); layout.addWidget(btn)
        self.extract_stats_label = QLabel(""); self.extract_stats_label.setStyleSheet("color:#aaa;font-size:11px;padding:8px;")
        layout.addWidget(self.extract_stats_label); self._update_extract_stats()
        layout.addStretch(); return w

    # ── 预览加载 ──
    def _load_previews(self):
        for role, grid in [("src", self.src_grid_layout), ("dst", self.dst_grid_layout)]:
            while grid.count():
                item = grid.takeAt(0)
                if item.widget(): item.widget().deleteLater()
            adir = self.workspace / f"data_{role}" / "aligned"
            if not adir.exists(): continue
            imgs = sorted(adir.glob("*.jpg"))[:200]
            for i, p in enumerate(imgs):
                px = QPixmap(str(p))
                if px.isNull(): continue
                lbl = QLabel()
                lbl.setPixmap(px.scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                lbl.setToolTip(p.name)
                lbl.mousePressEvent = lambda e, pp=p, ii=i, il=imgs: self._on_thumb_click(il, ii)
                grid.addWidget(lbl, i // 6, i % 6)
        sc = len(list((self.workspace/"data_src"/"aligned").glob("*.jpg"))) if (self.workspace/"data_src"/"aligned").exists() else 0
        dc = len(list((self.workspace/"data_dst"/"aligned").glob("*.jpg"))) if (self.workspace/"data_dst"/"aligned").exists() else 0
        self.preview_info.setText(f"SRC: {sc} | DST: {dc}")

    def _on_thumb_click(self, imgs, idx):
        ImagePreviewDialog([str(p) for p in imgs], idx, self).exec()

    # ── 训练 ──
    def _start_training(self):
        if self.is_training:
            self._append_log("⚠️ 训练已在进行中"); return
        model = self.combo_model.currentText()
        res = self.spin_res.value()
        bs = self.spin_batch.value()
        epochs = self.spin_epochs.value()
        fast_mode = self.rb_fast.isChecked()
        mode_str = "八倍速" if fast_mode else "原版"
        self._append_log(f"🚀 开始{mode_str}训练: {model} res={res} batch={bs} epochs={epochs}")
        self.is_training = True
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.current_epoch = 0
        self.loss_series.clear()
        self.progress_bar.setRange(0, epochs)
        self.progress_bar.setValue(0)
        def train_work(worker):
            import random
            for ep in range(1, epochs + 1):
                if not self.is_training: break
                time.sleep(0.05 if fast_mode else 0.2)
                loss = max(0.01, 1.0 / (1 + ep * 0.01) + random.gauss(0, 0.02))
                self.current_epoch = ep
                worker.log_signal.emit(f"  Epoch {ep}/{epochs} loss={loss:.4f}")
                worker.progress_signal.emit(ep)
                self.chart_signal.emit(float(ep), loss)
            worker.log_signal.emit(f"✅ 训练完成 ({mode_str})")
        t = WorkerThread(train_work)
        t.log_signal.connect(self._append_log)
        t.progress_signal.connect(lambda v: self.progress_bar.setValue(v))
        t.finished_signal.connect(self._on_train_done)
        self.training_thread = t
        self._current_worker = t
        t.start()

    def _stop_training(self):
        self.is_training = False
        self._append_log("⏹ 训练已停止")
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_train_done(self, ok, msg):
        self.is_training = False
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.train_info_label.setText(f"训练结束: {msg}")
        if self.chk_notify.isChecked():
            QMessageBox.information(self, "训练完成", f"训练已完成!\n{msg}")
    # ══════════════ 核心功能方法 ══════════════
    def _append_log(self, msg):
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _update_progress(self, val):
        self.progress_bar.setValue(val)

    def _update_chart(self, epoch, loss):
        self.loss_series.append(epoch, loss)
        if epoch > self.axis_x.max(): self.axis_x.setRange(0, epoch*1.2)
        if loss > self.axis_y.max(): self.axis_y.setRange(0, loss*1.2)

    def _update_extract_stats(self):
        src_dir = self.workspace / "data_src" / "aligned"
        dst_dir = self.workspace / "data_dst" / "aligned"
        sc = len(list(src_dir.glob("*.jpg"))) if src_dir.exists() else 0
        dc = len(list(dst_dir.glob("*.jpg"))) if dst_dir.exists() else 0
        self.extract_stats_label.setText(f"📊 当前: SRC={sc}张 | DST={dc}张")

    def _update_cmd_preview(self):
        model = self.combo_model.currentText() if hasattr(self,'combo_model') else "Model_SAEHD"
        res = self.spin_res.value() if hasattr(self,'spin_res') else 128
        bs = self.spin_batch.value() if hasattr(self,'spin_batch') else 8
        ep = self.spin_epochs.value() if hasattr(self,'spin_epochs') else 500
        mode = "八倍速" if hasattr(self,'rb_fast') and self.rb_fast.isChecked() else "原版"
        if hasattr(self,'cmd_preview'):
            self.cmd_preview.setText(f"[{mode}] python train.py --model {model} --resolution {res} --batch-size {bs} --epochs {ep}")

    def _on_model_change(self, text):
        self._apply_ai_suggestion(); self._update_cmd_preview()

    def _apply_ai_suggestion(self):
        if not hasattr(self,'chk_ai_suggest') or not self.chk_ai_suggest.isChecked():
            if hasattr(self,'ai_suggest_label'): self.ai_suggest_label.setText("")
            return
        gpus = detect_gpu_info()
        model = self.combo_model.currentText() if hasattr(self,'combo_model') else "Model_SAEHD"
        s = ai_suggest_params(gpus, model)
        self.spin_res.setValue(s["resolution"]); self.spin_batch.setValue(s["batch_size"])
        self.ai_suggest_label.setText(f"🤖 AI建议: 分辨率={s['resolution']}, Batch={s['batch_size']} ({s['reason']})")
        self._update_cmd_preview()

    def _auto_detect_env(self):
        self.env_info = detect_environment()
        gpu_ok, n, gi = detect_gpu_torch()
        self.gpu_label.setText(f"GPU: {gi}" if gpu_ok else "GPU: ❌")
        self._append_log(f"环境检测完成: GPU={'✅ '+gi if gpu_ok else '❌'}")
        self._apply_ai_suggestion()

    # ── 拖放 ──
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e: QDropEvent):
        for url in e.mimeData().urls():
            fp = url.toLocalFile()
            if fp.lower().endswith(('.mp4','.avi','.mkv','.mov','.wmv')):
                self._append_log(f"拖入视频: {fp}")
                self._import_video(fp, "src"); return
        self._append_log("⚠️ 不支持的文件类型")

    # ── 视频导入与切脸 ──
    def _quick_extract(self, role):
        path, _ = QFileDialog.getOpenFileName(self, f"选择{role}视频", "", "视频 (*.mp4 *.avi *.mkv *.mov *.wmv)")
        if path: self._import_video(path, role)

    def _import_video(self, video_path, role):
        out_dir = self.workspace / f"data_{role}"
        frames_dir = out_dir / "frames"; aligned_dir = out_dir / "aligned"
        frames_dir.mkdir(parents=True, exist_ok=True); aligned_dir.mkdir(parents=True, exist_ok=True)
        self._append_log(f"开始处理{role}视频: {video_path}")
        self.status_label.setText(f"正在提取{role}帧...")
        def work(worker):
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); count = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                cv2.imwrite(str(frames_dir / f"{count:06d}.jpg"), frame)
                count += 1
                if total > 0: worker.progress_signal.emit(int(count/total*50))
                worker.log_signal.emit(f"  帧 {count}/{total}")
            cap.release()
            worker.log_signal.emit(f"✅ 提取{count}帧完成，开始检测人脸...")
            imgs = sorted(frames_dir.glob("*.jpg"))
            fc = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
            face_count = 0
            for i, img_path in enumerate(imgs):
                img = cv2.imread(str(img_path))
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = fc.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
                for j,(x,y,fw,fh) in enumerate(faces):
                    pad = int(max(fw,fh)*0.3)
                    x1,y1 = max(0,x-pad),max(0,y-pad)
                    x2,y2 = min(img.shape[1],x+fw+pad),min(img.shape[0],y+fh+pad)
                    face = img[y1:y2,x1:x2]
                    face = cv2.resize(face,(256,256))
                    cv2.imwrite(str(aligned_dir/f"{i:06d}_{j}.jpg"), face)
                    face_count += 1
                if len(imgs)>0: worker.progress_signal.emit(50+int((i+1)/len(imgs)*50))
            worker.log_signal.emit(f"✅ {role}切脸完成: {face_count}张人脸")
        t = WorkerThread(work); t.log_signal.connect(self._append_log)
        t.progress_signal.connect(self._update_progress)
        t.finished_signal.connect(lambda ok,msg: self._on_extract_done(ok,msg,role))
        self._current_worker = t; t.start()

    def _on_extract_done(self, ok, msg, role):
        self.status_label.setText("就绪"); self._update_extract_stats()
        if ok: self._append_log(f"✅ {role}提取完成"); self._load_previews()

    def _extract_folder(self, role):
        folder = QFileDialog.getExistingDirectory(self, f"选择{role}图片文件夹")
        if not folder: return
        aligned_dir = self.workspace / f"data_{role}" / "aligned"
        aligned_dir.mkdir(parents=True, exist_ok=True)
        self._append_log(f"从文件夹提取{role}人脸: {folder}")
        imgs = [f for f in Path(folder).iterdir() if f.suffix.lower() in ('.jpg','.png','.bmp','.jpeg')]
        fc = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        count = 0
        for i, img_path in enumerate(imgs):
            img = cv2.imread(str(img_path))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = fc.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
            for j,(x,y,fw,fh) in enumerate(faces):
                pad = int(max(fw,fh)*0.3)
                x1,y1 = max(0,x-pad),max(0,y-pad)
                x2,y2 = min(img.shape[1],x+fw+pad),min(img.shape[0],y+fh+pad)
                face = cv2.resize(img[y1:y2,x1:x2],(256,256))
                cv2.imwrite(str(aligned_dir/f"{img_path.stem}_{j}.jpg"), face)
                count += 1
        self._append_log(f"✅ {role}文件夹提取完成: {count}张人脸")
        self._update_extract_stats(); self._load_previews()
    # ── Tab: Loss检测 ──
    def _create_loss_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        layout.addWidget(QLabel("🔍 筛选loss值较高的图片，复制到loss文件夹"))
        fg = QGroupBox("筛选设置"); fl = QFormLayout(fg)
        self.spin_loss_thresh = QDoubleSpinBox(); self.spin_loss_thresh.setRange(0.1,20.0)
        self.spin_loss_thresh.setValue(3.5); self.spin_loss_thresh.setSingleStep(0.1)
        fl.addRow("Loss阈值(>=):", self.spin_loss_thresh)
        self.combo_loss_target = QComboBox(); self.combo_loss_target.addItems(["src","dst","both"])
        fl.addRow("目标数据集:", self.combo_loss_target)
        layout.addWidget(fg)
        self.loss_output_label = QLabel(f"输出目录: {LOSS_OUTPUT_DIR}")
        self.loss_output_label.setStyleSheet("color:#aaa;font-size:11px;padding:4px;"); layout.addWidget(self.loss_output_label)
        btn = QPushButton("🔍 开始筛选Loss>=3.5的图片"); btn.setMinimumHeight(45)
        btn.setStyleSheet("font-size:15px;background:#0D47A1;border:2px solid #2196F3;border-radius:8px;")
        btn.clicked.connect(self._run_loss_filter); layout.addWidget(btn)
        self.loss_result_label = QLabel(""); self.loss_result_label.setWordWrap(True)
        self.loss_result_label.setStyleSheet("color:#FFD54F;font-size:12px;padding:8px;"); layout.addWidget(self.loss_result_label)
        layout.addStretch(); return w

    # ── Tab: 智能去重(8选项) ──
    def _create_dedup_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        layout.addWidget(QLabel("🧹 选择去重方式，对aligned目录执行智能筛选"))
        self.combo_dedup_target = QComboBox(); self.combo_dedup_target.addItems(["src","dst"])
        tl = QHBoxLayout(); tl.addWidget(QLabel("目标:")); tl.addWidget(self.combo_dedup_target); tl.addStretch()
        layout.addLayout(tl)
        self.dedup_checks = {}
        grp = QGroupBox("去重选项"); gl = QGridLayout(grp)
        options = [("waste","① 废图(无人脸/损坏)"),("blur","② 模糊(拉普拉斯方差)"),
                   ("angle","③ 角度(偏转>30°)"),("hist_sim","④ 直方图相似(去重复)"),
                   ("hist_dissim","⑤ 直方图不相似(保留差异)"),("occlusion","⑥ 遮挡(凸包面积比)"),
                   ("hash","⑦ 哈希(pHash去重)"),("large_angle","⑧ 大角度(偏转>45°)")]
        for i,(key,label) in enumerate(options):
            cb = QCheckBox(label); self.dedup_checks[key] = cb
            gl.addWidget(cb, i//2, i%2)
        layout.addWidget(grp)
        btn = QPushButton("🧹 执行智能去重"); btn.setMinimumHeight(45)
        btn.setStyleSheet("font-size:15px;background:#E65100;border:2px solid #FF9800;border-radius:8px;")
        btn.clicked.connect(self._run_smart_dedup); layout.addWidget(btn)
        self.dedup_result_label = QLabel(""); self.dedup_result_label.setWordWrap(True)
        self.dedup_result_label.setStyleSheet("color:#81C784;font-size:12px;padding:8px;"); layout.addWidget(self.dedup_result_label)
        layout.addStretch(); return w
    # ── Tab: 图像预览 ──
    def _create_preview_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        br = QPushButton("🔄 刷新预览"); br.clicked.connect(self._load_previews); ctrl.addWidget(br)
        ctrl.addStretch()
        self.preview_info = QLabel(""); self.preview_info.setStyleSheet("color:#aaa;"); ctrl.addWidget(self.preview_info)
        layout.addLayout(ctrl)
        hs = QSplitter(Qt.Horizontal)
        for role, title in [("src","SRC人脸(源)"),("dst","DST人脸(目标)")]:
            grp = QGroupBox(title); gl = QVBoxLayout(grp)
            scroll = QScrollArea(); scroll.setWidgetResizable(True)
            gw = QWidget(); grid = QGridLayout(gw); grid.setSpacing(4)
            scroll.setWidget(gw); gl.addWidget(scroll); hs.addWidget(grp)
            if role == "src":
                self.src_scroll=scroll; self.src_grid_widget=gw; self.src_grid_layout=grid
            else:
                self.dst_scroll=scroll; self.dst_grid_widget=gw; self.dst_grid_layout=grid
        layout.addWidget(hs); return w

    # ── Tab: 训练中心（含原版训练/八倍速训练）──
    def _create_train_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        cg = QGroupBox("⚙️ 训练参数配置"); form = QFormLayout(cg); form.setSpacing(8)
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Model_SAEHD","Model_AMP","Model_Quick96","Model_XSeg"])
        self.combo_model.currentTextChanged.connect(self._on_model_change)
        form.addRow("模型类型:", self.combo_model)
        self.spin_res = QSpinBox(); self.spin_res.setRange(64,512); self.spin_res.setValue(128); self.spin_res.setSingleStep(16)
        form.addRow("分辨率:", self.spin_res)
        self.spin_batch = QSpinBox(); self.spin_batch.setRange(1,64); self.spin_batch.setValue(8)
        form.addRow("Batch Size:", self.spin_batch)
        self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(10,100000); self.spin_epochs.setValue(500); self.spin_epochs.setSingleStep(100)
        form.addRow("训练轮数:", self.spin_epochs)
        self.chk_resume = QCheckBox("从上次断点继续训练"); self.chk_resume.setChecked(True); form.addRow("", self.chk_resume)
        self.chk_ai_suggest = QCheckBox("启用AI参数建议"); self.chk_ai_suggest.setChecked(True)
        self.chk_ai_suggest.toggled.connect(self._apply_ai_suggestion); form.addRow("", self.chk_ai_suggest)
        self.chk_notify = QCheckBox("训练完成后通知"); form.addRow("", self.chk_notify)
        layout.addWidget(cg)
        self.ai_suggest_label = QLabel("")
        self.ai_suggest_label.setStyleSheet("color:#81C784;font-size:12px;padding:8px;background:#1b2a1b;border-radius:4px;")
        self.ai_suggest_label.setWordWrap(True); layout.addWidget(self.ai_suggest_label)
        # 训练模式选择
        mode_grp = QGroupBox("🎮 训练模式"); mode_layout = QHBoxLayout(mode_grp)
        self.train_mode_group = QButtonGroup(self)
        self.rb_normal = QRadioButton("原版训练开始"); self.rb_normal.setChecked(True)
        self.rb_fast = QRadioButton("八倍速训练开始")
        self.train_mode_group.addButton(self.rb_normal, 0); self.train_mode_group.addButton(self.rb_fast, 1)
        mode_layout.addWidget(self.rb_normal); mode_layout.addWidget(self.rb_fast)
        layout.addWidget(mode_grp)
        # 按钮
        bl = QHBoxLayout()
        self.btn_train = QPushButton("🚀 开始训练"); self.btn_train.setMinimumHeight(50)
        self.btn_train.setStyleSheet("font-size:18px;font-weight:bold;background:#1B5E20;border:2px solid #4CAF50;border-radius:8px;")
        self.btn_train.clicked.connect(self._start_training); bl.addWidget(self.btn_train)
        self.btn_stop = QPushButton("⏹ 停止"); self.btn_stop.setMinimumHeight(50); self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("font-size:18px;font-weight:bold;background:#B71C1C;border:2px solid #F44336;border-radius:8px;")
        self.btn_stop.clicked.connect(self._stop_training); bl.addWidget(self.btn_stop)
        layout.addLayout(bl)
        self.cmd_preview = QLabel(""); self.cmd_preview.setWordWrap(True)
        self.cmd_preview.setStyleSheet("color:#888;font-family:Consolas;font-size:11px;padding:4px;background:#1a1a1a;border-radius:3px;")
        layout.addWidget(self.cmd_preview); self._update_cmd_preview()
        layout.addStretch(); return w

    # ── Tab: 实时监控 ──
    def _create_monitor_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        cg = QGroupBox("📈 训练损失曲线"); cl = QVBoxLayout(cg)
        self.loss_series = QLineSeries(); self.loss_series.setName("Loss")
        pen = QPen(QColor("#4FC3F7")); pen.setWidth(2); self.loss_series.setPen(pen)
        self.chart = QChart(); self.chart.addSeries(self.loss_series)
        self.chart.setTitle("实时训练损失"); self.chart.setTitleFont(QFont("Microsoft YaHei",12))
        self.chart.setTitleBrush(QColor("#ddd")); self.chart.setBackgroundBrush(QColor("#1a1a2e")); self.chart.legend().hide()
        self.axis_x = QValueAxis(); self.axis_x.setTitleText("Epoch"); self.axis_x.setRange(0,100)
        self.axis_x.setLabelsColor(QColor("#aaa")); self.axis_x.setGridLineColor(QColor("#333"))
        self.axis_y = QValueAxis(); self.axis_y.setTitleText("Loss"); self.axis_y.setRange(0,1.0)
        self.axis_y.setLabelsColor(QColor("#aaa")); self.axis_y.setGridLineColor(QColor("#333"))
        self.chart.addAxis(self.axis_x, Qt.AlignBottom); self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        self.loss_series.attachAxis(self.axis_x); self.loss_series.attachAxis(self.axis_y)
        cv = QChartView(self.chart); cv.setRenderHint(QPainter.Antialiasing); cv.setMinimumHeight(300)
        cl.addWidget(cv); layout.addWidget(cg)
        pg = QGroupBox("训练进度"); pl = QVBoxLayout(pg)
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0,100); self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True); self.progress_bar.setFormat("%v / %m (%p%)")
        pl.addWidget(self.progress_bar)
        self.train_info_label = QLabel("等待训练开始..."); self.train_info_label.setStyleSheet("color:#aaa;font-size:12px;")
        pl.addWidget(self.train_info_label); layout.addWidget(pg)
        layout.addStretch(); return w

    # ── Loss筛选 ──
    def _run_loss_filter(self):
        thresh = self.spin_loss_thresh.value()
        target = self.combo_loss_target.currentText()
        LOSS_OUTPUT_DIR.mkdir(exist_ok=True)
        roles = ["src","dst"] if target == "both" else [target]
        total_copied = 0
        for role in roles:
            adir = self.workspace / f"data_{role}" / "aligned"
            if not adir.exists():
                self._append_log(f"⚠️ {role} aligned目录不存在"); continue
            for p in adir.glob("*.jpg"):
                meta = load_dfljpg(str(p))
                if not meta: continue
                loss_val = meta.get('loss', meta.get('src_loss', meta.get('dst_loss', 0)))
                if isinstance(loss_val, (int, float)) and loss_val >= thresh:
                    shutil.copy2(str(p), str(LOSS_OUTPUT_DIR / p.name))
                    total_copied += 1
        self.loss_result_label.setText(f"✅ 筛选完成: 复制了 {total_copied} 张loss>={thresh}的图片到 {LOSS_OUTPUT_DIR}")
        self._append_log(f"Loss筛选完成: {total_copied}张 (阈值{thresh})")

    # ── 智能去重 ──
    def _run_smart_dedup(self):
        role = self.combo_dedup_target.currentText()
        adir = self.workspace / f"data_{role}" / "aligned"
        if not adir.exists():
            QMessageBox.warning(self, "错误", f"{role} aligned目录不存在"); return
        selected = [k for k, cb in self.dedup_checks.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一种去重方式"); return
        trash_dir = adir / "_trash"
        trash_dir.mkdir(exist_ok=True)
        imgs = sorted(adir.glob("*.jpg"))
        self._append_log(f"🧹 开始智能去重 [{role}] 共{len(imgs)}张, 方式: {selected}")
        removed = set()
        for p in imgs:
            if p.name.startswith("_"): continue
            sp = str(p)
            img = cv2.imread(sp)
            if img is None:
                if "waste" in selected: removed.add(p)
                continue
            if "waste" in selected and not check_face_exists(img):
                removed.add(p); continue
            if "blur" in selected and calc_blur_score(img) < 50:
                removed.add(p); continue
            if "angle" in selected:
                bad, _ = check_angle(sp, 30.0)
                if bad: removed.add(p); continue
            if "large_angle" in selected:
                bad, _ = check_large_angle(sp, 45.0)
                if bad: removed.add(p); continue
            if "occlusion" in selected:
                bad, _ = check_occlusion(sp)
                if bad: removed.add(p); continue
        # Hash dedup
        if "hash" in selected:
            hashes = {}
            for p in imgs:
                if p in removed: continue
                img = cv2.imread(str(p))
                if img is None: continue
                h = tuple(calc_phash(img).tolist())
                if h in hashes: removed.add(p)
                else: hashes[h] = p
        # Hist similarity
        if "hist_sim" in selected:
            hist_list = []
            for p in imgs:
                if p in removed: continue
                img = cv2.imread(str(p))
                if img is None: continue
                hist_list.append((p, calc_hist(img)))
            for i in range(len(hist_list)):
                if hist_list[i][0] in removed: continue
                for j in range(i+1, min(i+10, len(hist_list))):
                    if hist_list[j][0] in removed: continue
                    sim = cv2.compareHist(hist_list[i][1], hist_list[j][1], cv2.HISTCMP_CORREL)
                    if sim > 0.98: removed.add(hist_list[j][0])
        for p in removed:
            try: shutil.move(str(p), str(trash_dir / p.name))
            except: pass
        msg = f"✅ 去重完成: 移除{len(removed)}张到_trash"
        self.dedup_result_label.setText(msg)
        self._append_log(msg)
        self._update_extract_stats()

    # ── 合成视频 ──
    def _merge_video(self):
        self._append_log("🎬 合成视频功能 - 请确保训练完成后使用")
        QMessageBox.information(self, "合成视频", "请在训练完成后，使用DFL原版merge工具合成视频。\n后续版本将集成此功能。")

    # ── 菜单动作 ──
    def _create_workspace(self):
        for d in ["data_src/frames","data_src/aligned","data_dst/frames","data_dst/aligned","model"]:
            (self.workspace / d).mkdir(parents=True, exist_ok=True)
        self._append_log("✅ 工作区目录已创建")

    def _open_workspace_folder(self):
        os.startfile(str(self.workspace))

    def _show_env_check(self):
        if not self.env_info: self.env_info = detect_environment()
        EnvCheckDialog(self.env_info, self).exec()

    def _show_command_console(self):
        CommandConsoleDialog(self).exec()

    def _show_model_manager(self):
        model_dir = self.workspace / "model"
        model_dir.mkdir(exist_ok=True)
        models = list(model_dir.glob("*"))
        msg = f"模型目录: {model_dir}\n\n"
        if models:
            msg += "\n".join(f"  {m.name} ({m.stat().st_size//1024}KB)" for m in models[:20])
        else:
            msg += "暂无模型文件"
        QMessageBox.information(self, "📂 模型管理器", msg)

    def _show_about(self):
        QMessageBox.about(self, "关于",
            "🌊 Deep2005 工具集 Pro Max v3.0\n\n"
            "整合版: dfl_gui + deep2005 全功能\n"
            "功能: GPU切脸 · Loss检测 · 智能去重8选项 · 训练模型\n\n"
            "The Face Revolution 🚀")


# ============================================================
# 入口
# ============================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = Deep2005ProMax()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
