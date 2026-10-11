import sys
import os
import csv
import time
import json
import base64
import traceback
import subprocess
import wave
import tempfile
import shutil
import cv2
import numpy as np
from datetime import timedelta
from typing import Optional, Tuple, Any, Dict, List
from PySide2.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QProgressBar, QSlider, QComboBox, 
                               QFileDialog, QRadioButton, QButtonGroup, QMessageBox,
                               QSpinBox, QCheckBox, QGroupBox, QGridLayout, QTabWidget,
                               QFrame, QScrollArea, QSizePolicy, QLineEdit, QRubberBand,
                               QColorDialog, QDialog, QDialogButtonBox, QShortcut)
from PySide2.QtCore import QThread, Signal, Qt, QTimer, QRect, QPoint
from PySide2.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QPainter, QPen, QKeySequence

# Try to import DaVinci Resolve Scripting API
try:
    import DaVinciResolveScript as dvr_script
    resolve = dvr_script.scriptapp("Resolve")
    # Verify we actually got a valid resolve object
    if not resolve or not resolve.GetProjectManager():
        resolve = None
except ImportError:
    resolve = None

# ============================================================
# CONSTANTS
# ============================================================
MAX_PREVIEW_WIDTH = 1000
MAX_PREVIEW_HEIGHT = 650
THUMB_W, THUMB_H = 216, 126
SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".cv_matcher_settings.json")
PROFILES_DIR = os.path.join(os.path.expanduser("~"), ".cv_matcher_profiles")

# ============================================================
# DARK THEME STYLESHEET
# ============================================================
DARK_STYLE = """
QWidget {
    background-color: #2b2b2b;
    color: #d0d0d0;
    font-size: 11px;
}
QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QPushButton {
    background-color: #3c3c3c;
    border: 1px solid #4a4a4a;
    border-radius: 3px;
    padding: 5px 12px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #4a4a4a;
    border-color: #5a5a5a;
}
QPushButton:pressed {
    background-color: #2a2a2a;
}
QPushButton:disabled {
    background-color: #333333;
    color: #666666;
}
QPushButton#runBtn {
    background-color: #4a7a4a;
    font-weight: bold;
    font-size: 12px;
    padding: 10px;
}
QPushButton#runBtn:hover {
    background-color: #5a8a5a;
}
QPushButton#runBtn:disabled {
    background-color: #3a5a3a;
    color: #777777;
}
QPushButton#dangerBtn {
    background-color: #7a3a3a;
}
QPushButton#dangerBtn:hover {
    background-color: #8a4a4a;
}
QSlider::groove:horizontal {
    border: 1px solid #3c3c3c;
    height: 4px;
    background: #1e1e1e;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #5a9a5a;
    border: 1px solid #4a8a4a;
    width: 12px;
    margin: -5px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover {
    background: #6aaa6a;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #3c3c3c;
    border: 1px solid #4a4a4a;
    border-radius: 3px;
    padding: 3px 8px;
    min-height: 20px;
}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover {
    border-color: #5a5a5a;
}
QComboBox QAbstractItemView {
    background-color: #3c3c3c;
    border: 1px solid #4a4a4a;
    selection-background-color: #4a7a4a;
}
QProgressBar {
    border: 1px solid #3c3c3c;
    border-radius: 3px;
    text-align: center;
    background: #1e1e1e;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #5a9a5a;
    border-radius: 2px;
}
QLabel#titleLabel {
    font-size: 14px;
    font-weight: bold;
    color: #5a9a5a;
    padding: 4px;
}
QLabel#sectionLabel {
    font-weight: bold;
    color: #7a9a7a;
}
QLabel#previewLabel {
    border: 2px dashed #4a4a4a;
    border-radius: 4px;
    background-color: #1e1e1e;
}
QTabWidget::pane {
    border: 1px solid #3c3c3c;
    border-radius: 3px;
}
QTabBar::tab {
    background-color: #2b2b2b;
    border: 1px solid #3c3c3c;
    padding: 5px 12px;
    margin-right: 2px;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
}
QTabBar::tab:selected {
    background-color: #3c5c3c;
    border-bottom: none;
}
QTabBar::tab:hover:!selected {
    background-color: #333333;
}
QCheckBox {
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 2px;
    border: 1px solid #5a5a5a;
    background: #1e1e1e;
}
QCheckBox::indicator:checked {
    background: #5a9a5a;
    border-color: #6aaa6a;
}
QScrollArea {
    border: none;
}
QToolTip {
    background-color: #1e1e1e;
    color: #d0d0d0;
    border: 1px solid #5a9a5a;
    padding: 4px;
}
"""

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def cv_img_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format_RGB888).copy()
    return qimg

def cv_img_to_pixmap(frame: np.ndarray, max_w: Optional[int] = None, max_h: Optional[int] = None) -> QPixmap:
    qimg = cv_img_to_qimage(frame)
    pix = QPixmap.fromImage(qimg)
    if max_w and max_h:
        pix = pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix

def get_marker_dict_value(markers: Dict[Any, Any], frame: int, key: Optional[str] = None) -> Any:
    for k in (str(frame), frame):
        if k in markers:
            if key is None:
                return markers[k]
            return markers[k].get(key)
    return None if key is not None else {}

# ============================================================
# VISUAL ROI PICKER DIALOG
# ============================================================
class CanvasLabel(QLabel):
    roi_selected = Signal(tuple)

    def __init__(self, frame: np.ndarray):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.orig_pixmap = cv_img_to_pixmap(frame)
        oh, ow = frame.shape[:2]
        scale = min(MAX_PREVIEW_WIDTH / ow, MAX_PREVIEW_HEIGHT / oh, 1.0)
        self.scale_factor = scale
        self.scaled_w = int(ow * scale)
        self.scaled_h = int(oh * scale)
        self.scaled_pixmap = self.orig_pixmap.scaled(self.scaled_w, self.scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self.scaled_pixmap)
        self.setFixedSize(self.scaled_w, self.scaled_h)
        self.start_pt = QPoint()
        self.end_pt = QPoint()
        self.is_drawing = False

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.start_pt.isNull() and not self.end_pt.isNull():
            painter = QPainter(self)
            rect = QRect(self.start_pt, self.end_pt).normalized()
            painter.setBrush(QColor(0, 0, 0, 150))
            painter.setPen(Qt.NoPen)
            painter.drawRect(0, 0, self.width(), rect.top())
            painter.drawRect(0, rect.bottom(), self.width(), self.height() - rect.bottom())
            painter.drawRect(0, rect.top(), rect.left(), rect.height())
            painter.drawRect(rect.right(), rect.top(), self.width() - rect.right(), rect.height())
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pt = event.pos()
            self.end_pt = event.pos()
            self.is_drawing = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            # Clamp coordinates to label bounds
            x = max(0, min(event.pos().x(), self.width()))
            y = max(0, min(event.pos().y(), self.height()))
            self.end_pt = QPoint(x, y)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.is_drawing:
            x = max(0, min(event.pos().x(), self.width()))
            y = max(0, min(event.pos().y(), self.height()))
            self.end_pt = QPoint(x, y)
            self.is_drawing = False
            self.update()
            rect = QRect(self.start_pt, self.end_pt).normalized()
            if rect.width() > 5 and rect.height() > 5:
                x = int(rect.x() / self.scale_factor)
                y = int(rect.y() / self.scale_factor)
                w = int(rect.width() / self.scale_factor)
                h = int(rect.height() / self.scale_factor)
                self.roi_selected.emit((x, y, w, h))
            else:
                self.roi_selected.emit(None)

class VisualPickerDialog(QDialog):
    def __init__(self, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Section / Marker ROI")
        self.setModal(True)
        self.setStyleSheet(DARK_STYLE)
        layout = QVBoxLayout()
        self.canvas = CanvasLabel(frame)
        self.canvas.roi_selected.connect(self.update_roi_info)
        layout.addWidget(self.canvas, 1, Qt.AlignCenter)
        self.info_label = QLabel("Click and drag on the image to select a region. Release to set.")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #aaa; padding: 5px;")
        layout.addWidget(self.info_label)
        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Selection")
        self.clear_btn.clicked.connect(self.clear_roi)
        btn_layout.addWidget(self.clear_btn)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        btn_layout.addWidget(btn_box)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.roi_rect = None

    def update_roi_info(self, roi: Optional[Tuple[int, int, int, int]]):
        self.roi_rect = roi
        if roi:
            x, y, w, h = roi
            self.info_label.setText(f"Selected Region: X={x}, Y={y}, W={w}, H={h}")
        else:
            self.info_label.setText("No region selected. (Whole image will be used)")

    def clear_roi(self):
        self.canvas.start_pt = QPoint()
        self.canvas.end_pt = QPoint()
        self.canvas.update()
        self.update_roi_info(None)


# ============================================================
# WORKER THREAD
# ============================================================
class CVWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    status = Signal(str)
    marker_found = Signal(dict)
    match_verified = Signal(object)
    
    # Thread-safe signals for Resolve API
    marker_to_add = Signal(dict)
    playhead_to_set = Signal(int)
    auto_fit_request = Signal(list)

    def __init__(self, video_path: str, timeline_start_frame: int, source_start_frame: int, 
                 threshold: float, marker_color: str, mode: str, marker_duration: int = 1,
                 template_img: Optional[np.ndarray] = None, frame_skip: int = 1, 
                 orb_distance: int = 50, scene_method: str = "Histogram", 
                 min_segment_length: int = 5, timeline_end_frame: Optional[int] = None, 
                 source_end_frame: Optional[int] = None, marker_prefix: str = "CV", 
                 roi_rect: Optional[Tuple[int, int, int, int]] = None, 
                 target_color_hsv: Optional[Tuple[int, int, int]] = None,
                 auto_fit_duration: bool = False, sync_playhead: bool = False,
                 use_clahe: bool = False, max_analysis_res: int = 1080,
                 audio_filter_enabled: bool = False, audio_threshold_db: float = -10.0,
                 audio_sustained: bool = False):
        super().__init__()
        self.video_path = video_path
        self.timeline_start_frame = timeline_start_frame
        self.source_start_frame = source_start_frame
        self.threshold = threshold
        self.marker_color = marker_color
        self.mode = mode
        self.marker_duration = marker_duration
        self.template_img = template_img
        self.frame_skip = max(1, frame_skip)
        self.orb_distance = orb_distance
        self.scene_method = scene_method
        self.min_segment_length = min_segment_length
        self.timeline_end_frame = timeline_end_frame
        self.source_end_frame = source_end_frame
        self.marker_prefix = marker_prefix
        self.roi_rect = roi_rect 
        self.target_color_hsv = target_color_hsv 
        self.auto_fit_duration = auto_fit_duration
        self.sync_playhead = sync_playhead
        self.use_clahe = use_clahe
        self.max_analysis_res = max_analysis_res
        
        # Audio filter state
        self.audio_filter_enabled = audio_filter_enabled
        self.audio_threshold_db = audio_threshold_db
        self.audio_sustained = audio_sustained
        self.audio_db_per_frame = {} # Map: source_frame -> dB
        
        self._cancelled = False
        self.added_marker_frames: List[int] = [] 
        self._color_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.event_marker_added = False  
        self._static_count = 0  
        self._ui_match_count = 0 

    def cancel(self):
        self._cancelled = True

    def _downscale_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None: return frame
        oh, ow = frame.shape[:2]
        if max(oh, ow) > self.max_analysis_res:
            scale = self.max_analysis_res / max(oh, ow)
            return cv2.resize(frame, (int(ow * scale), int(oh * scale)))
        return frame

    def _extract_audio_data(self) -> bool:
        if not self.audio_filter_enabled:
            return True
            
        self.status.emit("Extracting audio for spike analysis...")
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            self.finished.emit("Error: ffmpeg not found in system PATH. Audio filter disabled.")
            self.audio_filter_enabled = False
            return False
            
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()
        
        # Prevent console window from popping up on Windows
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        try:
            subprocess.run(
                [ffmpeg_path, "-y", "-i", self.video_path, "-vn", "-ac", "1", "-ar", "44100", "-f", "wav", temp_wav.name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
                startupinfo=startupinfo
            )
            
            with wave.open(temp_wav.name, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)
                
            dtypes = {1: np.int8, 2: np.int16, 4: np.int32}
            audio = np.frombuffer(audio_bytes, dtype=dtypes.get(sampwidth, np.int16)).astype(np.float32)
            audio /= np.max(np.abs(audio) + 1e-6) 
            
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            cap.release()
            
            samples_per_frame = int(framerate / fps)
            if samples_per_frame == 0: samples_per_frame = 1
            
            for i in range(0, len(audio), samples_per_frame):
                chunk = audio[i:i+samples_per_frame]
                if len(chunk) > 0:
                    rms = np.sqrt(np.mean(chunk**2))
                    db = 20 * np.log10(rms + 1e-6)
                    frame_idx = i // samples_per_frame
                    self.audio_db_per_frame[frame_idx] = db
                    
            self.status.emit("Audio extraction complete.")
            return True
            
        except Exception as e:
            self.finished.emit(f"Audio analysis failed: {e}")
            self.audio_filter_enabled = False
            return False
        finally:
            if os.path.exists(temp_wav.name):
                os.remove(temp_wav.name)

    def _passes_audio_filter(self, frame: int) -> bool:
        if not self.audio_filter_enabled:
            return True
            
        source_f = frame - self.timeline_start_frame
        
        if self.audio_sustained:
            for i in range(max(0, source_f - self.min_segment_length + 1), source_f + 1):
                db = self.audio_db_per_frame.get(i, -60.0)
                if db < self.audio_threshold_db:
                    return False
            return True
        else:
            db = self.audio_db_per_frame.get(source_f, -60.0)
            return db >= self.audio_threshold_db

    def _add_marker(self, frame: int, name: str, note: str, marker_type: str):
        marker_data = {
            'frame': frame,
            'color': self.marker_color,
            'name': name,
            'note': note,
            'duration': self.marker_duration
        }
        self.marker_to_add.emit(marker_data)
        self.added_marker_frames.append(frame)
        self.marker_found.emit({"frame": frame, "name": name, "note": note, "type": marker_type})
        if self.sync_playhead:
            self.playhead_to_set.emit(frame)

    def _get_next_marker_id(self) -> int:
        return len(self.added_marker_frames) + 1

    def _compute_color_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.target_color_hsv is None:
            return None
        h, s, v = self.target_color_hsv
        lower = np.array([max(0, h-15), max(50, s-50), 50], dtype=np.uint8)
        upper = np.array([min(179, h+15), min(255, s+50), 255], dtype=np.uint8)
        return lower, upper

    def run(self):
        if not resolve:
            self.finished.emit("Error: DaVinci Resolve API not found.")
            return

        try:
            pm = resolve.GetProjectManager()
            if not pm:
                self.finished.emit("Error: Could not get Project Manager.")
                return
            project = pm.GetCurrentProject()
            if not project:
                self.finished.emit("Error: No project currently open in Resolve.")
                return
            timeline = project.GetCurrentTimeline()
            if not timeline:
                self.finished.emit("Error: No timeline currently active in Resolve.")
                return
        except Exception as e:
            self.finished.emit(f"Error: Could not access Resolve project/timeline.\n{e}")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit(f"Error: Could not open video file:\n{self.video_path}")
            return

        if self.audio_filter_enabled:
            if not self._extract_audio_data():
                cap.release()
                return

        try:
            self._run_analysis(cap)
        except Exception as e:
            tb = traceback.format_exc()
            self.finished.emit(f"Error during analysis:\n{e}\n\n{tb}")
        finally:
            cap.release()
            if self.auto_fit_duration and len(self.added_marker_frames) >= 2:
                self.auto_fit_request.emit(sorted(self.added_marker_frames))

    def _run_analysis(self, cap):
        source_start = self.source_start_frame
        if self.source_end_frame is not None:
            total_to_analyze = self.source_end_frame - source_start
        else:
            total_to_analyze = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - source_start

        cap.set(cv2.CAP_PROP_POS_FRAMES, source_start)
        ret, first_frame = cap.read()
        if not ret:
            self.finished.emit("Error: Could not read first frame of video.")
            return
        
        prev_frame = self._downscale_frame(first_frame)
        frame_count = 0
        start_time = time.time()
        last_marker_timeline_frame = self.timeline_start_frame
        
        self._static_count = 0
        self._ui_match_count = 0

        # -------------------------------------------------------
        # MODE SETUP
        # -------------------------------------------------------
        orb = None
        akaze = None
        matcher = None
        kp1 = des1 = None
        template_gray = None
        face_cascade = None
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if self.use_clahe else None

        if self.mode in ("Recurring Match", "Segment Match"):
            if self.template_img is None or not isinstance(self.template_img, np.ndarray) or self.template_img.size == 0:
                self.finished.emit("Error: No valid template frame captured.")
                return

            template_proc = self._downscale_frame(self.template_img)

            if self.mode == "Segment Match":
                akaze = cv2.AKAZE_create()
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            else:
                orb = cv2.ORB_create(nfeatures=1000)
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            if self.mode == "Segment Match" and self.roi_rect:
                x, y, w, h = self.roi_rect
                ih, iw = template_proc.shape[:2]
                x, y = max(0, x), max(0, y)
                w, h = min(iw-x, w), min(ih-y, h)
                if w <= 0 or h <= 0:
                    self.finished.emit("Error: Invalid ROI region selected.")
                    return
                template_patch = template_proc[y:y+h, x:x+w]
                template_gray = cv2.cvtColor(template_patch, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = cv2.cvtColor(template_proc, cv2.COLOR_BGR2GRAY)

            if self.use_clahe:
                template_gray = clahe.apply(template_gray)

            if self.mode == "Segment Match":
                kp1, des1 = akaze.detectAndCompute(template_gray, None)
            else:
                kp1, des1 = orb.detectAndCompute(template_gray, None)

            if des1 is None or len(des1) < 5:
                self.finished.emit("Error: Not enough visual features in template/ROI. Try a more detailed area.")
                return

        elif self.mode in ("UI Template Match", "Prolonged UI Match"):
            if self.template_img is None or not isinstance(self.template_img, np.ndarray) or self.template_img.size == 0:
                self.finished.emit("Error: No valid template frame captured.")
                return
            template_proc = self._downscale_frame(self.template_img)
            template_gray = cv2.cvtColor(template_proc, cv2.COLOR_BGR2GRAY)

        elif self.mode == "Face Detection":
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                self.finished.emit(f"Error: Could not load Haar cascade from:\n{cascade_path}")
                return

        elif self.mode in ("Color Match", "Health Bar Tracker"):
            self._color_bounds = self._compute_color_bounds()
            if self._color_bounds is None:
                self.finished.emit("Error: No target color specified for Color Match.")
                return

        # -------------------------------------------------------
        # MAIN ANALYSIS LOOP
        # -------------------------------------------------------
        while True:
            if self._cancelled:
                self.finished.emit(f"Cancelled. Added {len(self.added_marker_frames)} markers.")
                return

            if self.frame_skip > 1 and frame_count > 0:
                for _ in range(self.frame_skip - 1):
                    if not cap.grab(): break
                    frame_count += 1

            ret, curr_frame = cap.read()
            if not ret: break

            if self.source_end_frame is not None and (source_start + frame_count) >= self.source_end_frame:
                break

            timeline_frame = self.timeline_start_frame + frame_count
            proc_frame = self._downscale_frame(curr_frame)

            if self.mode == "Scene Match":
                is_scene_change, similarity_str = self._detect_scene_change(prev_frame, proc_frame)
                if is_scene_change:
                    segment_length = timeline_frame - last_marker_timeline_frame
                    if segment_length >= self.min_segment_length:
                        marker_name = f"{self.marker_prefix} Scene {self._get_next_marker_id()}"
                        marker_note = f"Sim: {similarity_str} | Len: {segment_length}f"
                        self._add_marker(timeline_frame, marker_name, marker_note, "Scene")
                        last_marker_timeline_frame = timeline_frame
                        self.status.emit(f"Scene cut at frame {timeline_frame} ({similarity_str})")

            elif self.mode == "Segment Match":
                was_matching = self._detect_segment(proc_frame, akaze, matcher, kp1, des1, template_gray, clahe, timeline_frame, False)

            elif self.mode == "Recurring Match":
                was_matching = self._detect_recurring(proc_frame, orb, matcher, kp1, des1, clahe, timeline_frame, False)

            elif self.mode == "Motion Detection":
                was_matching = self._detect_motion(prev_frame, proc_frame, timeline_frame, False)

            elif self.mode == "HUD Motion":
                was_matching = self._detect_hud_motion(prev_frame, proc_frame, timeline_frame, False)

            elif self.mode == "UI Template Match":
                was_matching = self._detect_ui_template(proc_frame, template_gray, timeline_frame, False)
                
            elif self.mode == "Prolonged UI Match":
                was_matching = self._detect_prolonged_ui(proc_frame, template_gray, timeline_frame, False)

            elif self.mode == "Health Bar Tracker":
                was_matching = self._detect_health_bar(proc_frame, timeline_frame, False)

            elif self.mode == "Loading Screen":
                self._static_count = self._detect_loading_screen(prev_frame, proc_frame, self._static_count, timeline_frame)
                
            elif self.mode == "Fade Detection":
                was_matching = self._detect_fade(proc_frame, timeline_frame, False)

            elif self.mode == "Sudden Freeze":
                was_matching = self._detect_freeze(prev_frame, proc_frame, timeline_frame, False)

            elif self.mode == "Camera Flick":
                was_matching = self._detect_camera_flick(prev_frame, proc_frame, timeline_frame, False)

            elif self.mode == "Face Detection":
                was_matching = self._detect_face(proc_frame, face_cascade, clahe, timeline_frame, False)

            elif self.mode == "Focus Detection":
                was_matching = self._detect_focus(proc_frame, clahe, timeline_frame, False)

            elif self.mode == "Color Match":
                was_matching = self._detect_color(proc_frame, timeline_frame, False)

            elif self.mode == "Flash Detection":
                was_matching = self._detect_flash(prev_frame, proc_frame, timeline_frame, False)

            prev_frame = proc_frame
            frame_count += self.frame_skip

            if frame_count % 3 == 0:
                self._emit_progress(frame_count, total_to_analyze, start_time)

        self.progress.emit(100)
        elapsed_total = str(timedelta(seconds=int(time.time() - start_time)))
        self.finished.emit(f"Done! Added {len(self.added_marker_frames)} markers in {elapsed_total}.")

    # --- DETECTION METHODS ---

    def _detect_scene_change(self, prev_frame: np.ndarray, proc_frame: np.ndarray) -> Tuple[bool, str]:
        if self.scene_method == "Histogram":
            prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
            curr_hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
            prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(prev_hist, prev_hist, 0,1, cv2.NORM_MINMAX)
            cv2.normalize(curr_hist, curr_hist, 0,1, cv2.NORM_MINMAX)
            similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            return similarity < self.threshold, f"{similarity:.3f}"

        elif self.scene_method == "Edges":
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            prev_edges = cv2.Canny(prev_gray, 50, 150)
            curr_edges = cv2.Canny(curr_gray, 50, 150)
            diff = cv2.absdiff(prev_edges, curr_edges)
            change_ratio = cv2.countNonZero(cv2.compare(diff, 30, cv2.CMP_GT)) / diff.size
            return change_ratio > (1 - self.threshold), f"{1-change_ratio:.3f}"

        elif self.scene_method == "Combined":
            prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
            curr_hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
            prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(prev_hist, prev_hist, 0,1, cv2.NORM_MINMAX)
            cv2.normalize(curr_hist, curr_hist, 0,1, cv2.NORM_MINMAX)
            hist_sim = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            prev_mean = np.mean(prev_hsv[:,:,2])
            curr_mean = np.mean(curr_hsv[:,:,2])
            brightness_diff = abs(prev_mean - curr_mean) / 255.0
            is_change = (hist_sim < self.threshold) or (brightness_diff > 0.3)
            return is_change, f"H:{hist_sim:.2f} B:{brightness_diff:.2f}"
        return False, ""

    def _detect_segment(self, proc_frame: np.ndarray, akaze, matcher, kp1, des1, template_gray,
                        clahe, timeline_frame: int, was_matching: bool) -> bool:
        curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        if self.use_clahe and clahe is not None:
            curr_gray = clahe.apply(curr_gray)
        kp2, des2 = akaze.detectAndCompute(curr_gray, None)
        good_matches = []
        if des2 is not None and len(des2) > 5:
            matches = matcher.knnMatch(des1, des2, k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance: 
                    good_matches.append(m)
        if len(good_matches) <= 8:
            self.event_marker_added = False
            return False
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if matrix is None:
            self.event_marker_added = False
            return False
        inliers = int(np.sum(mask))
        if inliers < self.threshold:
            self.event_marker_added = False
            return False
        if not was_matching or (self.audio_filter_enabled and not self.event_marker_added):
            if self._passes_audio_filter(timeline_frame):
                h, w = template_gray.shape
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, matrix)
                img_match = proc_frame.copy()
                img_match = cv2.polylines(img_match, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)
                marker_name = f"{self.marker_prefix} Segment {self._get_next_marker_id()}"
                marker_note = f"Inliers: {inliers}"
                self._add_marker(timeline_frame, marker_name, marker_note, "Segment")
                self.status.emit(f"Segment verified at frame {timeline_frame} ({inliers} inliers)")
                self.match_verified.emit(img_match)
                self.event_marker_added = True
            elif not was_matching:
                was_matching = True
        return True

    def _detect_recurring(self, proc_frame: np.ndarray, orb, matcher, kp1, des1, clahe,
                          timeline_frame: int, was_matching: bool) -> bool:
        curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        if self.use_clahe and clahe is not None:
            curr_gray = clahe.apply(curr_gray)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        good_matches = []
        if des2 is not None and len(des2) > 5:
            matches = matcher.match(des1, des2)
            good_matches = [m for m in matches if m.distance < self.orb_distance]
        if len(good_matches) <= self.threshold:
            self.event_marker_added = False
            return False
        if not was_matching or (self.audio_filter_enabled and not self.event_marker_added):
            if self._passes_audio_filter(timeline_frame):
                marker_name = f"{self.marker_prefix} Recur {self._get_next_marker_id()}"
                marker_note = f"Matches: {len(good_matches)}"
                self._add_marker(timeline_frame, marker_name, marker_note, "Recurring")
                self.status.emit(f"Recurring match at frame {timeline_frame} ({len(good_matches)} matches)")
                self.event_marker_added = True
            elif not was_matching:
                was_matching = True
        return True

    def _detect_motion(self, prev_frame: np.ndarray, proc_frame: np.ndarray, timeline_frame: int,
                       was_matching: bool) -> bool:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        prev_blur = cv2.GaussianBlur(prev_gray, (21, 21), 0)
        curr_blur = cv2.GaussianBlur(curr_gray, (21, 21), 0)
        diff = cv2.absdiff(prev_blur, curr_blur)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(area for c in contours if (area := cv2.contourArea(c)) > 500)
        motion_ratio = motion_area / (curr_gray.shape[0] * curr_gray.shape[1])
        if motion_ratio <= self.threshold:
            self.event_marker_added = False
            return False
        if not was_matching or (self.audio_filter_enabled and not self.event_marker_added):
            if self._passes_audio_filter(timeline_frame):
                marker_name = f"{self.marker_prefix} Motion {self._get_next_marker_id()}"
                marker_note = f"Motion: {motion_ratio:.3f} ({motion_ratio*100:.1f}%)"
                self._add_marker(timeline_frame, marker_name, marker_note, "Motion")
                self.status.emit(f"Motion detected at frame {timeline_frame} ({motion_ratio:.3f})")
                self.event_marker_added = True
            elif not was_matching:
                was_matching = True
        return True

    def _detect_hud_motion(self, prev_frame: np.ndarray, proc_frame: np.ndarray, timeline_frame: int,
                           was_matching: bool) -> bool:
        if not self.roi_rect:
            return False
        x, y, w, h = self.roi_rect
        ih, iw = proc_frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(iw-x, w), min(ih-y, h)
        if w <= 0 or h <= 0: return False
        
        prev_crop = prev_frame[y:y+h, x:x+w]
        curr_crop = proc_frame[y:y+h, x:x+w]
        
        prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)
        prev_blur = cv2.GaussianBlur(prev_gray, (21, 21), 0)
        curr_blur = cv2.GaussianBlur(curr_gray, (21, 21), 0)
        diff = cv2.absdiff(prev_blur, curr_blur)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        motion_area = cv2.countNonZero(thresh)
        motion_ratio = motion_area / (w * h)
        
        if motion_ratio <= self.threshold:
            self.event_marker_added = False
            return False
        if not was_matching or (self.audio_filter_enabled and not self.event_marker_added):
            if self._passes_audio_filter(timeline_frame):
                marker_name = f"{self.marker_prefix} HUD {self._get_next_marker_id()}"
                marker_note = f"HUD Motion: {motion_ratio*100:.1f}%"
                self._add_marker(timeline_frame, marker_name, marker_note, "HUD")
                self.status.emit(f"HUD motion at frame {timeline_frame} ({motion_ratio:.3f})")
                self.event_marker_added = True
            elif not was_matching:
                was_matching = True
        return True

    def _detect_ui_template(self, proc_frame: np.ndarray, template_gray: np.ndarray,
                            timeline_frame: int, was_matching: bool) -> bool:
        curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(curr_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val < self.threshold:
            self.event_marker_added = False
            return False
        if not was_matching or (self.audio_filter_enabled and not self.event_marker_added):
            if self._passes_audio_filter(timeline_frame):
                h, w = template_gray.shape
                img_match = proc_frame.copy()
                cv2.rectangle(img_match, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
                
                marker_name = f"{self.marker_prefix} UI {self._get_next_marker_id()}"
                marker_note = f"Match: {max_val*100:.1f}%"
                self._add_marker(timeline_frame, marker_name, marker_note, "UI")
                self.status.emit(f"UI match at frame {timeline_frame} ({max_val*100:.1f}%)")
                self.match_verified.emit(img_match)
                self.event_marker_added = True
            elif not was_matching:
                was_matching = True
        return True

    def _detect_prolonged_ui(self, proc_frame: np.ndarray, template_gray: np.ndarray,
                             timeline_frame: int, was_matching: bool) -> bool:
        curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(curr_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val >= self.threshold:
            self._ui_match_count += 1
        else:
            self._ui_match_count = 0
            
        if self._ui_match_count >= self.min_segment_length:
            if not was_matching:
                if self._passes_audio_filter(timeline_frame):
                    h, w = template_gray.shape
                    img_match = proc_frame.copy()
                    cv2.rectangle(img_match, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
                    
                    marker_name = f"{self.marker_prefix} Results {self._get_next_marker_id()}"
                    marker_note = f"UI Sustained: {max_val*100:.1f}%"
                    self._add_marker(timeline_frame, marker_name, marker_note, "Results")
                    self.status.emit(f"Results screen found at frame {timeline_frame}")
                    self.match_verified.emit(img_match)
                    self.event_marker_added = True
                    return True
                elif not was_matching:
                    return True
            return True
        elif self._ui_match_count == 0:
            return False
        return was_matching

    def _detect_health_bar(self, proc_frame: np.ndarray, timeline_frame: int,
                           was_matching: bool) -> bool:
        if not self.roi_rect or self._color_bounds is None:
            return False
            
        x, y, w, h = self.roi_rect
        ih, iw = proc_frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(iw-x, w), min(ih-y, h)
        if w <= 0 or h <= 0: return False
        
        crop = proc_frame[y:y+h, x:x+w]
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower, upper = self._color_bounds
        mask = cv2.inRange(crop_hsv, lower, upper)
        
        col_sum = np.sum(mask, axis=0) / 255
        width = np.count_nonzero(col_sum)
        ratio = width / w if w > 0 else 0
        
        if ratio <= 0.01 and was_matching:
            if self._passes_audio_filter(timeline_frame):
                marker_name = f"{self.marker_prefix} Defeated {self._get_next_marker_id()}"
                marker_note = f"HP Depleted"
                self._add_marker(timeline_frame, marker_name, marker_note, "HP-0")
                self.status.emit(f"Boss defeated at frame {timeline_frame}")
            self.event_marker_added = False
            return False
            
        if ratio > 0.01 and ratio < 0.10 and not was_matching:
            if self._passes_audio_filter(timeline_frame):
                marker_name = f"{self.marker_prefix} Low HP {self._get_next_marker_id()}"
                marker_note = f"HP: {ratio*100:.1f}%"
                self._add_marker(timeline_frame, marker_name, marker_note, "HP-Low")
                self.status.emit(f"Low health at frame {timeline_frame} ({ratio*100:.1f}%)")
            self.event_marker_added = False
            return True
            
        self.event_marker_added = False
        return ratio > 0.01

    def _detect_loading_screen(self, prev_frame: np.ndarray, proc_frame: np.ndarray, static_count: int,
                               timeline_frame: int) -> int:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        
        prev_blur = cv2.GaussianBlur(prev_gray, (21, 21), 0)
        curr_blur = cv2.GaussianBlur(curr_gray, (21, 21), 0)
        diff = cv2.absdiff(prev_blur, curr_blur)
        change_ratio = cv2.countNonZero(cv2.compare(diff, 5, cv2.CMP_GT)) / diff.size
        
        if change_ratio < (1.0 - self.threshold):
            static_count += 1
            if static_count == self.min_segment_length:
                marker_name = f"{self.marker_prefix} Loading {self._get_next_marker_id()}"
                marker_note = f"Static frame started"
                self._add_marker(timeline_frame - static_count, marker_name, marker_note, "Loading")
                self.status.emit(f"Loading screen at frame {timeline_frame}")
        else:
            if static_count >= self.min_segment_length:
                marker_name = f"{self.marker_prefix} Level Start {self._get_next_marker_id()}"
                marker_note = f"Transition from static"
                self._add_marker(timeline_frame, marker_name, marker_note, "LevelStart")
                self.status.emit(f"Level start at frame {timeline_frame}")
            static_count = 0
            
        return static_count

    def _detect_fade(self, proc_frame: np.ndarray, timeline_frame: int, was_matching: bool) -> bool:
        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        
        is_black = mean_val < self.threshold
        is_white = mean_val > (255.0 - self.threshold)
        
        if is_black or is_white:
            self._static_count += 1
        else:
            self._static_count = 0
            
        if self._static_count >= self.min_segment_length:
            if not was_matching:
                if self._passes_audio_filter(timeline_frame):
                    marker_name = f"{self.marker_prefix} Fade {self._get_next_marker_id()}"
                    marker_note = f"Brightness: {mean_val:.1f}"
                    self._add_marker(timeline_frame, marker_name, marker_note, "Fade")
                    self.status.emit(f"Fade detected at frame {timeline_frame}")
                    self.event_marker_added = True
                    return True
                elif not was_matching:
                    return True
            return True
        elif self._static_count == 0:
            self.event_marker_added = False
            return False
        return was_matching

    def _detect_freeze(self, prev_frame: np.ndarray, proc_frame: np.ndarray, timeline_frame: int,
                       was_matching: bool) -> bool:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        
        curr_edges = cv2.Canny(curr_gray, 50, 150)
        edge_density = cv2.countNonZero(curr_edges) / curr_edges.size
        
        if edge_density > 0.02:
            diff = cv2.absdiff(prev_gray, curr_gray)
            change_ratio = cv2.countNonZero(cv2.compare(diff, 5, cv2.CMP_GT)) / diff.size
            
            if change_ratio < (1.0 - self.threshold):
                self._static_count += 1
            else:
                self._static_count = 0
                
            if self._static_count >= self.min_segment_length:
                if not was_matching:
                    if self._passes_audio_filter(timeline_frame):
                        marker_name = f"{self.marker_prefix} Freeze {self._get_next_marker_id()}"
                        marker_note = "Screen Frozen"
                        self._add_marker(timeline_frame, marker_name, marker_note, "Freeze")
                        self.status.emit(f"Sudden freeze at frame {timeline_frame}")
                        self.event_marker_added = True
                        return True
                    elif not was_matching:
                        return True
                return True
            elif self._static_count == 0:
                self.event_marker_added = False
                return False
        else:
            self._static_count = 0
            
        return was_matching

    def _detect_camera_flick(self, prev_frame: np.ndarray, proc_frame: np.ndarray, timeline_frame: int,
                             was_matching: bool) -> bool:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        
        prev_edges = cv2.Canny(prev_gray, 50, 150)
        curr_edges = cv2.Canny(curr_gray, 50, 150)
        diff = cv2.absdiff(prev_edges, curr_edges)
        change_ratio = cv2.countNonZero(cv2.compare(diff, 30, cv2.CMP_GT)) / diff.size
        
        if change_ratio <= self.threshold:
            self.event_marker_added = False
            return False
        if not was_matching or (self.audio_filter_enabled and not self.event_marker_added):
            if self._passes_audio_filter(timeline_frame):
                marker_name = f"{self.marker_prefix} Flick {self._get_next_marker_id()}"
                marker_note = f"Snap: {change_ratio*100:.1f}%"
                self._add_marker(timeline_frame, marker_name, marker_note, "Flick")
                self.status.emit(f"Camera flick at frame {timeline_frame} ({change_ratio:.3f})")
                self.event_marker_added = True
            elif not was_matching:
                was_matching = True
        return True

    def _detect_face(self, proc_frame: np.ndarray, face_cascade, clahe, timeline_frame: int,
                     was_matching: bool) -> bool:
        curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        if self.use_clahe and clahe is not None:
            curr_gray = clahe.apply(curr_gray)
        faces = face_cascade.detectMultiScale(curr_gray, 1.3, 5)
        if len(faces) == 0:
            self.event_marker_added = False
            return False
        if not was_matching or (self.audio_filter_enabled and not self.event_marker_added):
            if self._passes_audio_filter(timeline_frame):
                face_info = ", ".join([f"{w}x{h}" for (x,y,w,h) in faces[:3]])
                marker_name = f"{self.marker_prefix} Face {self._get_next_marker_id()}"
                marker_note = f"Faces: {len(faces)} | Sizes: {face_info}"
                self._add_marker(timeline_frame, marker_name, marker_note, "Face")
                self.status.emit(f"Face detected at frame {timeline_frame} ({len(faces)} faces)")
                self.event_marker_added = True
            elif not was_matching:
                was_matching = True
        return True

    def _detect_focus(self, proc_frame: np.ndarray, clahe, timeline_frame: int,
                      was_matching: bool) -> bool:
        curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        if self.use_clahe and clahe is not None:
            curr_gray = clahe.apply(curr_gray)
        variance = cv2.Laplacian(curr_gray, cv2.CV_64F).var()
        if variance >= self.threshold:
            self.event_marker_added = False
            return False
        if not was_matching or (self.audio_filter_enabled and not self.event_marker_added):
            if self._passes_audio_filter(timeline_frame):
                marker_name = f"{self.marker_prefix} Blur {self._get_next_marker_id()}"
                marker_note = f"Focus Var: {variance:.1f} (Thr: {self.threshold:.1f})"
                self._add_marker(timeline_frame, marker_name, marker_note, "Blur")
                self.status.emit(f"Blurry shot at frame {timeline_frame} (Var: {variance:.1f})")
                self.event_marker_added = True
            elif not was_matching:
                was_matching = True
        return True

    def _detect_color(self, proc_frame: np.ndarray, timeline_frame: int,
                      was_matching: bool) -> bool:
        if self._color_bounds is None:
            return False
        lower, upper = self._color_bounds
        curr_hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(curr_hsv, lower, upper)
        color_ratio = cv2.countNonZero(mask) / mask.size
        if color_ratio <= self.threshold:
            self.event_marker_added = False
            return False
        if not was_matching or (self.audio_filter_enabled and not self.event_marker_added):
            if self._passes_audio_filter(timeline_frame):
                marker_name = f"{self.marker_prefix} Color {self._get_next_marker_id()}"
                marker_note = f"Color Area: {color_ratio*100:.1f}%"
                self._add_marker(timeline_frame, marker_name, marker_note, "Color")
                self.status.emit(f"Color match at frame {timeline_frame} ({color_ratio*100:.1f}%)")
                self.event_marker_added = True
            elif not was_matching:
                was_matching = True
        return True

    def _detect_flash(self, prev_frame: np.ndarray, proc_frame: np.ndarray, timeline_frame: int,
                      was_matching: bool) -> bool:
        prev_mean = np.mean(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
        curr_mean = np.mean(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY))
        flash_diff = curr_mean - prev_mean
        if flash_diff <= self.threshold:
            self.event_marker_added = False
            return False
        if not was_matching or (self.audio_filter_enabled and not self.event_marker_added):
            if self._passes_audio_filter(timeline_frame):
                marker_name = f"{self.marker_prefix} Flash {self._get_next_marker_id()}"
                marker_note = f"Brightness Spike: +{flash_diff:.1f}"
                self._add_marker(timeline_frame, marker_name, marker_note, "Flash")
                self.status.emit(f"Flash detected at frame {timeline_frame} (+{flash_diff:.1f})")
                self.event_marker_added = True
            elif not was_matching:
                was_matching = True
        return True

    def _emit_progress(self, frame_count: int, total_to_analyze: int, start_time: float):
        if total_to_analyze <= 0:
            return
        progress_percent = min(int((frame_count / total_to_analyze) * 100), 99)
        elapsed = time.time() - start_time
        if progress_percent > 2 and elapsed > 0:
            total_est = elapsed / (progress_percent / 100)
            remaining = total_est - elapsed
            eta_str = str(timedelta(seconds=int(remaining)))
            fps = frame_count / elapsed if elapsed > 0 else 0
            self.status.emit(f"Processing... {progress_percent}% | {frame_count}f | {fps:.1f} fps | ETA: {eta_str}")
        self.progress.emit(progress_percent)


# ============================================================
# TEMPLATE PREVIEW WIDGET
# ============================================================
class TemplatePreviewWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedSize(220, 130)
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        self.label = QLabel("No template")
        self.label.setObjectName("previewLabel")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(False) 
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.current_pixmap: Optional[QPixmap] = None
        self.scale_factor = 1.0
        self.roi_rect = None

    def set_frame(self, frame: Optional[np.ndarray]):
        if frame is None:
            self.label.setText("No template")
            self.label.setPixmap(QPixmap())
            self.current_pixmap = None
            self.roi_rect = None
            return
        self.current_pixmap = cv_img_to_pixmap(frame)
        scaled_pix = self.current_pixmap.scaled(THUMB_W, THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scale_factor = scaled_pix.width() / frame.shape[1]
        self.label.setPixmap(scaled_pix)
        self.roi_rect = None
        self.update_roi_preview()

    def set_roi(self, roi: Optional[Tuple[int, int, int, int]]):
        self.roi_rect = roi
        self.update_roi_preview()

    def update_roi_preview(self):
        if not self.current_pixmap:
            return
        if self.roi_rect:
            scaled_pix = self.current_pixmap.scaled(THUMB_W, THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter = QPainter(scaled_pix)
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            x, y, w, h = self.roi_rect
            sx = int(x * self.scale_factor)
            sy = int(y * self.scale_factor)
            sw = int(w * self.scale_factor)
            sh = int(h * self.scale_factor)
            painter.drawRect(sx, sy, sw, sh)
            painter.end()
            self.label.setPixmap(scaled_pix)
        else:
            scaled_pix = self.current_pixmap.scaled(THUMB_W, THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pix)


# ============================================================
# MAIN WIDGET
# ============================================================
class CVMarkerExtension(QWidget):
    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.setWindowTitle("CV Matcher Pro v9.2 (Gaming & Transitions)")
        self.resize(600, 820)
        self.setStyleSheet(DARK_STYLE)
        
        self.worker: Optional[CVWorker] = None
        self.template_path: Optional[str] = None
        self.template_frame_data: Optional[np.ndarray] = None
        self.marker_history: List[Dict] = []
        self.fps = 24
        self.target_color_hsv: Optional[Tuple[int, int, int]] = None
        
        self.batch_clips: Optional[List] = None
        self.batch_index = 0
        self.batch_total = 0
        self.batch_params: Dict = {}
        
        # Ensure profiles directory exists
        if not os.path.exists(PROFILES_DIR):
            try:
                os.makedirs(PROFILES_DIR)
            except Exception:
                pass
                
        self._build_ui()
        self._load_settings()
        self._refresh_profiles()
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Esc"), self, activated=self.cancel_analysis)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.run_cv)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_last_analysis)

    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(6)
        
        header_layout = QHBoxLayout()
        title = QLabel("CV Segment & Matcher Pro v9.2")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignLeft)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        self.on_top_check = QCheckBox("📌 Always on Top")
        self.on_top_check.setToolTip("Keep window above DaVinci Resolve")
        self.on_top_check.toggled.connect(self.toggle_on_top)
        header_layout.addWidget(self.on_top_check)
        main_layout.addLayout(header_layout)

        self.tabs = QTabWidget()
        
        # --- SETTINGS TAB ---
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setSpacing(8)
        
        mode_group = QGroupBox("Analysis Mode")
        mode_layout = QVBoxLayout()
        self.mode_group = QButtonGroup()
        
        self.radio_scene = QRadioButton("🎬 Scene Match — Detect cuts/changes")
        self.radio_segment = QRadioButton("🧩 Segment Match — Find image portion (Advanced AKAZE)")
        self.radio_recur = QRadioButton("🔁 Recurring Match — Find recurring segments (ORB)")
        self.radio_motion = QRadioButton("🏃 Motion Detection — Detect movement")
        self.radio_hud = QRadioButton("🎮 HUD Motion (ROI) — Detect motion in specific box")
        self.radio_ui = QRadioButton("🎯 UI Template Match — Find exact 2D icons (Loot)")
        self.radio_prolonged_ui = QRadioButton("📋 Prolonged UI Match — Find results screens (Duration)")
        self.radio_health = QRadioButton("❤️ Health Bar Tracker — Boss HP / Low Health")
        self.radio_loading = QRadioButton("⏳ Loading Screen — Detect static frames / level transitions")
        self.radio_fade = QRadioButton("⬛ Fade to Black/White — Detect mission transitions")
        self.radio_freeze = QRadioButton("🧊 Sudden Freeze — Game Over / You Died")
        self.radio_flick = QRadioButton("💨 Camera Flick — Detect fast pans/snaps")
        self.radio_face = QRadioButton("👤 Face Detection — Detect faces")
        self.radio_focus = QRadioButton("🔍 Focus Detection — Find blurry shots")
        self.radio_color = QRadioButton("🌈 Color Match — Find specific color dominance")
        self.radio_flash = QRadioButton("⚡ Flash Detection — Detect strobes/flashes")
        
        self.radio_scene.setChecked(True)
        for rb in [self.radio_scene, self.radio_segment, self.radio_recur, self.radio_motion, 
                   self.radio_hud, self.radio_ui, self.radio_prolonged_ui, self.radio_health, self.radio_loading,
                   self.radio_fade, self.radio_freeze, self.radio_flick, self.radio_face, 
                   self.radio_focus, self.radio_color, self.radio_flash]:
            self.mode_group.addButton(rb)
            mode_layout.addWidget(rb)
        
        mode_group.setLayout(mode_layout)
        settings_layout.addWidget(mode_group)

        self.scene_method_group = QGroupBox("Scene Detection Method")
        scene_method_layout = QHBoxLayout()
        scene_method_label = QLabel("Method:")
        self.scene_method_combo = QComboBox()
        self.scene_method_combo.addItems(["Histogram (HSV)", "Edges (Canny)", "Combined (Hist+Brightness)"])
        scene_method_layout.addWidget(scene_method_label)
        scene_method_layout.addWidget(self.scene_method_combo)
        scene_method_layout.addStretch()
        self.scene_method_group.setLayout(scene_method_layout)
        settings_layout.addWidget(self.scene_method_group)

        self.template_group = QGroupBox("Template Source & Visual ROI Picker")
        template_layout = QVBoxLayout()
        
        # Profiles UI
        profile_row1 = QHBoxLayout()
        profile_row1.addWidget(QLabel("Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(150)
        profile_row1.addWidget(self.profile_combo, 1)
        
        self.load_profile_btn = QPushButton("📂 Load")
        self.load_profile_btn.clicked.connect(self._load_profile)
        profile_row1.addWidget(self.load_profile_btn)
        
        self.del_profile_btn = QPushButton("🗑️ Delete")
        self.del_profile_btn.clicked.connect(self._delete_profile)
        profile_row1.addWidget(self.del_profile_btn)
        template_layout.addLayout(profile_row1)
        
        profile_row2 = QHBoxLayout()
        self.profile_name_input = QLineEdit()
        self.profile_name_input.setPlaceholderText("Enter profile name...")
        profile_row2.addWidget(self.profile_name_input, 1)
        
        self.save_profile_btn = QPushButton("💾 Save Profile")
        self.save_profile_btn.clicked.connect(self._save_profile)
        profile_row2.addWidget(self.save_profile_btn)
        template_layout.addLayout(profile_row2)
        
        file_row = QHBoxLayout()
        self.file_btn = QPushButton("📁 Load Image File")
        self.file_btn.clicked.connect(self.load_image_file)
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #888;")
        file_row.addWidget(self.file_btn)
        file_row.addWidget(self.file_label, 1)
        template_layout.addLayout(file_row)
        
        grab_row = QHBoxLayout()
        self.grab_btn = QPushButton("🎥 Grab from Playhead")
        self.grab_btn.clicked.connect(self.grab_playhead_template)
        self.grab_label = QLabel("No frame grabbed")
        self.grab_label.setStyleSheet("color: #888;")
        self.grab_label.setWordWrap(True)
        grab_row.addWidget(self.grab_btn)
        grab_row.addWidget(self.grab_label, 1)
        template_layout.addLayout(grab_row)
        
        picker_row = QHBoxLayout()
        self.pick_roi_btn = QPushButton("✂️ Pick Section / Marker (Visual roi)")
        self.pick_roi_btn.clicked.connect(self.open_visual_picker)
        self.pick_roi_btn.setEnabled(False)
        self.pick_roi_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        picker_row.addWidget(self.pick_roi_btn)
        template_layout.addLayout(picker_row)
        
        preview_row = QHBoxLayout()
        preview_label = QLabel("Preview:\n(Shows selected area)")
        preview_label.setObjectName("sectionLabel")
        self.preview_widget = TemplatePreviewWidget()
        preview_row.addWidget(preview_label)
        preview_row.addWidget(self.preview_widget)
        preview_row.addStretch()
        template_layout.addLayout(preview_row)

        verify_row = QHBoxLayout()
        verify_label = QLabel("Match Verification:\n(Green box = found)")
        verify_label.setObjectName("sectionLabel")
        self.verify_thumb_label = QLabel("Waiting for match...")
        self.verify_thumb_label.setFixedSize(220, 130)
        self.verify_thumb_label.setObjectName("previewLabel")
        self.verify_thumb_label.setAlignment(Qt.AlignCenter)
        verify_row.addWidget(verify_label)
        verify_row.addWidget(self.verify_thumb_label)
        verify_row.addStretch()
        template_layout.addLayout(verify_row)
        self.template_group.setLayout(template_layout)
        settings_layout.addWidget(self.template_group)

        self.color_picker_group = QGroupBox("Target Color")
        color_picker_layout = QHBoxLayout()
        self.pick_color_btn = QPushButton("🎨 Pick Target Color")
        self.pick_color_btn.clicked.connect(self.pick_target_color)
        self.color_preview_label = QLabel()
        self.color_preview_label.setFixedSize(30, 20)
        self.color_preview_label.setStyleSheet("background-color: #000;")
        self.color_info_label = QLabel("No color picked")
        self.color_info_label.setStyleSheet("color: #888;")
        color_picker_layout.addWidget(self.pick_color_btn)
        color_picker_layout.addWidget(self.color_preview_label)
        color_picker_layout.addWidget(self.color_info_label, 1)
        self.color_picker_group.setLayout(color_picker_layout)
        settings_layout.addWidget(self.color_picker_group)

        thresh_group = QGroupBox("Threshold / Sensitivity")
        thresh_layout = QGridLayout()
        preset_label = QLabel("Presets:")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "Low Sensitivity", "Medium", "High Sensitivity"])
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        thresh_layout.addWidget(preset_label, 0, 0)
        thresh_layout.addWidget(self.preset_combo, 0, 1)
        
        self.thresh_label = QLabel("Threshold:")
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(5, 95)
        self.thresh_slider.setValue(50)
        self.thresh_value_label = QLabel("0.50")
        self.thresh_value_label.setMinimumWidth(50)
        self.thresh_slider.valueChanged.connect(self.update_thresh_label)
        self.thresh_slider.valueChanged.connect(lambda: self.preset_combo.setCurrentIndex(0))
        thresh_layout.addWidget(self.thresh_label, 1, 0)
        thresh_layout.addWidget(self.thresh_slider, 1, 1)
        thresh_layout.addWidget(self.thresh_value_label, 1, 2)
        
        self.orb_dist_label = QLabel("ORB Match Distance:")
        self.orb_dist_slider = QSlider(Qt.Horizontal)
        self.orb_dist_slider.setRange(20, 100)
        self.orb_dist_slider.setValue(50)
        self.orb_dist_value_label = QLabel("50")
        self.orb_dist_slider.valueChanged.connect(lambda v: self.orb_dist_value_label.setText(str(v)))
        thresh_layout.addWidget(self.orb_dist_label, 2, 0)
        thresh_layout.addWidget(self.orb_dist_slider, 2, 1)
        thresh_layout.addWidget(self.orb_dist_value_label, 2, 2)
        thresh_group.setLayout(thresh_layout)
        settings_layout.addWidget(thresh_group)

        # AUDIO FILTER GROUP
        audio_group = QGroupBox("Audio Spike Filter (Optional)")
        audio_layout = QGridLayout()
        self.audio_check = QCheckBox("Require Audio Spike (Screams/Explosions)")
        audio_layout.addWidget(self.audio_check, 0, 0, 1, 3)
        
        self.audio_sustained_check = QCheckBox("Require Sustained Audio (Jingle/Stinger)")
        audio_layout.addWidget(self.audio_sustained_check, 1, 0, 1, 3)
        
        audio_layout.addWidget(QLabel("Audio Threshold (dB):"), 2, 0)
        self.audio_thresh_slider = QSlider(Qt.Horizontal)
        self.audio_thresh_slider.setRange(-60, 0)
        self.audio_thresh_slider.setValue(-15)
        self.audio_thresh_value_label = QLabel("-15 dB")
        self.audio_thresh_value_label.setMinimumWidth(50)
        self.audio_thresh_slider.valueChanged.connect(lambda v: self.audio_thresh_value_label.setText(f"{v} dB"))
        audio_layout.addWidget(self.audio_thresh_slider, 2, 1)
        audio_layout.addWidget(self.audio_thresh_value_label, 2, 2)
        audio_group.setLayout(audio_layout)
        settings_layout.addWidget(audio_group)

        range_group = QGroupBox("Analysis Range (Current Clip)")
        range_layout = QHBoxLayout()
        self.range_check = QCheckBox("Process Custom Range:")
        self.range_check.toggled.connect(self.toggle_range)
        range_layout.addWidget(self.range_check)
        range_layout.addWidget(QLabel("From:"))
        self.range_start_spin = QSpinBox()
        self.range_start_spin.setRange(0, 999999)
        self.range_start_spin.setEnabled(False)
        range_layout.addWidget(self.range_start_spin)
        range_layout.addWidget(QLabel("To:"))
        self.range_end_spin = QSpinBox()
        self.range_end_spin.setRange(0, 999999)
        self.range_end_spin.setEnabled(False)
        range_layout.addWidget(self.range_end_spin)
        range_group.setLayout(range_layout)
        settings_layout.addWidget(range_group)

        adv_group = QGroupBox("Advanced Settings")
        adv_layout = QGridLayout()
        skip_label = QLabel("Frame Skip:")
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(1, 30)
        self.skip_spin.setValue(1)
        adv_layout.addWidget(skip_label, 0, 0)
        adv_layout.addWidget(self.skip_spin, 0, 1)
        
        dur_label = QLabel("Marker Duration (frames):")
        self.dur_spin = QSpinBox()
        self.dur_spin.setRange(1, 10000)
        self.dur_spin.setValue(1)
        adv_layout.addWidget(dur_label, 1, 0)
        adv_layout.addWidget(self.dur_spin, 1, 1)
        
        self.min_seg_label = QLabel("Min Segment Length (frames):")
        self.min_seg_spin = QSpinBox()
        self.min_seg_spin.setRange(1, 1000)
        self.min_seg_spin.setValue(5)
        adv_layout.addWidget(self.min_seg_label, 2, 0)
        adv_layout.addWidget(self.min_seg_spin, 2, 1)
        
        res_label = QLabel("Max Analysis Res:")
        self.res_combo = QComboBox()
        self.res_combo.addItems(["720p", "1080p", "Original"])
        self.res_combo.setCurrentText("1080p")
        adv_layout.addWidget(res_label, 3, 0)
        adv_layout.addWidget(self.res_combo, 3, 1)

        self.clahe_check = QCheckBox("🌗 Enable CLAHE (Enhances flat/low-contrast features)")
        adv_layout.addWidget(self.clahe_check, 4, 0, 1, 2)
        self.auto_fit_check = QCheckBox("📏 Auto-fit marker duration (fill gaps between markers)")
        adv_layout.addWidget(self.auto_fit_check, 5, 0, 1, 2)
        self.sync_playhead_check = QCheckBox("🔄 Sync playhead to markers as they are found")
        adv_layout.addWidget(self.sync_playhead_check, 6, 0, 1, 2)
        adv_group.setLayout(adv_layout)
        settings_layout.addWidget(adv_group)

        marker_group = QGroupBox("Marker Appearance")
        marker_layout = QGridLayout()
        color_label = QLabel("Marker Color:")
        self.color_dropdown = QComboBox()
        self.color_dropdown.addItems(["Blue", "Red", "Green", "Yellow", "Pink", "Cyan", "Purple", "Orange"])
        self.color_dropdown.setCurrentText("Green")
        marker_layout.addWidget(color_label, 0, 0)
        marker_layout.addWidget(self.color_dropdown, 0, 1)
        prefix_label = QLabel("Marker Prefix:")
        self.prefix_input = QLineEdit("CV")
        marker_layout.addWidget(prefix_label, 1, 0)
        marker_layout.addWidget(self.prefix_input, 1, 1)
        marker_group.setLayout(marker_layout)
        settings_layout.addWidget(marker_group)
        settings_layout.addStretch()
        self.tabs.addTab(settings_widget, "⚙️ Settings")

        # --- TOOLS TAB ---
        tools_widget = QWidget()
        tools_layout = QVBoxLayout(tools_widget)
        
        batch_group = QGroupBox("Batch Processing")
        batch_layout = QVBoxLayout()
        batch_desc = QLabel("Analyze all clips on the current video track (V1).")
        batch_desc.setWordWrap(True)
        batch_desc.setStyleSheet("color: #999;")
        self.batch_check = QCheckBox("Process entire timeline (all clips on V1)")
        batch_layout.addWidget(batch_desc)
        batch_layout.addWidget(self.batch_check)
        batch_group.setLayout(batch_layout)
        tools_layout.addWidget(batch_group)
        
        export_group = QGroupBox("Export / Import Markers")
        export_layout = QVBoxLayout()
        export_row = QHBoxLayout()
        self.export_csv_btn = QPushButton("📥 Export to CSV")
        self.export_csv_btn.clicked.connect(self.export_markers_csv)
        self.export_srt_btn = QPushButton("📥 Export to SRT")
        self.export_srt_btn.clicked.connect(self.export_markers_srt)
        export_row.addWidget(self.export_csv_btn)
        export_row.addWidget(self.export_srt_btn)
        export_group.setLayout(export_layout)
        tools_layout.addWidget(export_group)
        
        stats_group = QGroupBox("Marker Statistics")
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel("No markers yet.")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        refresh_btn = QPushButton("🔄 Refresh Stats")
        refresh_btn.clicked.connect(self.update_marker_stats)
        stats_layout.addWidget(refresh_btn)
        stats_group.setLayout(stats_layout)
        tools_layout.addWidget(stats_group)
        
        # SMART CLEANUP GROUP
        cleanup_group = QGroupBox("Smart Marker Cleanup (Merge & Pad)")
        cleanup_layout = QGridLayout()
        cleanup_layout.addWidget(QLabel("Merge Gap (frames):"), 0, 0)
        self.merge_gap_spin = QSpinBox()
        self.merge_gap_spin.setRange(1, 1000)
        self.merge_gap_spin.setValue(15)
        cleanup_layout.addWidget(self.merge_gap_spin, 0, 1)
        
        cleanup_layout.addWidget(QLabel("Min Duration (frames):"), 1, 0)
        self.min_dur_spin = QSpinBox()
        self.min_dur_spin.setRange(1, 10000)
        self.min_dur_spin.setValue(10)
        cleanup_layout.addWidget(self.min_dur_spin, 1, 1)
        
        self.cleanup_btn = QPushButton("🧹 Clean Up Markers")
        self.cleanup_btn.clicked.connect(self.cleanup_markers)
        cleanup_layout.addWidget(self.cleanup_btn, 2, 0, 1, 2)
        cleanup_group.setLayout(cleanup_layout)
        tools_layout.addWidget(cleanup_group)
        
        nav_group = QGroupBox("Timeline Navigation")
        nav_layout = QHBoxLayout()
        self.prev_marker_btn = QPushButton("⏮ Prev Marker")
        self.prev_marker_btn.clicked.connect(self.goto_prev_marker)
        self.next_marker_btn = QPushButton("Next Marker ⏭")
        self.next_marker_btn.clicked.connect(self.goto_next_marker)
        nav_layout.addWidget(self.prev_marker_btn)
        nav_layout.addWidget(self.next_marker_btn)
        nav_group.setLayout(nav_layout)
        tools_layout.addWidget(nav_group)
        
        undo_group = QGroupBox("History")
        undo_layout = QVBoxLayout()
        self.undo_btn = QPushButton("↩️ Undo Last Analysis (Remove last batch of markers)")
        self.undo_btn.setObjectName("dangerBtn")
        self.undo_btn.clicked.connect(self.undo_last_analysis)
        undo_layout.addWidget(self.undo_btn)
        undo_group.setLayout(undo_layout)
        tools_layout.addWidget(undo_group)
        tools_layout.addStretch()
        self.tabs.addTab(tools_widget, "🛠️ Tools")
        main_layout.addWidget(self.tabs)

        action_layout = QHBoxLayout()
        self.run_btn = QPushButton("▶️  Run Analysis")
        self.run_btn.setObjectName("runBtn")
        self.run_btn.setToolTip("Ctrl+R")
        self.run_btn.clicked.connect(self.run_cv)
        self.cancel_btn = QPushButton("⏹ Cancel")
        self.cancel_btn.setObjectName("dangerBtn")
        self.cancel_btn.setToolTip("Esc")
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        self.cancel_btn.setEnabled(False)
        self.clear_btn = QPushButton("🗑️ Clear Markers")
        self.clear_btn.clicked.connect(self.clear_markers)
        action_layout.addWidget(self.run_btn)
        action_layout.addWidget(self.cancel_btn)
        action_layout.addWidget(self.clear_btn)
        main_layout.addLayout(action_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #8a8; padding: 2px;")
        main_layout.addWidget(self.status_label)
        
        feed_header_layout = QHBoxLayout()
        feed_label = QLabel("Live Marker Feed:")
        feed_label.setObjectName("sectionLabel")
        feed_header_layout.addWidget(feed_label)
        feed_header_layout.addStretch()
        self.clear_feed_btn = QPushButton("Clear Feed")
        self.clear_feed_btn.setFixedHeight(20)
        self.clear_feed_btn.clicked.connect(self._clear_feed)
        feed_header_layout.addWidget(self.clear_feed_btn)
        main_layout.addLayout(feed_header_layout)
        
        self.feed_scroll = QScrollArea()
        self.feed_scroll.setWidgetResizable(True)
        self.feed_scroll.setMaximumHeight(100)
        self.feed_scroll.setMinimumHeight(60)
        feed_container = QWidget()
        self.feed_layout = QVBoxLayout(feed_container)
        self.feed_layout.setContentsMargins(4, 4, 4, 4)
        self.feed_layout.setSpacing(2)
        self.feed_layout.addStretch()
        self.feed_scroll.setWidget(feed_container)
        main_layout.addWidget(self.feed_scroll)
        self.setLayout(main_layout)
        
        for rb in [self.radio_scene, self.radio_segment, self.radio_recur, self.radio_motion, 
                   self.radio_hud, self.radio_ui, self.radio_prolonged_ui, self.radio_health, 
                   self.radio_loading, self.radio_fade, self.radio_freeze, self.radio_flick, 
                   self.radio_face, self.radio_focus, self.radio_color, self.radio_flash]:
            rb.toggled.connect(self.update_ui_mode)
        self.update_ui_mode()

        # Disable run button if Resolve API is not found
        if not resolve:
            self.run_btn.setEnabled(False)
            self.run_btn.setToolTip("DaVinci Resolve API not found. Run this script inside Resolve.")
            self.status_label.setText("❌ Status: DaVinci Resolve API not found.")

    def _refresh_profiles(self):
        self.profile_combo.clear()
        try:
            files = [f.replace('.json', '') for f in os.listdir(PROFILES_DIR) if f.endswith('.json')]
            self.profile_combo.addItems(sorted(files))
        except Exception:
            pass

    def _save_profile(self):
        if self.template_frame_data is None:
            QMessageBox.warning(self, "Warning", "Please load or grab a template image first.")
            return
            
        name = self.profile_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a profile name.")
            return
            
        try:
            ret, buffer = cv2.imencode('.png', self.template_frame_data)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            data = {
                'roi': self.preview_widget.roi_rect,
                'image': img_str
            }
            
            file_path = os.path.join(PROFILES_DIR, f"{name}.json")
            with open(file_path, 'w') as f:
                json.dump(data, f)
                
            self._refresh_profiles()
            self.profile_combo.setCurrentText(name)
            self.status_label.setText(f"💾 Profile '{name}' saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save profile:\n{e}")

    def _load_profile(self):
        name = self.profile_combo.currentText()
        if not name:
            QMessageBox.warning(self, "Warning", "No profile selected.")
            return
            
        file_path = os.path.join(PROFILES_DIR, f"{name}.json")
        if not os.path.exists(file_path):
            return
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            img_str = data.get('image')
            roi = data.get('roi')
            
            if img_str:
                nparr = np.frombuffer(base64.b64decode(img_str), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    self.template_frame_data = img
                    self.file_label.setText(f"✅ {name}")
                    self.preview_widget.set_frame(img)
                    self.pick_roi_btn.setEnabled(True)
                    
            if roi and isinstance(roi, list) and len(roi) == 4:
                self.preview_widget.set_roi(tuple(roi))
                
            self.profile_name_input.setText(name)
            self.status_label.setText(f"📂 Profile '{name}' loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load profile:\n{e}")

    def _delete_profile(self):
        name = self.profile_combo.currentText()
        if not name:
            return
            
        reply = QMessageBox.question(self, "Delete Profile", f"Are you sure you want to delete '{name}'?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            file_path = os.path.join(PROFILES_DIR, f"{name}.json")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                self._refresh_profiles()
                self.status_label.setText(f"🗑️ Profile '{name}' deleted.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete profile:\n{e}")

    def _set_ui_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.tabs.tabBar().setTabEnabled(0, not running)
        self.tabs.tabBar().setTabEnabled(1, not running)
        self.clear_btn.setEnabled(not running)

    def toggle_on_top(self, checked: bool):
        if checked:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()

    def toggle_range(self, checked: bool):
        self.range_start_spin.setEnabled(checked)
        self.range_end_spin.setEnabled(checked)

    def pick_target_color(self):
        color = QColorDialog.getColor(Qt.white, self, "Pick Target Color")
        if color.isValid():
            bgr = np.uint8([[[color.blue(), color.green(), color.red()]]])
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
            self.target_color_hsv = (int(hsv[0]), int(hsv[1]), int(hsv[2]))
            self.color_preview_label.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});")
            self.color_info_label.setText(f"Target HSV: H:{self.target_color_hsv[0]} S:{self.target_color_hsv[1]} V:{self.target_color_hsv[2]}")

    def update_ui_mode(self):
        self.thresh_slider.setVisible(True)
        self.thresh_value_label.setVisible(True)
        
        is_scene = self.radio_scene.isChecked()
        is_segment = self.radio_segment.isChecked()
        is_recur = self.radio_recur.isChecked()
        is_motion = self.radio_motion.isChecked()
        is_hud = self.radio_hud.isChecked()
        is_ui = self.radio_ui.isChecked()
        is_prolonged_ui = self.radio_prolonged_ui.isChecked()
        is_health = self.radio_health.isChecked()
        is_loading = self.radio_loading.isChecked()
        is_fade = self.radio_fade.isChecked()
        is_freeze = self.radio_freeze.isChecked()
        is_flick = self.radio_flick.isChecked()
        is_face = self.radio_face.isChecked()
        is_focus = self.radio_focus.isChecked()
        is_color = self.radio_color.isChecked()
        is_flash = self.radio_flash.isChecked()
        
        needs_template = is_segment or is_recur or is_ui or is_prolonged_ui
        needs_roi = is_segment or is_hud or is_health
        needs_color = is_color or is_health
        needs_orb = is_recur
        
        self.template_group.setVisible(needs_template)
        self.color_picker_group.setVisible(needs_color)
        self.scene_method_group.setVisible(is_scene)
        
        if needs_roi:
            self.pick_roi_btn.setEnabled(self.template_frame_data is not None)
            self.pick_roi_btn.setText("✂️ Pick Region of Interest (ROI)")
        
        self.orb_dist_label.setVisible(needs_orb)
        self.orb_dist_slider.setVisible(needs_orb)
        self.orb_dist_value_label.setVisible(needs_orb)
        
        self.min_seg_label.setVisible(is_scene or is_motion or is_loading or is_fade or is_freeze or is_prolonged_ui)
        self.min_seg_spin.setVisible(is_scene or is_motion or is_loading or is_fade or is_freeze or is_prolonged_ui)
        
        if is_scene:
            self.min_seg_label.setText("Min Segment Length (frames):")
            self.thresh_slider.setRange(5, 95)
            self.thresh_label.setText("Scene Change Threshold:")
        elif is_segment:
            self.thresh_slider.setRange(5, 50) 
            self.thresh_label.setText("Min Geometric Inliers:")
        elif is_recur:
            self.thresh_slider.setRange(10, 150)
            self.thresh_label.setText("Min Feature Matches:")
        elif is_motion or is_hud:
            self.thresh_slider.setRange(1, 100)
            self.thresh_label.setText("Motion Area Threshold (%):")
        elif is_ui or is_prolonged_ui:
            self.thresh_slider.setRange(50, 99)
            self.thresh_label.setText("Template Match Correlation (%):")
            if is_prolonged_ui:
                self.min_seg_label.setText("Min UI Duration (frames):")
        elif is_health:
            self.thresh_slider.setRange(1, 100)
            self.thresh_label.setText("Low HP Trigger Threshold (%):")
        elif is_loading:
            self.thresh_slider.setRange(90, 99)
            self.thresh_label.setText("Static Frame Similarity (%):")
            self.min_seg_label.setText("Min Static Duration (frames):")
        elif is_fade:
            self.thresh_slider.setRange(1, 50)
            self.thresh_label.setText("Brightness Tolerance (0-50):")
            self.min_seg_label.setText("Min Fade Duration (frames):")
        elif is_freeze:
            self.thresh_slider.setRange(90, 99)
            self.thresh_label.setText("Similarity Threshold (%):")
            self.min_seg_label.setText("Min Freeze Duration (frames):")
        elif is_flick:
            self.thresh_slider.setRange(10, 80)
            self.thresh_label.setText("Edge Displacement Threshold (%):")
        elif is_face:
            self.thresh_slider.setVisible(False)
            self.thresh_value_label.setText("Auto")
        elif is_focus:
            self.thresh_slider.setRange(10, 500)
            self.thresh_label.setText("Blur Threshold (Laplacian Var):")
        elif is_color:
            self.thresh_slider.setRange(1, 100)
            self.thresh_label.setText("Color Area Threshold (%):")
        elif is_flash:
            self.thresh_slider.setRange(20, 100)
            self.thresh_label.setText("Flash Brightness Spike:")
            
        self.update_thresh_label()

    def update_thresh_label(self):
        val = self.thresh_slider.value()
        if self.radio_recur.isChecked() or self.radio_segment.isChecked() or self.radio_focus.isChecked() or self.radio_fade.isChecked():
            self.thresh_value_label.setText(str(val))
        elif self.radio_motion.isChecked() or self.radio_hud.isChecked() or self.radio_color.isChecked() or self.radio_health.isChecked():
            self.thresh_value_label.setText(f"{val}%")
        elif self.radio_ui.isChecked() or self.radio_prolonged_ui.isChecked() or self.radio_loading.isChecked() or self.radio_freeze.isChecked() or self.radio_flick.isChecked():
            self.thresh_value_label.setText(f"{val}%")
        elif self.radio_face.isChecked():
            self.thresh_value_label.setText("Auto")
        elif self.radio_flash.isChecked():
            self.thresh_value_label.setText(f"+{val}")
        else:
            self.thresh_value_label.setText(f"{val / 100.0:.2f}")

    def apply_preset(self, index: int):
        if index == 0: return
        mode = self._get_mode()
        preset_map = {
            1: {"Scene Match": 30, "Segment Match": 25, "Recurring Match": 60, "Motion Detection": 20, "HUD Motion": 20, "UI Template Match": 80, "Prolonged UI Match": 80, "Health Bar Tracker": 10, "Loading Screen": 95, "Fade Detection": 10, "Sudden Freeze": 95, "Camera Flick": 40, "Focus Detection": 50, "Color Match": 20, "Flash Detection": 70},
            2: {"Scene Match": 50, "Segment Match": 15, "Recurring Match": 40, "Motion Detection": 10, "HUD Motion": 10, "UI Template Match": 90, "Prolonged UI Match": 90, "Health Bar Tracker": 20, "Loading Screen": 98, "Fade Detection": 5, "Sudden Freeze": 98, "Camera Flick": 30, "Focus Detection": 100, "Color Match": 10, "Flash Detection": 50},
            3: {"Scene Match": 70, "Segment Match": 8, "Recurring Match": 25, "Motion Detection": 5, "HUD Motion": 5, "UI Template Match": 95, "Prolonged UI Match": 95, "Health Bar Tracker": 30, "Loading Screen": 99, "Fade Detection": 2, "Sudden Freeze": 99, "Camera Flick": 20, "Focus Detection": 200, "Color Match": 5, "Flash Detection": 30}
        }
        val = preset_map.get(index, {}).get(mode)
        if val is not None:
            self.thresh_slider.setValue(val)

    def _get_mode(self) -> str:
        if self.radio_scene.isChecked(): return "Scene Match"
        if self.radio_segment.isChecked(): return "Segment Match"
        if self.radio_recur.isChecked(): return "Recurring Match"
        if self.radio_motion.isChecked(): return "Motion Detection"
        if self.radio_hud.isChecked(): return "HUD Motion"
        if self.radio_ui.isChecked(): return "UI Template Match"
        if self.radio_prolonged_ui.isChecked(): return "Prolonged UI Match"
        if self.radio_health.isChecked(): return "Health Bar Tracker"
        if self.radio_loading.isChecked(): return "Loading Screen"
        if self.radio_fade.isChecked(): return "Fade Detection"
        if self.radio_freeze.isChecked(): return "Sudden Freeze"
        if self.radio_flick.isChecked(): return "Camera Flick"
        if self.radio_face.isChecked(): return "Face Detection"
        if self.radio_focus.isChecked(): return "Focus Detection"
        if self.radio_color.isChecked(): return "Color Match"
        if self.radio_flash.isChecked(): return "Flash Detection"
        return "Scene Match"

    def load_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Template Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                QMessageBox.warning(self, "Error", f"Could not read image file:\n{file_path}")
                return
            self.template_path = file_path
            self.template_frame_data = img
            self.file_label.setText(f"✅ {os.path.basename(file_path)}")
            self.preview_widget.set_frame(self.template_frame_data)
            self.pick_roi_btn.setEnabled(True)

    def grab_playhead_template(self):
        if not resolve:
            QMessageBox.critical(self, "Error", "DaVinci Resolve API not found.")
            return
        try:
            project = resolve.GetProjectManager().GetCurrentProject()
            timeline = project.GetCurrentTimeline()
            self.fps = timeline.GetSetting("timelineFrameRate") or 24
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not access timeline:\n{e}")
            return
        
        playhead_frame = timeline.GetPlayhead()
        clip = self._get_clip_at_playhead(timeline, playhead_frame)
        if not clip:
            self.grab_label.setText("❌ No clip found at playhead.")
            return
        clip_start = int(clip.GetStart())
        clip_end = int(clip.GetEnd())
        self.range_start_spin.setValue(clip_start)
        self.range_end_spin.setValue(clip_end)
        clip_path = clip.GetFileProperty("FilePath")
        timeline_start = int(clip.GetStart())
        try:
            left_offset = int(clip.GetLeftOffset())
        except Exception:
            left_offset = 0
        source_frame = left_offset + (playhead_frame - timeline_start)

        cap = cv2.VideoCapture(clip_path)
        try:
            if not cap.isOpened():
                self.grab_label.setText("❌ Could not open clip file.")
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ret, frame = cap.read()
        finally:
            cap.release()

        if ret and frame is not None:
            self.template_frame_data = frame
            self.grab_label.setText(f"✅ Frame {playhead_frame}")
            self.preview_widget.set_frame(frame)
            self.pick_roi_btn.setEnabled(True)
        else:
            self.grab_label.setText("❌ Failed to grab frame.")

    def open_visual_picker(self):
        if self.template_frame_data is None:
            QMessageBox.warning(self, "Warning", "Please load an image or grab a frame first.")
            return
        dialog = VisualPickerDialog(self.template_frame_data, self)
        if dialog.exec_() == QDialog.Accepted:
            roi = dialog.roi_rect
            self.preview_widget.set_roi(roi)
            if roi:
                self.status_label.setText(f"✅ ROI selected: X={roi[0]}, Y={roi[1]}, W={roi[2]}, H={roi[3]}")
            else:
                self.status_label.setText("✅ No specific ROI drawn. Whole image will be matched.")

    def _get_clip_at_playhead(self, timeline, playhead_frame: Optional[int] = None):
        if playhead_frame is None:
            playhead_frame = timeline.GetPlayhead()
        clip = timeline.GetCurrentVideoItem()
        if not clip:
            clips = timeline.GetItemsInTrack("video", 1)
            if clips:
                for c in clips.values():
                    if c.GetStart() <= playhead_frame <= c.GetEnd():
                        clip = c
                        break
        return clip

    def run_cv(self):
        if not resolve:
            self.status_label.setText("❌ Resolve API not found.")
            return
        try:
            project = resolve.GetProjectManager().GetCurrentProject()
            timeline = project.GetCurrentTimeline()
        except Exception as e:
            self.status_label.setText(f"❌ Could not access timeline: {e}")
            return
        
        mode = self._get_mode()
        needs_template = mode in ("Segment Match", "Recurring Match", "UI Template Match", "Prolonged UI Match")
        needs_roi = mode in ("HUD Motion", "Health Bar Tracker", "Segment Match")
        needs_color = mode in ("Color Match", "Health Bar Tracker")
        
        if needs_template:
            if self.template_frame_data is None or not isinstance(self.template_frame_data, np.ndarray) or self.template_frame_data.size == 0:
                self.status_label.setText(f"❌ Please load image/grab frame first.")
                return
                
        if needs_roi and self.preview_widget.roi_rect is None:
            self.status_label.setText("❌ Please pick a Region of Interest (ROI) first.")
            return

        if needs_color and self.target_color_hsv is None:
            self.status_label.setText("❌ Please pick a target color first.")
            return

        if mode == "Recurring Match":
            threshold = self.thresh_slider.value()
        elif mode in ("Motion Detection", "HUD Motion", "Color Match", "Health Bar Tracker", "UI Template Match", "Prolonged UI Match", "Loading Screen", "Sudden Freeze", "Camera Flick"):
            threshold = self.thresh_slider.value() / 100.0
        elif mode == "Fade Detection":
            threshold = float(self.thresh_slider.value())
        elif mode == "Face Detection":
            threshold = 0.5 
        elif mode in ("Focus Detection", "Flash Detection", "Segment Match"):
            threshold = float(self.thresh_slider.value())
        else:
            threshold = self.thresh_slider.value() / 100.0

        marker_color = self.color_dropdown.currentText()
        marker_duration = self.dur_spin.value()
        frame_skip = self.skip_spin.value()
        orb_distance = self.orb_dist_slider.value()
        scene_method_map = {"Histogram (HSV)": "Histogram", "Edges (Canny)": "Edges", "Combined (Hist+Brightness)": "Combined"}
        scene_method = scene_method_map.get(self.scene_method_combo.currentText(), "Histogram")
        min_segment_length = self.min_seg_spin.value()
        marker_prefix = (self.prefix_input.text() or "CV").upper()
        roi_rect = self.preview_widget.roi_rect if needs_roi else None
        auto_fit = self.auto_fit_check.isChecked()
        sync_playhead = self.sync_playhead_check.isChecked()
        use_clahe = self.clahe_check.isChecked()
        
        res_map = {"720p": 720, "1080p": 1080, "Original": 99999}
        max_analysis_res = res_map.get(self.res_combo.currentText(), 1080)
        
        audio_filter_enabled = self.audio_check.isChecked()
        audio_threshold_db = float(self.audio_thresh_slider.value())
        audio_sustained = self.audio_sustained_check.isChecked()

        if self.batch_check.isChecked():
            self._run_batch(timeline, mode, threshold, marker_color, marker_duration, frame_skip, 
                           orb_distance, scene_method, min_segment_length, marker_prefix, roi_rect, 
                           auto_fit, sync_playhead, use_clahe, max_analysis_res, audio_filter_enabled, audio_threshold_db, audio_sustained)
            return

        clip = self._get_clip_at_playhead(timeline)
        if not clip:
            self.status_label.setText("❌ No clip found at playhead.")
            return

        clip_path = clip.GetFileProperty("FilePath")
        timeline_start_frame = int(clip.GetStart())
        timeline_end_frame = int(clip.GetEnd())
        try:
            source_start_frame = int(clip.GetLeftOffset())
        except Exception:
            source_start_frame = 0
        
        if self.range_check.isChecked():
            r_start = self.range_start_spin.value()
            r_end = self.range_end_spin.value()
            r_start = max(timeline_start_frame, r_start)
            r_end = min(timeline_end_frame, r_end)
            if r_start >= r_end:
                self.status_label.setText("❌ Invalid range. Start must be before End.")
                return
            source_start_frame = source_start_frame + (r_start - timeline_start_frame)
            source_end_frame = source_start_frame + (r_end - r_start)
            timeline_start_frame = r_start
            timeline_end_frame = r_end
        else:
            source_end_frame = source_start_frame + (timeline_end_frame - timeline_start_frame)

        self._start_worker(clip_path, timeline_start_frame, source_start_frame, threshold, 
                          marker_color, mode, marker_duration, self.template_frame_data, frame_skip, 
                          orb_distance, scene_method, min_segment_length, timeline_end_frame, 
                          source_end_frame, marker_prefix, roi_rect, auto_fit, sync_playhead, 
                          use_clahe, max_analysis_res, audio_filter_enabled, audio_threshold_db, audio_sustained)

    def _run_batch(self, timeline, mode: str, threshold: float, marker_color: str, marker_duration: int, 
                   frame_skip: int, orb_distance: int, scene_method: str, min_segment_length: int, 
                   marker_prefix: str, roi_rect: Optional[Tuple[int, int, int, int]], auto_fit: bool, 
                   sync_playhead: bool, use_clahe: bool, max_analysis_res: int, audio_filter: bool, audio_thresh: float, audio_sustained: bool):
        clips = timeline.GetItemsInTrack("video", 1)
        if not clips:
            self.status_label.setText("❌ No clips found on V1 track.")
            return
        self.batch_clips = list(clips.values())
        self.batch_index = 0
        self.batch_total = len(self.batch_clips)
        self.batch_params = {
            'mode': mode, 'threshold': threshold, 'marker_color': marker_color,
            'marker_duration': marker_duration, 'frame_skip': frame_skip,
            'orb_distance': orb_distance, 'scene_method': scene_method,
            'min_segment_length': min_segment_length, 'marker_prefix': marker_prefix,
            'roi_rect': roi_rect, 'auto_fit': auto_fit, 'sync_playhead': sync_playhead,
            'use_clahe': use_clahe, 'max_analysis_res': max_analysis_res,
            'audio_filter': audio_filter, 'audio_thresh': audio_thresh, 'audio_sustained': audio_sustained
        }
        self._process_next_batch_clip()

    def _process_next_batch_clip(self):
        if not self.batch_clips or self.batch_index >= self.batch_total:
            self.status_label.setText(f"✅ Batch complete! Processed {self.batch_total} clips.")
            self._set_ui_running(False)
            self.batch_clips = None
            return
        clip = self.batch_clips[self.batch_index]
        clip_path = clip.GetFileProperty("FilePath")
        timeline_start = int(clip.GetStart())
        timeline_end = int(clip.GetEnd())
        try:
            source_start = int(clip.GetLeftOffset())
        except Exception:
            source_start = 0
        source_end = source_start + (timeline_end - timeline_start)
        p = self.batch_params
        self.status_label.setText(f"Batch: Clip {self.batch_index+1}/{self.batch_total} — {os.path.basename(clip_path)}")
        self._start_worker(clip_path, timeline_start, source_start, p['threshold'], p['marker_color'], 
                          p['mode'], p['marker_duration'], self.template_frame_data, p['frame_skip'], 
                          p['orb_distance'], p['scene_method'], p['min_segment_length'], timeline_end, 
                          source_end, p['marker_prefix'], p['roi_rect'], p['auto_fit'], p['sync_playhead'], 
                          p['use_clahe'], p['max_analysis_res'], p['audio_filter'], p['audio_thresh'], p['audio_sustained'])

    def _start_worker(self, clip_path: str, timeline_start: int, source_start: int, threshold: float, 
                      marker_color: str, mode: str, marker_duration: int, template_data: Optional[np.ndarray], 
                      frame_skip: int, orb_distance: int, scene_method: str, min_segment_length: int, 
                      timeline_end: Optional[int] = None, source_end: Optional[int] = None, 
                      marker_prefix: str = "CV", roi_rect: Optional[Tuple[int, int, int, int]] = None, 
                      auto_fit: bool = False, sync_playhead: bool = False, use_clahe: bool = False, 
                      max_analysis_res: int = 1080, audio_filter: bool = False, audio_thresh: float = -15.0, audio_sustained: bool = False):
        self.status_label.setText(f"Analyzing: {os.path.basename(clip_path)}...")
        self._set_ui_running(True)
        self.progress_bar.setValue(0)
        self._clear_feed()

        self.worker = CVWorker(clip_path, timeline_start, source_start, threshold, marker_color, mode, 
                              marker_duration, template_data, frame_skip, orb_distance, scene_method, 
                              min_segment_length, False, timeline_end, source_end, marker_prefix, 
                              roi_rect, self.target_color_hsv, auto_fit, sync_playhead, use_clahe, 
                              max_analysis_res, audio_filter, audio_thresh, audio_sustained)
        
        # Connect standard signals
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(lambda msg: self.status_label.setText(f"Status: {msg}"))
        self.worker.marker_found.connect(self._on_marker_found)
        self.worker.match_verified.connect(self._on_match_verified)
        self.worker.finished.connect(self.on_finished)
        
        # Connect thread-safe Resolve API signals
        self.worker.marker_to_add.connect(self._on_marker_to_add)
        self.worker.playhead_to_set.connect(self._on_playhead_to_set)
        self.worker.auto_fit_request.connect(self._on_auto_fit_request)
        
        self.worker.start()

    def _on_marker_to_add(self, data: dict):
        """Thread-safe slot to add markers to Resolve"""
        if not resolve: return
        try:
            timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
            if timeline:
                timeline.AddMarker(data['frame'], data['color'], data['name'], data['note'], data['duration'])
        except Exception as e:
            print(f"Error adding marker: {e}")

    def _on_playhead_to_set(self, frame: int):
        """Thread-safe slot to set playhead"""
        if not resolve: return
        try:
            timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
            if timeline:
                timeline.SetPlayhead(frame)
        except Exception as e:
            print(f"Error setting playhead: {e}")

    def _on_auto_fit_request(self, frames: list):
        """Thread-safe slot to auto-fit marker durations"""
        if not resolve: return
        try:
            timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
            markers = timeline.GetMarkers()
            if not markers: return
            for i, frame in enumerate(frames):
                frame_int = int(frame)
                data = get_marker_dict_value(markers, frame_int)
                if not data: continue
                if i < len(frames) - 1:
                    dur = int(frames[i+1]) - frame_int
                else:
                    dur = 1
                try:
                    timeline.DeleteMarker(frame_int, data['color'], data['name'])
                    timeline.AddMarker(frame_int, data['color'], data['name'], data.get('note', ''), dur)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error auto-fitting: {e}")

    def _on_match_verified(self, frame_with_box: Optional[np.ndarray]):
        if frame_with_box is not None:
            pix = cv_img_to_pixmap(frame_with_box, THUMB_W, THUMB_H)
            self.verify_thumb_label.setPixmap(pix)

    def cancel_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.status_label.setText("Cancelling...")
            self.batch_clips = None

    def _on_marker_found(self, marker_info: Dict):
        label = QLabel(f"📍 Frame {marker_info['frame']} | {marker_info['name']} | {marker_info['note']}")
        label.setStyleSheet("color: #9c9; font-size: 10px; padding: 1px;")
        self.feed_layout.insertWidget(self.feed_layout.count() - 1, label)
        self.marker_history.append(marker_info)

    def _clear_feed(self):
        while self.feed_layout.count() > 1:
            item = self.feed_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def clear_markers(self):
        if not resolve: return
        try:
            timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
            markers = timeline.GetMarkers()
        except Exception as e:
            self.status_label.setText(f"❌ Could not access markers: {e}")
            return
        prefix = (self.prefix_input.text() or "CV").upper()
        deleted_count = 0
        for frame, data in list(markers.items()):
            if data['name'].startswith(prefix):
                try:
                    timeline.DeleteMarker(int(frame), data['color'], data['name'])
                    deleted_count += 1
                except Exception:
                    pass
        self.status_label.setText(f"🗑️ Cleared {deleted_count} {prefix} markers.")
        self._clear_feed()
        self.marker_history.clear()
        self.update_marker_stats()

    def undo_last_analysis(self):
        if not self.marker_history:
            self.status_label.setText("Nothing to undo.")
            return
        if not resolve: return
        try:
            timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
        except Exception as e:
            self.status_label.setText(f"❌ Could not access timeline: {e}")
            return
        count = 0
        current_color = self.color_dropdown.currentText()
        while self.marker_history:
            marker = self.marker_history.pop()
            try:
                timeline.DeleteMarker(int(marker['frame']), current_color, marker['name'])
                count += 1
            except Exception:
                pass
        self.status_label.setText(f"↩️ Undone: removed {count} markers.")
        self._clear_feed()
        self.update_marker_stats()

    def cleanup_markers(self):
        """Merges nearby markers of the same type and pads their duration."""
        if not resolve: return
        try:
            timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
            markers = timeline.GetMarkers()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not access markers:\n{e}")
            return
            
        prefix = (self.prefix_input.text() or "CV").upper()
        merge_gap = self.merge_gap_spin.value()
        min_duration = self.min_dur_spin.value()
        
        cv_markers = {f: d for f, d in markers.items() if d['name'].startswith(prefix)}
        if not cv_markers:
            QMessageBox.information(self, "Cleanup", "No markers found to cleanup.")
            return
            
        sorted_frames = sorted([int(f) for f in cv_markers.keys()])
        
        groups = []
        current_group = []
        
        for frame in sorted_frames:
            data = cv_markers[str(frame)] if str(frame) in cv_markers else cv_markers[frame]
            marker_type = data['name'].split()[1] if len(data['name'].split()) > 1 else "General"
            duration = int(data.get('duration', 1))
            if duration <= 0: duration = 1
            
            if not current_group:
                current_group = [{'frame': frame, 'data': data, 'duration': duration, 'type': marker_type}]
            else:
                last = current_group[-1]
                last_end = last['frame'] + last['duration']
                gap = frame - last_end
                
                if last['type'] == marker_type and gap <= merge_gap:
                    current_group.append({'frame': frame, 'data': data, 'duration': duration, 'type': marker_type})
                else:
                    groups.append(current_group)
                    current_group = [{'frame': frame, 'data': data, 'duration': duration, 'type': marker_type}]
                    
        if current_group:
            groups.append(current_group)
            
        new_markers = []
        for group in groups:
            start_frame = group[0]['frame']
            end_frame = max([m['frame'] + m['duration'] for m in group])
            
            if end_frame - start_frame < min_duration:
                end_frame = start_frame + min_duration
                
            new_duration = end_frame - start_frame
            first_data = group[0]['data']
            new_markers.append({
                'frame': start_frame,
                'color': first_data['color'],
                'name': first_data['name'],
                'note': first_data.get('note', ''),
                'duration': new_duration
            })
            
        for frame in sorted_frames:
            data = cv_markers[str(frame)] if str(frame) in cv_markers else cv_markers[frame]
            try:
                timeline.DeleteMarker(frame, data['color'], data['name'])
            except Exception:
                pass
                
        for m in new_markers:
            try:
                timeline.AddMarker(m['frame'], m['color'], m['name'], m['note'], m['duration'])
            except Exception:
                pass
                
        self.status_label.setText(f"🧹 Cleaned up {len(sorted_frames)} markers into {len(new_markers)} clean markers.")
        self.update_marker_stats()

    def export_markers_csv(self):
        if not resolve: return
        try:
            timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
            markers = timeline.GetMarkers()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not access markers:\n{e}")
            return
        prefix = (self.prefix_input.text() or "CV").upper()
        cv_markers = {f: d for f, d in markers.items() if d['name'].startswith(prefix)}
        if not cv_markers:
            QMessageBox.information(self, "Export", "No markers found to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Markers", "", "CSV Files (*.csv)")
        if not file_path: return
        try:
            fps = float(timeline.GetSetting("timelineFrameRate") or 24)
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Frame", "Timecode", "Name", "Color", "Note", "Duration"])
                for frame in sorted(cv_markers.keys(), key=lambda x: int(x)):
                    frame_int = int(frame)
                    data = cv_markers[frame]
                    tc = str(timedelta(seconds=frame_int / fps))
                    writer.writerow([frame_int, tc, data['name'], data['color'], data.get('note', ''), data.get('duration', 1)])
            self.status_label.setText(f"📥 Exported {len(cv_markers)} markers to CSV.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{e}")

    def export_markers_srt(self):
        if not resolve: return
        try:
            timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
            markers = timeline.GetMarkers()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not access markers:\n{e}")
            return
        prefix = (self.prefix_input.text() or "CV").upper()
        cv_markers = {f: d for f, d in markers.items() if d['name'].startswith(prefix)}
        if not cv_markers:
            QMessageBox.information(self, "Export", "No markers found to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export SRT", "", "Subtitle Files (*.srt)")
        if not file_path: return
        try:
            fps = float(timeline.GetSetting("timelineFrameRate") or 24)
            def format_srt_time(seconds: float) -> str:
                hrs = int(seconds // 3600)
                mins = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int((seconds - int(seconds)) * 1000)
                return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"
            with open(file_path, 'w', encoding='utf-8') as f:
                for i, frame in enumerate(sorted(cv_markers.keys(), key=lambda x: int(x))):
                    frame_int = int(frame)
                    data = cv_markers[frame]
                    start_sec = frame_int / fps
                    end_sec = (frame_int + data.get('duration', 1)) / fps
                    f.write(f"{i+1}\n")
                    f.write(f"{format_srt_time(start_sec)} --> {format_srt_time(end_sec)}\n")
                    f.write(f"{data['name']}: {data.get('note', '')}\n\n")
            self.status_label.setText(f"📥 Exported {len(cv_markers)} markers to SRT.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export SRT:\n{e}")

    def goto_prev_marker(self):
        if not resolve: return
        try:
            timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
            markers = timeline.GetMarkers()
        except Exception:
            return
        prefix = (self.prefix_input.text() or "CV").upper()
        frames = sorted([int(f) for f, d in markers.items() if d['name'].startswith(prefix)])
        if not frames: return
        playhead = timeline.GetPlayhead()
        prev_frames = [f for f in frames if f < playhead]
        if prev_frames:
            timeline.SetPlayhead(prev_frames[-1])
            self.status_label.setText(f"Jumped to frame {prev_frames[-1]}")
        else:
            self.status_label.setText("No previous marker found.")

    def goto_next_marker(self):
        if not resolve: return
        try:
            timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
            markers = timeline.GetMarkers()
        except Exception:
            return
        prefix = (self.prefix_input.text() or "CV").upper()
        frames = sorted([int(f) for f, d in markers.items() if d['name'].startswith(prefix)])
        if not frames: return
        playhead = timeline.GetPlayhead()
        next_frames = [f for f in frames if f > playhead]
        if next_frames:
            timeline.SetPlayhead(next_frames[0])
            self.status_label.setText(f"Jumped to frame {next_frames[0]}")
        else:
            self.status_label.setText("No next marker found.")

    def update_marker_stats(self):
        if not resolve:
            self.stats_label.setText("Resolve API not found.")
            return
        try:
            timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
            markers = timeline.GetMarkers()
        except Exception as e:
            self.stats_label.setText(f"Error: {e}")
            return
        prefix = (self.prefix_input.text() or "CV").upper()
        cv_markers = {f: d for f, d in markers.items() if d['name'].startswith(prefix)}
        if not cv_markers:
            self.stats_label.setText("No markers found with current prefix.")
            return
        type_counts = {}
        for data in cv_markers.values():
            parts = data['name'].split()
            if len(parts) >= 2:
                mtype = parts[1]
                type_counts[mtype] = type_counts.get(mtype, 0) + 1
        stats_lines = [f"Total Markers ({prefix}): {len(cv_markers)}"]
        for t, c in sorted(type_counts.items()):
            stats_lines.append(f"  • {t}: {c}")
        frames = sorted([int(f) for f in cv_markers.keys()])
        if len(frames) > 1:
            avg_gap = (frames[-1] - frames[0]) / (len(frames) - 1)
            stats_lines.append(f"Average gap: {avg_gap:.1f} frames")
        self.stats_label.setText("\n".join(stats_lines))

    def update_progress(self, val: int):
        self.progress_bar.setValue(val)

    def on_finished(self, message: str):
        self.status_label.setText(f"✅ {message}")
        self._set_ui_running(False)
        self.update_marker_stats()
        if self.batch_clips is not None and self.batch_index < self.batch_total - 1:
            self.batch_index += 1
            QTimer.singleShot(500, self._process_next_batch_clip)
        elif self.batch_clips is not None:
            self.batch_clips = None
            self.status_label.setText(f"✅ Batch complete! Processed {self.batch_total} clips.")

    def _save_settings(self):
        settings = {
            'mode': self._get_mode(),
            'threshold': self.thresh_slider.value(),
            'orb_distance': self.orb_dist_slider.value(),
            'marker_color': self.color_dropdown.currentText(),
            'frame_skip': self.skip_spin.value(),
            'marker_duration': self.dur_spin.value(),
            'min_segment_length': self.min_seg_spin.value(),
            'scene_method': self.scene_method_combo.currentText(),
            'marker_prefix': self.prefix_input.text(),
            'auto_fit': self.auto_fit_check.isChecked(),
            'sync_playhead': self.sync_playhead_check.isChecked(),
            'on_top': self.on_top_check.isChecked(),
            'use_clahe': self.clahe_check.isChecked(),
            'max_analysis_res': self.res_combo.currentText(),
            'audio_filter': self.audio_check.isChecked(),
            'audio_thresh': self.audio_thresh_slider.value(),
            'audio_sustained': self.audio_sustained_check.isChecked(),
            'merge_gap': self.merge_gap_spin.value(),
            'min_duration': self.min_dur_spin.value()
        }
        try:
            with open(SETTINGS_PATH, 'w') as f:
                json.dump(settings, f)
        except Exception:
            pass

    def _load_settings(self):
        if not os.path.exists(SETTINGS_PATH):
            return
        try:
            with open(SETTINGS_PATH, 'r') as f:
                settings = json.load(f)
            mode = settings.get('mode', 'Scene Match')
            mode_radios = {
                "Scene Match": self.radio_scene, "Segment Match": self.radio_segment,
                "Recurring Match": self.radio_recur, "Motion Detection": self.radio_motion,
                "HUD Motion": self.radio_hud, "UI Template Match": self.radio_ui,
                "Prolonged UI Match": self.radio_prolonged_ui, "Health Bar Tracker": self.radio_health,
                "Loading Screen": self.radio_loading, "Fade Detection": self.radio_fade,
                "Sudden Freeze": self.radio_freeze, "Camera Flick": self.radio_flick,
                "Face Detection": self.radio_face, "Focus Detection": self.radio_focus,
                "Color Match": self.radio_color, "Flash Detection": self.radio_flash
            }
            if mode in mode_radios:
                mode_radios[mode].setChecked(True)
            self.thresh_slider.setValue(settings.get('threshold', 50))
            self.orb_dist_slider.setValue(settings.get('orb_distance', 50))
            self.color_dropdown.setCurrentText(settings.get('marker_color', 'Green'))
            self.skip_spin.setValue(settings.get('frame_skip', 1))
            self.dur_spin.setValue(settings.get('marker_duration', 1))
            self.min_seg_spin.setValue(settings.get('min_segment_length', 5))
            self.scene_method_combo.setCurrentText(settings.get('scene_method', 'Histogram (HSV)'))
            self.prefix_input.setText(settings.get('marker_prefix', 'CV'))
            self.auto_fit_check.setChecked(settings.get('auto_fit', False))
            self.sync_playhead_check.setChecked(settings.get('sync_playhead', False))
            self.on_top_check.setChecked(settings.get('on_top', False))
            self.clahe_check.setChecked(settings.get('use_clahe', False))
            self.res_combo.setCurrentText(settings.get('max_analysis_res', '1080p'))
            self.audio_check.setChecked(settings.get('audio_filter', False))
            self.audio_thresh_slider.setValue(settings.get('audio_thresh', -15))
            self.audio_sustained_check.setChecked(settings.get('audio_sustained', False))
            self.merge_gap_spin.setValue(settings.get('merge_gap', 15))
            self.min_dur_spin.setValue(settings.get('min_duration', 10))
        except Exception as e:
            print(f"Warning: Could not load settings: {e}")

    def closeEvent(self, event):
        self._save_settings()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            if not self.worker.wait(2000):
                self.worker.terminate()
                self.worker.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = CVMarkerExtension(app)
    window.show()