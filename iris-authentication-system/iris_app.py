"""
Iris Authentication System — GUI Application
Real-time webcam feed with iris detection, registration and authentication.
"""

import tkinter as tk
from tkinter import font as tkfont
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.image_auth_system import ImageAuthSystem
from core.iris_detector import IrisDetector

# ─────────────────────────────────────────────
#  Color palette  (dark cyberpunk theme)
# ─────────────────────────────────────────────
BG          = "#0a0a0f"
PANEL       = "#12121a"
CARD        = "#1a1a28"
ACCENT      = "#00e5ff"      # cyan
ACCENT2     = "#7c4dff"      # purple
SUCCESS     = "#00e676"
DANGER      = "#ff1744"
WARNING     = "#ffc400"
TEXT        = "#e8eaf6"
TEXT_DIM    = "#616190"
BORDER      = "#2a2a45"

FONT_TITLE  = ("Courier New", 22, "bold")
FONT_HEAD   = ("Courier New", 13, "bold")
FONT_BODY   = ("Courier New", 11)
FONT_SMALL  = ("Courier New", 9)
FONT_STATUS = ("Courier New", 14, "bold")

SAMPLES_NEEDED = 30   # frames to collect during registration


class ScanlineCanvas(tk.Canvas):
    """Camera feed canvas with HUD overlay."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._img_ref = None
        self._iris_box = None
        self._scan_y = 0
        self._scanning = False
        self._after_id = None

    def update_frame(self, bgr_frame, iris_crop=None, iris_box=None):
        h = self.winfo_height() or 380
        w = self.winfo_width() or 560

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (w, h))

        # Draw iris bounding box if detected
        if iris_box:
            x1, y1, x2, y2 = iris_box
            # scale box to canvas size
            fh, fw = bgr_frame.shape[:2]
            sx, sy = w / fw, h / fh
            rx1, ry1 = int(x1 * sx), int(y1 * sy)
            rx2, ry2 = int(x2 * sx), int(y2 * sy)
            pad = 20
            cv2.rectangle(rgb, (rx1 - pad, ry1 - pad),
                          (rx2 + pad, ry2 + pad), (0, 229, 255), 2)
            # corner accents
            L = 12
            c = (0, 229, 255)
            for px, py, dx, dy in [
                (rx1-pad, ry1-pad, 1, 1), (rx2+pad, ry1-pad, -1, 1),
                (rx1-pad, ry2+pad, 1, -1), (rx2+pad, ry2+pad, -1, -1)
            ]:
                cv2.line(rgb, (px, py), (px + dx*L, py), c, 3)
                cv2.line(rgb, (px, py), (px, py + dy*L), c, 3)

        img = Image.fromarray(rgb)
        self._img_ref = ImageTk.PhotoImage(img)
        self.delete("frame")
        self.create_image(0, 0, anchor="nw", image=self._img_ref, tags="frame")

        if self._scanning:
            self._draw_scanline()

    def _draw_scanline(self):
        h = self.winfo_height() or 380
        w = self.winfo_width() or 560
        self.delete("scanline")
        self.create_line(0, self._scan_y, w, self._scan_y,
                         fill=ACCENT, width=2, tags="scanline")
        self._scan_y = (self._scan_y + 4) % h

    def start_scan(self):
        self._scanning = True

    def stop_scan(self):
        self._scanning = False
        self.delete("scanline")


class IrisAuthApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("IRIS AUTH SYSTEM v2.0")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.auth = ImageAuthSystem()
        self.iris_detector = IrisDetector()

        self.cap = None
        self._current_frame = None
        self._current_iris = None
        self._current_box = None

        self.mode = "idle"
        self._reg_samples = []
        self._reg_name = ""
        self._last_status = ""
        self._auth_cooldown = 0
        self._fps = 0
        self._prev_time = time.time()

        # Liveness: blink detection
        self._blink_confirmed = False   # True after a blink detected
        self._eye_was_closed = False    # previous frame eye state
        self._liveness_timer = 0        # countdown to reset liveness

        self._build_ui()
        self._open_camera()
        self.after(100, self._camera_loop)

    # ═══════════════════════════════════════
    #  UI CONSTRUCTION
    # ═══════════════════════════════════════

    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(16, 0))

        tk.Label(hdr, text="◈ IRIS AUTHENTICATION SYSTEM",
                 font=FONT_TITLE, fg=ACCENT, bg=BG).pack(side="left")
        self.lbl_clock = tk.Label(hdr, text="", font=FONT_SMALL,
                                  fg=TEXT_DIM, bg=BG)
        self.lbl_clock.pack(side="right")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20, pady=8)

        # ── Main row ──
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", padx=20, pady=0)

        # Left: camera
        cam_card = tk.Frame(main, bg=CARD, bd=0,
                            highlightthickness=1,
                            highlightbackground=BORDER)
        cam_card.pack(side="left", fill="both")

        self.canvas = ScanlineCanvas(cam_card, width=560, height=380,
                                     bg="#000010", highlightthickness=0)
        self.canvas.pack(padx=2, pady=2)

        # iris crop preview strip
        strip = tk.Frame(cam_card, bg=CARD)
        strip.pack(fill="x", padx=2, pady=(0, 4))
        tk.Label(strip, text="IRIS CROP", font=FONT_SMALL,
                 fg=TEXT_DIM, bg=CARD).pack(side="left", padx=8)
        self.iris_preview = tk.Label(strip, bg="#000010",
                                     width=80, height=40)
        self.iris_preview.pack(side="left", padx=4)
        self.lbl_iris_status = tk.Label(strip, text="● NO IRIS DETECTED",
                                        font=FONT_SMALL, fg=DANGER, bg=CARD)
        self.lbl_iris_status.pack(side="left", padx=8)

        # Right: control panel
        right = tk.Frame(main, bg=BG, width=280)
        right.pack(side="left", fill="both", padx=(14, 0))
        right.pack_propagate(False)

        # Status card
        self.status_card = tk.Frame(right, bg=CARD,
                                    highlightthickness=1,
                                    highlightbackground=BORDER)
        self.status_card.pack(fill="x", pady=(0, 10))

        tk.Label(self.status_card, text="SYSTEM STATUS",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack(anchor="w", padx=12, pady=(10, 2))

        self.lbl_status = tk.Label(self.status_card,
                                   text="READY", font=FONT_STATUS,
                                   fg=ACCENT, bg=CARD, wraplength=240,
                                   justify="left")
        self.lbl_status.pack(anchor="w", padx=12, pady=(0, 4))

        self.lbl_confidence = tk.Label(self.status_card, text="",
                                       font=FONT_SMALL, fg=TEXT_DIM, bg=CARD)
        self.lbl_confidence.pack(anchor="w", padx=12, pady=(0, 10))

        # Progress bar (for registration)
        self.progress_frame = tk.Frame(right, bg=BG)
        self.progress_frame.pack(fill="x", pady=(0, 8))
        self.progress_label = tk.Label(self.progress_frame, text="",
                                       font=FONT_SMALL, fg=TEXT_DIM, bg=BG)
        self.progress_label.pack(anchor="w")
        self.progress_bar_bg = tk.Frame(self.progress_frame, bg=BORDER,
                                        height=6, width=278)
        self.progress_bar_bg.pack(fill="x")
        self.progress_bar = tk.Frame(self.progress_bar_bg, bg=ACCENT,
                                     height=6, width=0)
        self.progress_bar.place(x=0, y=0)

        # ── Register section ──
        reg_card = tk.Frame(right, bg=CARD,
                            highlightthickness=1,
                            highlightbackground=BORDER)
        reg_card.pack(fill="x", pady=(0, 10))

        tk.Label(reg_card, text="REGISTER NEW USER",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack(anchor="w", padx=12, pady=(10, 4))

        self.entry_name = tk.Entry(reg_card,
                                   font=FONT_BODY, fg=TEXT, bg="#0d0d1a",
                                   insertbackground=ACCENT,
                                   relief="flat", bd=0,
                                   highlightthickness=1,
                                   highlightbackground=BORDER,
                                   highlightcolor=ACCENT)
        self.entry_name.pack(fill="x", padx=12, ipady=6)
        self.entry_name.insert(0, "Enter name...")
        self.entry_name.bind("<FocusIn>",
                             lambda e: self.entry_name.delete(0, "end")
                             if self.entry_name.get() == "Enter name..." else None)

        self.btn_register = self._btn(reg_card, "◉  START REGISTRATION",
                                      self._start_registration, ACCENT2)
        self.btn_register.pack(fill="x", padx=12, pady=(8, 12))

        # ── Auth section ──
        auth_card = tk.Frame(right, bg=CARD,
                             highlightthickness=1,
                             highlightbackground=BORDER)
        auth_card.pack(fill="x", pady=(0, 10))

        tk.Label(auth_card, text="AUTHENTICATE",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack(anchor="w", padx=12, pady=(10, 4))

        self.btn_auth = self._btn(auth_card, "⬡  SCAN IRIS",
                                  self._trigger_auth, ACCENT)
        self.btn_auth.pack(fill="x", padx=12, pady=(0, 12))

        # ── Users list ──
        users_card = tk.Frame(right, bg=CARD,
                              highlightthickness=1,
                              highlightbackground=BORDER)
        users_card.pack(fill="both", expand=True)

        tk.Label(users_card, text="REGISTERED USERS",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack(anchor="w", padx=12, pady=(10, 4))

        self.users_frame = tk.Frame(users_card, bg=CARD)
        self.users_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self._refresh_users()

        # ── Footer ──
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20, pady=8)
        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(footer, text="MediaPipe · LBP Features · SVM Classifier · IsolationForest Anti-Spoof",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=BG).pack(side="left")
        self.lbl_fps = tk.Label(footer, text="FPS: —", font=FONT_SMALL,
                                fg=TEXT_DIM, bg=BG)
        self.lbl_fps.pack(side="right")

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         font=FONT_HEAD, fg=color, bg=PANEL,
                         activeforeground=BG, activebackground=color,
                         relief="flat", bd=0, cursor="hand2",
                         pady=8,
                         highlightthickness=1,
                         highlightbackground=color)

    # ═══════════════════════════════════════
    #  CAMERA THREAD
    # ═══════════════════════════════════════

    def _open_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._set_status("CAMERA NOT FOUND", DANGER)

    def _camera_loop(self):
        """Single loop: read frame, detect iris, update GUI. Runs on main thread via after()."""
        if self.cap is None or not self.cap.isOpened():
            self.after(33, self._camera_loop)
            return

        ret, frame = self.cap.read()
        if ret:
            iris_crop = self.iris_detector.extract_iris(frame)
            iris_box = self._get_iris_box(frame)

            self._current_frame = frame
            self._current_iris = iris_crop
            self._current_box = iris_box

            # FPS
            now = time.time()
            self._fps = 1.0 / max(now - self._prev_time, 1e-6)
            self._prev_time = now

            # Auto-collect during registration
            if self.mode == "register" and iris_crop is not None:
                features = self.auth.image_processor.extract_features(iris_crop)
                if features is not None:
                    self._reg_samples.append(features)

            # Liveness: detect blink (eye open -> closed -> open)
            elif self.mode == "idle" and self.auth.is_trained and self._auth_cooldown <= 0:
                eye_open = iris_crop is not None and iris_crop.size > 0
                if self._eye_was_closed and eye_open:
                    self._blink_confirmed = True
                    self._liveness_timer = 3.0  # liveness valid for 3 seconds
                self._eye_was_closed = not eye_open

                # Decay liveness timer
                if self._liveness_timer > 0:
                    self._liveness_timer -= 0.033
                else:
                    self._blink_confirmed = False

                if not self._blink_confirmed:
                    self._set_status("BLINK TO AUTHENTICATE", ACCENT)
                    self.lbl_confidence.config(text="liveness check", fg=TEXT_DIM)
                elif iris_crop is not None and self._auth_cooldown <= 0:
                    features = self.auth.image_processor.extract_features(iris_crop)
                    if features is not None:
                        decision, confidence, username = self.auth.authenticate(features)
                        if confidence >= 0.60 or decision == "not_trained":
                            self._show_auth_result(decision, confidence, username)
                            self._blink_confirmed = False  # require new blink
                        else:
                            self._set_status("ADJUST POSITION", WARNING)
                            self.lbl_confidence.config(text=f"CONFIDENCE: {confidence*100:.1f}%", fg=WARNING)
                        self._auth_cooldown = 1.5

            # Update canvas
            self.canvas.update_frame(frame, iris_crop, iris_box)

            # Iris status indicator
            if iris_crop is not None and iris_crop.size > 0:
                self.lbl_iris_status.config(text="● IRIS DETECTED", fg=SUCCESS)
                try:
                    prev = cv2.resize(iris_crop, (80, 40))
                    prev_rgb = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
                    img = ImageTk.PhotoImage(Image.fromarray(prev_rgb))
                    self.iris_preview.config(image=img)
                    self.iris_preview.image = img
                except Exception:
                    pass
            else:
                self.lbl_iris_status.config(text="● NO IRIS DETECTED", fg=DANGER)

            self.lbl_fps.config(text=f"FPS: {self._fps:.0f}")

        # Clock
        self.lbl_clock.config(text=time.strftime("%H:%M:%S"))

        # Registration progress
        if self.mode == "register":
            n = len(self._reg_samples)
            self.progress_label.config(
                text=f"COLLECTING SAMPLES: {n}/{SAMPLES_NEEDED}")
            bar_w = int(278 * min(n, SAMPLES_NEEDED) / SAMPLES_NEEDED)
            self.progress_bar.config(width=bar_w)
            self.canvas.start_scan()
            if n >= SAMPLES_NEEDED:
                self._finish_registration()

        # Auth cooldown
        if self._auth_cooldown > 0:
            self._auth_cooldown -= 0.033
            self.btn_auth.config(state="disabled")
        else:
            self.btn_auth.config(state="normal")

        self.after(33, self._camera_loop)

    def _get_iris_box(self, frame):
        """Return bounding box (x1,y1,x2,y2) of iris or None."""
        try:
            if self.iris_detector.face_mesh is None:
                return None
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.iris_detector.face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                return None
            h, w, _ = frame.shape
            pts = [(int(lm.x * w), int(lm.y * h))
                   for lm in result.multi_face_landmarks[0].landmark[468:472]]
            xs, ys = zip(*pts)
            return min(xs), min(ys), max(xs), max(ys)
        except Exception:
            return None

    # ═══════════════════════════════════════
    #  ACTIONS
    # ═══════════════════════════════════════

    def _start_registration(self):
        name = self.entry_name.get().strip()
        if not name or name == "Enter name...":
            self._set_status("ENTER A NAME FIRST", WARNING)
            return

        self._reg_name = name
        self._reg_samples = []
        self.mode = "register"
        self._set_status(f"SCANNING…\nLOOK AT CAMERA", ACCENT2)
        self.btn_register.config(state="disabled")
        self.canvas.start_scan()

    def _finish_registration(self):
        self.mode = "idle"
        self.canvas.stop_scan()

        result = self.auth.register_user(self._reg_name, self._reg_samples)

        self.progress_label.config(text="")
        self.progress_bar.config(width=0)
        self.btn_register.config(state="normal")

        if result == "registered":
            self._set_status(f"✓ REGISTERED\n{self._reg_name.upper()}", SUCCESS)
            self._refresh_users()
        else:
            self._set_status("REGISTRATION FAILED", DANGER)

    def _trigger_auth(self):
        iris = self._current_iris

        if iris is None or iris.size == 0:
            self._set_status("NO IRIS DETECTED\nADJUST CAMERA", WARNING)
            return

        features = self.auth.image_processor.extract_features(iris)
        if features is None:
            self._set_status("FEATURE EXTRACTION\nFAILED", DANGER)
            return

        decision, confidence, username = self.auth.authenticate(features)
        self._show_auth_result(decision, confidence, username)
        self._auth_cooldown = 4.0   # hold result for 4 seconds

    def _show_auth_result(self, decision, confidence, username):
        if decision == "granted":
            self._set_status(f"✓ ACCESS GRANTED\n{username.upper()}", SUCCESS)
            self.lbl_confidence.config(
                text=f"CONFIDENCE: {confidence*100:.1f}%", fg=SUCCESS)
        elif decision == "denied_spoof":
            self._set_status("✗ SPOOFING DETECTED\nACCESS DENIED", DANGER)
            self.lbl_confidence.config(text="", fg=TEXT_DIM)
        elif decision == "denied_low_confidence":
            self._set_status("✗ NOT RECOGNIZED\nACCESS DENIED", DANGER)
            self.lbl_confidence.config(
                text=f"CONFIDENCE: {confidence*100:.1f}%", fg=DANGER)
        elif decision == "not_trained":
            self._set_status("REGISTER AT LEAST\n2 USERS FIRST", WARNING)
            self.lbl_confidence.config(text="", fg=TEXT_DIM)

    def _set_status(self, text, color=TEXT):
        self.lbl_status.config(text=text, fg=color)

    def _refresh_users(self):
        for w in self.users_frame.winfo_children():
            w.destroy()

        users = self.auth.get_registered_users()
        if not users:
            tk.Label(self.users_frame, text="No users registered",
                     font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack(anchor="w")
            return

        for name in users:
            row = tk.Frame(self.users_frame, bg=CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"◆ {name}", font=FONT_SMALL,
                     fg=TEXT, bg=CARD).pack(side="left")
            tk.Button(row, text="✕",
                      command=lambda n=name: self._delete_user(n),
                      font=FONT_SMALL, fg=DANGER, bg=CARD,
                      relief="flat", bd=0, cursor="hand2",
                      activeforeground=BG, activebackground=DANGER
                      ).pack(side="right")

    def _delete_user(self, name):
        self.auth.delete_user(name)
        self._set_status(f"DELETED: {name.upper()}", WARNING)
        self._refresh_users()

    # ═══════════════════════════════════════
    #  CLEANUP
    # ═══════════════════════════════════════

    def on_close(self):
        self._camera_running = False
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = IrisAuthApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()