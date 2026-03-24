"""
🎯 Focus Monitor - AI Distraction Alert Tool  (Side-Panel Edition)
===================================================================
Layout:  [ Camera Feed (640x480) | Video Panel (426x480) ]

Behaviour:
  • FOCUSED    → side panel shows the video frozen at frame 0
  • DISTRACTED → side panel plays the video; resets to start when you re-focus

Requirements:
    pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import time
import sys
import math
import os
import urllib.request
import tempfile
import pygame
from moviepy import VideoFileClip

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
PITCH_DOWN_THRESHOLD  = 8      # degrees – down (was 15)
PITCH_UP_THRESHOLD    = -10    # degrees – up  (was -20)
YAW_THRESHOLD         = 20     # degrees – side (was 40)
DISTRACTION_DELAY_SEC = 1.5    # grace period (was 2.0s)
CAMERA_INDEX          = 0
CAM_W, CAM_H          = 640, 480
PANEL_W               = 426    # side-panel width  (keeps 16:9 with 480 h)
VIDEO_PATH            = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IAS_ki_taiyari_chhod_do_forget_your_IAS_dream_IAS_kase_batae_hai_ias_exam_viral_trend_144P.mp4")
MODEL_PATH            = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
# ──────────────────────────────────────────────


# 3-D reference points for solvePnP fallback
FACE_3D_PTS = np.array([
    [0.0,    0.0,    0.0],
    [0.0,   -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0,  170.0, -135.0],
    [-150.0,-150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)
LM_INDICES = [1, 152, 263, 33, 287, 57]


def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("\033[1;33m[INFO] Downloading face landmark model (~6 MB) …\033[0m")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("\033[1;32m[INFO] Model downloaded.\033[0m")


AUDIO_WAV = os.path.join(tempfile.gettempdir(), "focus_monitor_audio.wav")


def extract_audio():
    """Extract MP4 audio to a temp WAV once; pygame can't load MP4 directly."""
    if os.path.exists(AUDIO_WAV):
        return
    print("\033[1;33m[INFO] Extracting audio track (one-time) …\033[0m")
    clip = VideoFileClip(VIDEO_PATH)
    clip.audio.write_audiofile(AUDIO_WAV, logger=None)
    clip.close()
    print("\033[1;32m[INFO] Audio ready.\033[0m")


def rotation_matrix_to_euler(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = math.degrees(math.atan2( R[2, 1], R[2, 2]))
        roll  = math.degrees(math.atan2( R[1, 0], R[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = 0.0
        roll  = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
    return pitch, yaw, roll


def get_angles_from_matrix(tm):
    R = np.array(tm.data, dtype=np.float64).reshape(4, 4)[:3, :3]
    return rotation_matrix_to_euler(R)


def get_angles_from_landmarks(lms, w, h):
    img_pts = np.array(
        [[lms[i].x * w, lms[i].y * h] for i in LM_INDICES], dtype=np.float64)
    focal = w
    cam_mat = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], np.float64)
    ok, rvec, _ = cv2.solvePnP(FACE_3D_PTS, img_pts, cam_mat,
                                np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None
    rmat, _ = cv2.Rodrigues(rvec)
    return rotation_matrix_to_euler(rmat)


def is_distracted(pitch, yaw):
    if pitch is None:
        return False
    return pitch > PITCH_DOWN_THRESHOLD or pitch < PITCH_UP_THRESHOLD or abs(yaw) > YAW_THRESHOLD


# ── Camera HUD ─────────────────────────────────
def draw_cam_hud(frame, pitch, yaw, roll, distracted, dist_secs):
    color  = (80, 210, 80) if not distracted else (50, 60, 220)
    status = "FOCUSED ✓" if not distracted else f"DISTRACTED! {dist_secs:.1f}s"
    cv2.rectangle(frame, (0, 0), (CAM_W, 110), (10, 10, 10), -1)
    cv2.putText(frame, status, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if pitch is not None:
        for i, (lbl, v) in enumerate([("Pitch", pitch), ("Yaw", yaw), ("Roll", roll)], 1):
            cv2.putText(frame, f"{lbl}: {v:+.1f}°", (12, 32 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (185, 185, 185), 1)
    else:
        cv2.putText(frame, "No face detected", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)


# ── Side‑panel builder ─────────────────────────
def build_panel(video_frame, distracted):
    """Scale video frame to PANEL_W × CAM_H and add overlay decorations."""
    panel = cv2.resize(video_frame, (PANEL_W, CAM_H))

    if not distracted:
        # Dim the panel and show a PAUSED badge
        panel = (panel * 0.30).astype(np.uint8)
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (PANEL_W, CAM_H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, panel, 0.65, 0, panel)
        # Pause icon  ▌▌
        cx, cy = PANEL_W // 2, CAM_H // 2
        bar_h, bar_w, gap = 50, 14, 10
        cv2.rectangle(panel, (cx - gap - bar_w, cy - bar_h),
                      (cx - gap,           cy + bar_h), (200, 200, 200), -1)
        cv2.rectangle(panel, (cx + gap,           cy - bar_h),
                      (cx + gap + bar_w,   cy + bar_h), (200, 200, 200), -1)
        cv2.putText(panel, "Stay Focused!", (PANEL_W//2 - 80, CAM_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 140, 140), 1)
    else:
        # Red border pulse
        cv2.rectangle(panel, (0, 0), (PANEL_W-1, CAM_H-1), (30, 30, 220), 5)
        cv2.putText(panel, "⚠ GET BACK TO WORK!", (10, CAM_H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 220), 2)
    return panel


# ── Divider ────────────────────────────────────
def make_divider(h, distracted):
    color = (30, 30, 220) if distracted else (60, 60, 60)
    div = np.zeros((h, 4, 3), dtype=np.uint8)
    div[:] = color
    return div


def main():
    download_model()
    extract_audio()

    print("\033[1;36m")
    print("╔══════════════════════════════════════════════════════╗")
    print("║         🎯  FOCUS MONITOR  (Side-Panel Mode)         ║")
    print("╚══════════════════════════════════════════════════════╝\033[0m")
    print(f"  Pitch threshold : >{PITCH_DOWN_THRESHOLD}° (down), <{PITCH_UP_THRESHOLD}° (up)")
    print(f"  Yaw threshold   : ±{YAW_THRESHOLD}°")
    print(f"  Grace period    : {DISTRACTION_DELAY_SEC}s")
    print("  Press  Q  to quit.\n")

    # ── Audio (pygame mixer) ──────────────────
    pygame.mixer.init()
    pygame.mixer.music.load(AUDIO_WAV)   # WAV extracted from the MP4
    audio_playing = False

    # ── MediaPipe landmarker ───────────────────
    base_opts = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    # ── Camera ────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("\033[1;31m[ERROR] Cannot open camera.\033[0m")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    # ── Video ─────────────────────────────────
    vid = cv2.VideoCapture(VIDEO_PATH)
    if not vid.isOpened():
        print(f"\033[1;31m[ERROR] Cannot open video: {VIDEO_PATH}\033[0m")
        sys.exit(1)
    vid_fps      = vid.get(cv2.CAP_PROP_FPS) or 30
    frame_delay  = 1.0 / vid_fps       # target seconds per video frame
    last_vid_t   = time.time()
    ok_v, first_frame = vid.read()
    if not ok_v:
        print("\033[1;31m[ERROR] Could not read video.\033[0m")
        sys.exit(1)
    current_vid_frame = first_frame.copy()

    distraction_start = None
    was_distracted    = False           # track transition for terminal logs
    audio_playing     = False

    try:
        while True:
            ok, cam_frame = cap.read()
            if not ok:
                continue

            cam_frame = cv2.flip(cam_frame, 1)
            h, w      = cam_frame.shape[:2]
            rgb       = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            pitch = yaw = roll = None
            if result.face_landmarks:
                if result.facial_transformation_matrixes:
                    pitch, yaw, roll = get_angles_from_matrix(
                        result.facial_transformation_matrixes[0])
                else:
                    lms = result.face_landmarks[0]
                    pitch, yaw, roll = get_angles_from_landmarks(lms, w, h)

            distracted_raw = is_distracted(pitch, yaw)
            now = time.time()

            # Apply grace period
            if distracted_raw:
                if distraction_start is None:
                    distraction_start = now
                distracted = (now - distraction_start) >= DISTRACTION_DELAY_SEC
            else:
                distraction_start = None
                distracted = False

            # ── Terminal log on state change ──
            if distracted and not was_distracted:
                reasons = []
                if pitch and pitch > PITCH_DOWN_THRESHOLD:
                    reasons.append(f"down {pitch:.1f}°")
                if pitch and pitch < PITCH_UP_THRESHOLD:
                    reasons.append(f"up {pitch:.1f}°")
                if yaw and abs(yaw) > YAW_THRESHOLD:
                    reasons.append(f"side {yaw:.1f}°")
                print(f"\033[1;31m[{time.strftime('%H:%M:%S')}] "
                      f"DISTRACTED – {', '.join(reasons)}\033[0m")
            elif not distracted and was_distracted:
                print(f"\033[1;32m[{time.strftime('%H:%M:%S')}] "
                      f"FOCUSED – video reset ✓\033[0m")
            was_distracted = distracted

            # ── Video + Audio control ──────────
            if distracted:
                # Start audio on first distracted frame
                if not audio_playing:
                    pygame.mixer.music.play()
                    audio_playing = True
                # Advance video frame at correct fps
                if now - last_vid_t >= frame_delay:
                    ok_v, vf = vid.read()
                    if not ok_v:
                        # Loop: restart both video and audio
                        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        pygame.mixer.music.rewind()
                        ok_v, vf = vid.read()
                    if ok_v:
                        current_vid_frame = vf
                    last_vid_t = now
            else:
                # Stop audio and rewind to start
                if audio_playing:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.rewind()
                    audio_playing = False
                # Freeze video at frame 0
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_v, vf = vid.read()
                if ok_v:
                    current_vid_frame = vf
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                last_vid_t = now

            # ── Compose combined window ────────
            dist_secs = (now - distraction_start) if distraction_start else 0.0
            draw_cam_hud(cam_frame, pitch, yaw, roll, distracted, dist_secs)
            panel   = build_panel(current_vid_frame, distracted)
            divider = make_divider(CAM_H, distracted)
            combined = np.hstack([cam_frame, divider, panel])

            cv2.imshow("Focus Monitor  (Q to quit)", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        landmarker.close()
        cap.release()
        vid.release()
        cv2.destroyAllWindows()
        print("\n\033[1;36m[INFO] Focus Monitor stopped. Stay focused! 💪\033[0m\n")


if __name__ == "__main__":
    main()
