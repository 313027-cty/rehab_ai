import os
import warnings

# 降低 MediaPipe / TFLite / absl 的雜訊輸出（避免控制台被 WARNING 佔滿）
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOGGING_MIN_LEVEL", "3")

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    message=r".*SymbolDatabase\\.GetPrototype\\(\\) is deprecated.*",
    category=UserWarning,
)

import cv2
import mediapipe as mp
import numpy as np
import screeninfo
import pyttsx3
import threading
import sys

# --- 設定區 ---
# 取得目前程式執行的地方
if getattr(sys, "frozen", False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

VIDEO_FILE = os.path.join(base_path, "media", "0514v4.mp4")
TARGET_REPS = 10

# --- 語音系統 (支援即時中斷) ---
speech_lock = threading.Lock()
current_engine = None


def speak(text):
    def _say():
        global current_engine
        with speech_lock:
            try:
                engine = pyttsx3.init()
                current_engine = engine
                engine.setProperty("rate", 200)
                engine.setProperty("volume", 1.0)

                voices = engine.getProperty("voices")
                for v in voices:
                    if "Chinese" in v.name or "CHT" in v.name or "Han" in v.name:
                        engine.setProperty("voice", v.id)
                        break

                engine.say(text)
                engine.runAndWait()
            except:
                pass
            finally:
                current_engine = None

    threading.Thread(target=_say, daemon=True).start()


def stop_speech():
    """強制停止目前語音"""
    global current_engine
    if current_engine:
        try:
            current_engine.stop()
        except:
            pass

should_exit = False
skip_demo = False
is_paused = False
current_frame_idx = 0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

def click_event(event, x, y, flags, param):
    global should_exit, skip_demo, is_paused, current_frame_idx
    mode, sw, sh, total_frames = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if 0 < x < 200 and (sh - 100) < y < sh:
            should_exit = True
            stop_speech()
            return
        if mode == "DEMO":
            if (sw - 200) < x < sw and (sh - 100) < y < sh:
                skip_demo = True
                stop_speech()
            elif 50 < x < (sw - 50) and (sh - 135) < y < (sh - 100):
                progress_ratio = (x - 50) / (sw - 100)
                current_frame_idx = int(progress_ratio * (total_frames - 1))
            else:
                is_paused = not is_paused

def letterbox_image(image, target_w, target_h):
    h, w = image.shape[:2]
    aspect = w / h
    target_aspect = target_w / target_h
    if target_aspect > aspect:
        new_h = target_h
        new_w = int(aspect * new_h)
    else:
        new_w = target_w
        new_h = int(new_w / aspect)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off, x_off = (target_h-new_h)//2, (target_w-new_w)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def crop_to_fill_top(image, target_w, target_h):
    """垂直置中裁切邏輯 (同步自 v5)"""
    h_img, w_img = image.shape[:2]
    aspect = w_img / h_img
    target_aspect = target_w / target_h
    
    if aspect < target_aspect:
        raw_crop_w = w_img
        raw_crop_h = int(raw_crop_w * (target_h / target_w))
        raw_y_off = int((h_img - raw_crop_h) * 0.6)  # 偏向下一些，捕捉上半身
        raw_y_off = max(0, min(raw_y_off, h_img - raw_crop_h))
        crop = image[raw_y_off:raw_y_off+raw_crop_h, :]
    else:
        raw_crop_h = h_img
        raw_crop_w = int(raw_crop_h * (target_w / target_h))
        raw_x_off = (w_img - raw_crop_w) // 2
        crop = image[:, raw_x_off:raw_x_off+raw_crop_w]
        
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

def run_trainer():
    global should_exit, skip_demo, is_paused, current_frame_idx
    try:
        monitors = screeninfo.get_monitors()
        sw, sh = monitors[0].width, monitors[0].height
    except:
        sw, sh = 1280, 720

    has_demo_video = os.path.exists(VIDEO_FILE)

    win_name = 'AI Trainer'
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if has_demo_video:
        # 載入示範影片
        demo_cap = cv2.VideoCapture(VIDEO_FILE)
        fps = demo_cap.get(cv2.CAP_PROP_FPS) or 30
        all_frames = []
        while True:
            ret, frame = demo_cap.read()
            if not ret:
                break
            all_frames.append(frame)
        demo_cap.release()
    else:
        # 沒有示範影片也能跑：直接進入相機訓練模式，並用提示畫面當縮圖
        skip_demo = True
        fps = 30
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "DEMO VIDEO NOT FOUND",
            (30, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            placeholder,
            "Using live camera only",
            (55, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        all_frames = [placeholder]
    
    cv2.setMouseCallback(win_name, click_event, param=("DEMO", sw, sh, len(all_frames)))
    speak("歡迎使用 AI 教練。請先觀察示範動作，也可以點擊右下角跳過。")

    # === DEMO 播放階段 ===
    while not skip_demo and not should_exit:
        raw_frame = all_frames[current_frame_idx].copy()
        display_frame = letterbox_image(raw_frame, sw, sh)
        
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, sh-130), (sw, sh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        bar_start, bar_end = 50, sw-50
        cv2.line(display_frame, (bar_start, sh-110), (bar_end, sh-110), (80, 80, 80), 4)
        prog_x = bar_start + int((current_frame_idx / (len(all_frames)-1)) * (bar_end - bar_start))
        cv2.line(display_frame, (bar_start, sh-110), (prog_x, sh-110), (0, 0, 255), 6)
        cv2.circle(display_frame, (prog_x, sh-110), 8, (0, 0, 255), -1)

        curr_time = f"{int(current_frame_idx/fps)//60:02d}:{int(current_frame_idx/fps)%60:02d}"
        total_time = f"{int(len(all_frames)/fps)//60:02d}:{int(len(all_frames)/fps)%60:02d}"
        cv2.putText(display_frame, f"{curr_time} / {total_time}", (50, sh-60), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(display_frame, "PAUSED" if is_paused else "PLAYING", (sw//2-60, sh-60), 1, 1.5, (0, 255, 255), 2)
        
        cv2.rectangle(display_frame, (0, sh-50), (150, sh), (50, 50, 200), -1)
        cv2.putText(display_frame, "EXIT", (45, sh-15), 1, 1.5, (255, 255, 255), 2)
        cv2.rectangle(display_frame, (sw-150, sh-50), (sw, sh), (200, 50, 50), -1)
        cv2.putText(display_frame, "SKIP >>", (sw-135, sh-15), 1, 1.5, (255, 255, 255), 2)

        cv2.imshow(win_name, display_frame)
        if cv2.waitKey(20) & 0xFF == 27: 
            should_exit = True
            stop_speech()
            break
        if not is_paused:
            current_frame_idx += 1
            if current_frame_idx >= len(all_frames): 
                current_frame_idx = 0  # 循環播放

    # === 進入實際訓練階段 ===
    if not should_exit:
        cv2.setMouseCallback(win_name, click_event, param=("LIVE", sw, sh, 0))
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        reps = 0
        stage = "down"
        demo_idx = 0

        speak(f"現在開始訓練，請向上舉手，目標{TARGET_REPS}次。")
        intro_interrupted = False

        while cap.isOpened() and not should_exit:
            success, frame = cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            frame_display = letterbox_image(frame, sw, sh)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            color = (0, 0, 255)

            if results.pose_landmarks:
                if not intro_interrupted:
                    stop_speech()
                    intro_interrupted = True
                lm = results.pose_landmarks.landmark
                
                # === 簡單化的舉手動作偵測（比 v5 更簡單）===
                l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                # 只要任一隻手腕明顯高過肩膀就視為「舉起」
                is_raised = (l_wrist.visibility > 0.5 and l_wrist.y < l_shoulder.y - 0.05) or \
                            (r_wrist.visibility > 0.5 and r_wrist.y < r_shoulder.y - 0.05)

                if is_raised:
                    stage = "up"
                    color = (0, 255, 0)
                # 雙手都放回肩膀下方才算完成一次
                elif (l_wrist.y > l_shoulder.y and r_wrist.y > r_shoulder.y) and stage == "up":
                    stage = "down"
                    reps += 1
                    speak(str(reps))

                mp_drawing.draw_landmarks(frame_display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # UI 顯示
            cv2.rectangle(frame_display, (0, 0), (sw, 100), (30, 30, 30), -1)
            cv2.putText(frame_display, f"REPS: {reps} / {TARGET_REPS}", (40, 65), 1, 2.5, (0, 255, 255), 3)
            cv2.putText(frame_display, "HAND RAISE", (sw - 400, 65), 1, 2.8, (255, 255, 255), 3)

            cv2.rectangle(frame_display, (0, sh-60), (150, sh), (50, 50, 200), -1)
            cv2.putText(frame_display, "EXIT", (45, sh-20), 1, 1.5, (255, 255, 255), 2)

            # 右下角示範縮圖
            thumb_w, thumb_h = 400, 300
            thumb = crop_to_fill_top(all_frames[demo_idx], thumb_w, thumb_h)
            frame_display[sh-thumb_h-20:sh-20, sw-thumb_w-20:sw-20] = thumb
            cv2.rectangle(frame_display, (sw-thumb_w-20, sh-thumb_h-20), (sw-20, sh-20), color, 5)

            cv2.imshow(win_name, frame_display)
            demo_idx = (demo_idx + 1) % len(all_frames)
            
            if cv2.waitKey(1) & 0xFF == 27: 
                stop_speech()
                break
            if reps >= TARGET_REPS:
                speak("太棒了")
                cv2.waitKey(2000)
                break

        cap.release()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_trainer()
