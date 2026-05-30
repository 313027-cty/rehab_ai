import os

# 降低 MediaPipe / TFLite / absl 的雜訊輸出（避免控制台被 WARNING 佔滿）
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOGGING_MIN_LEVEL", "3")

import cv2
import mediapipe
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
mp = mediapipe
import numpy as np
import screeninfo
import pyttsx3
import threading
import warnings

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass

# 常見 protobuf 退場警告先關掉（不影響執行）
warnings.filterwarnings(
    "ignore",
    message=r".*SymbolDatabase\\.GetPrototype\\(\\) is deprecated.*",
    category=UserWarning,
)

import sys

# 取得目前程式執行的地方
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# 組合出影片路徑
VIDEO_FILE = os.path.join(base_path, "media", "0514v2.mp4")

# --- 語音系統 (支援即時中斷) ---
speech_lock = threading.Lock()  # <--- 就是漏了這一行！
current_engine = None
# --- 訓練設定 ---
TARGET_REPS_PER_SIDE = 5  # 每手要做幾次
def speak(text):
    
    def _say():
        global current_engine
        with speech_lock:
            try:
                engine = pyttsx3.init()
                current_engine = engine 
                engine.setProperty('rate', 200) 
                engine.setProperty('volume', 1.0)
                
                voices = engine.getProperty('voices')
                for v in voices:
                    if "Chinese" in v.name or "CHT" in v.name or "Han" in v.name:
                        engine.setProperty('voice', v.id)
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

# --- 影像裁切與處理 ---
def crop_to_fill_center(image, target_w, target_h):
    h, w = image.shape[:2]
    if (w/h) > (target_w/target_h):
        new_w = int(h * (target_w/target_h))
        offset = (w - new_w) // 2
        crop = image[:, offset:offset+new_w]
    else:
        new_h = int(w * (target_h/target_w))
        offset = int((h - new_h) * 0.5) 
        crop = image[offset:offset+new_h, :]
    return cv2.resize(crop, (target_w, target_h))

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

def get_hand_extension(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    dist_sum = sum([np.sqrt((hand_landmarks.landmark[t].x - wrist.x)**2 + 
                             (hand_landmarks.landmark[t].y - wrist.y)**2) for t in tips])
    return dist_sum / len(tips)

TARGET_REPS_PER_SIDE = 5  # 設定每一邊要做的次數
# --- 全域變數 ---
should_exit = False
skip_demo = False
is_paused = False
current_frame_idx = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
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
        demo_cap = cv2.VideoCapture(VIDEO_FILE)
        all_frames = []
        while True:
            ret, frame = demo_cap.read()
            if not ret:
                break
            all_frames.append(frame)
        demo_cap.release()

        demo_data_cache = [None] * len(all_frames)
    else:
        # 沒有示範影片也能跑：直接用相機訓練 + 用一張提示畫面當縮圖
        skip_demo = True
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
        demo_data_cache = [0.3]
    cv2.setMouseCallback(win_name, click_event, param=("DEMO", sw, sh, len(all_frames)))
    
    speak("歡迎使用 AI 教練。請先觀察示範動作，也可以點擊右下角跳過。")
    
    # --- 示範階段 ---
    while not skip_demo and not should_exit:
        raw_frame = all_frames[current_frame_idx].copy()
        display_frame = letterbox_image(raw_frame, sw, sh)
        if demo_data_cache[current_frame_idx] is None:
            res = hands.process(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
            demo_data_cache[current_frame_idx] = get_hand_extension(res.multi_hand_landmarks[0]) if res.multi_hand_landmarks else 0.3
        
        cv2.rectangle(display_frame, (0, sh-50), (150, sh), (50, 50, 200), -1)
        cv2.putText(display_frame, "EXIT", (45, sh-15), 1, 1.5, (255, 255, 255), 2)
        cv2.rectangle(display_frame, (sw-150, sh-50), (sw, sh), (200, 50, 50), -1)
        cv2.putText(display_frame, "SKIP >>", (sw-135, sh-15), 1, 1.5, (255, 255, 255), 2)
        cv2.imshow(win_name, display_frame)
        if cv2.waitKey(20) & 0xFF == 27: should_exit = True; break
        if not is_paused:
            current_frame_idx += 1
            if current_frame_idx >= len(all_frames): break

    # --- 訓練階段 ---
    if not should_exit:
        stop_speech()
        demo_data = [d if d is not None else 0.3 for d in demo_data_cache]
        cv2.setMouseCallback(win_name, click_event, param=("LIVE", sw, sh, 0))
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        current_target = "Right" 
        counters = {"Right": 0, "Left": 0} 
        stage = "wait"
        demo_idx = 0
        
        # 關鍵變數：確保開場白只被中斷一次，且不影響計次報數
        intro_interrupted = False

        speak("現在開始復健，我們先從右手開始。舉起手即可開始計次。")

        while cap.isOpened() and not should_exit:
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            frame = letterbox_image(frame, sw, sh)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            color = (0, 0, 255)

            if results.multi_hand_landmarks:
                # 【新邏輯】偵測到手部且開場白尚未結束時，立即中斷
                if not intro_interrupted:
                    stop_speech()
                    intro_interrupted = True # 設為 True 後，接下來的 speak (計次) 就不會被中斷

                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[i].classification[0].label
                    if label == current_target:
                        ext = get_hand_extension(hand_landmarks)
                        is_correct = abs(ext - demo_data[demo_idx]) < 0.15
                        color = (0, 255, 0) if is_correct else (0, 0, 255)

                        if ext < 0.25: stage = "close"
                        if ext > 0.35 and stage == "close":
                            stage = "open"
                            counters[label] += 1
                            speak(str(counters[label])) # 這裡的報數不會被中斷
                            
                            if current_target == "Right" and counters["Right"] >= TARGET_REPS_PER_SIDE:
                                current_target = "Left"
                                stage = "wait"
                                speak("右手完成！現在請換左手進行。")

                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # UI 繪製 (維持原樣)
            cv2.rectangle(frame, (0, 0), (sw, 130), (30, 30, 30), -1)
            cv2.putText(frame, f"RIGHT: {counters['Right']} / {TARGET_REPS_PER_SIDE}", (40, 100), 1, 2.5, (0, 255, 255), 3)
            cv2.putText(frame, f"LEFT: {counters['Left']} / {TARGET_REPS_PER_SIDE}", (40, 50), 1, 2.5, (0, 255, 255), 3)
            cv2.rectangle(frame, (0, sh-60), (150, sh), (50, 50, 200), -1)
            cv2.putText(frame, "EXIT", (45, sh-20), 1, 1.5, (255, 255, 255), 2)
            
            thumb_w, thumb_h = 400, 300
            thumb = crop_to_fill_center(all_frames[demo_idx], thumb_w, thumb_h)
            frame[sh-thumb_h-20:sh-20, sw-thumb_w-20:sw-20] = thumb
            cv2.rectangle(frame, (sw-thumb_w-20, sh-thumb_h-20), (sw-20, sh-20), color, 5)

            cv2.imshow(win_name, frame)
            demo_idx = (demo_idx + 1) % len(all_frames)
            
            if cv2.waitKey(1) & 0xFF == 27: break
            if counters["Left"] >= TARGET_REPS_PER_SIDE:
                speak("太棒了")
                cv2.waitKey(3000)
                break

        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_trainer()
