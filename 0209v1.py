import cv2
import mediapipe as mp
import numpy as np
import screeninfo
import os
import pyttsx3
import threading

VIDEO_FILE = r'C:\Users\User\Desktop\media\0209v\0514v1.mp4' 
TARGET_REPS_PER_SIDE = 5

speech_engine = None

def speak(text):
    global speech_engine
    def _say():
        global speech_engine
        try:
            speech_engine = pyttsx3.init()
            speech_engine.setProperty('rate', 150)
            speech_engine.say(text)
            speech_engine.runAndWait()
        except:
            pass
    threading.Thread(target=_say, daemon=True).start()

def stop_speech():
    global speech_engine
    try:
        if speech_engine:
            speech_engine.stop()
    except:
        pass

should_exit = False
skip_demo = False
is_paused = False
current_frame_idx = 0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

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

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

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

def crop_to_fill(image, target_w, target_h):
    h, w = image.shape[:2]
    if (w/h) > (target_w/target_h):
        new_w = int(h * (target_w/target_h))
        offset = (w - new_w) // 2
        crop = image[:, offset:offset+new_w]
    else:
        new_h = int(w * (target_h/target_w))
        offset = (h - new_h) // 2
        crop = image[offset:offset+new_h, :]
    return cv2.resize(crop, (target_w, target_h))

def run_trainer():
    global should_exit, skip_demo, is_paused, current_frame_idx
    try:
        monitors = screeninfo.get_monitors()
        sw, sh = monitors[0].width, monitors[0].height
    except:
        sw, sh = 1280, 720

    if not os.path.exists(VIDEO_FILE): return

    win_name = 'AI Trainer'
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    demo_cap = cv2.VideoCapture(VIDEO_FILE)
    fps = demo_cap.get(cv2.CAP_PROP_FPS) or 30
    all_frames = []
    while True:
        ret, frame = demo_cap.read()
        if not ret: break
        all_frames.append(frame)
    demo_cap.release()
    
    demo_data_cache = [None] * len(all_frames)
    cv2.setMouseCallback(win_name, click_event, param=("DEMO", sw, sh, len(all_frames)))
    speak("歡迎使用 AI 教練。請先觀察手肘彎曲示範，也可以點擊右下角跳過。")
    
    while not skip_demo and not should_exit:
        raw_frame = all_frames[current_frame_idx].copy()
        display_frame = letterbox_image(raw_frame, sw, sh)
        
        if demo_data_cache[current_frame_idx] is None:
            res = pose.process(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                p1 = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                p2 = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                p3 = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y]
                demo_data_cache[current_frame_idx] = calculate_angle(p1, p2, p3)
            else:
                demo_data_cache[current_frame_idx] = 180

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
            if current_frame_idx >= len(all_frames): break

    if not should_exit:
        demo_data = [d if d is not None else 180 for d in demo_data_cache]
        cv2.setMouseCallback(win_name, click_event, param=("LIVE", sw, sh, 0))
        cap = cv2.VideoCapture(0)
        current_target = "LEFT" 
        counters = {"LEFT": 0, "RIGHT": 0}
        stage = "wait"
        demo_idx = 0

        speak(f"現在開始訓練。請先使用右手進行{TARGET_REPS_PER_SIDE}次手肘彎曲。")

        while cap.isOpened() and not should_exit:
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            frame = letterbox_image(frame, sw, sh)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            color = (0, 0, 255)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if current_target == "LEFT":
                    s, e, w_pt = mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST
                else:
                    s, e, w_pt = mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST

                if lm[e].visibility > 0.7:
                    p1, p2, p3 = [lm[s].x, lm[s].y], [lm[e].x, lm[e].y], [lm[w_pt].x, lm[w_pt].y]
                    user_angle = calculate_angle(p1, p2, p3)
                    is_correct = abs(user_angle - demo_data[demo_idx]) < 25
                    color = (0, 255, 0) if is_correct else (0, 0, 255)

                    if user_angle > 160: stage = "stretch"
                    if user_angle < 45 and stage == "stretch":
                        stage = "flex"
                        counters[current_target] += 1
                        speak(str(counters[current_target]))
                        if counters["LEFT"] >= TARGET_REPS_PER_SIDE and current_target == "LEFT":
                            current_target = "RIGHT"
                            stage = "wait"
                            speak("做得好！現在請換左手進行。")

            cv2.rectangle(frame, (0, 0), (sw, 130), (30, 30, 30), -1)
            cv2.putText(frame, f"RIGHT: {counters['LEFT']} / {TARGET_REPS_PER_SIDE}", (40, 50), 1, 2.5, (0, 255, 255), 3)
            cv2.putText(frame, f"LEFT: {counters['RIGHT']} / {TARGET_REPS_PER_SIDE}", (40, 100), 1, 2.5, (0, 255, 255), 3)
            cv2.rectangle(frame, (0, sh-60), (150, sh), (50, 50, 200), -1)
            cv2.putText(frame, "EXIT", (45, sh-20), 1, 1.5, (255, 255, 255), 2)
            
            thumb_w, thumb_h = 400, 300
            thumb = crop_to_fill(all_frames[demo_idx], thumb_w, thumb_h)
            frame[sh-thumb_h-20:sh-20, sw-thumb_w-20:sw-20] = thumb
            cv2.rectangle(frame, (sw-thumb_w-20, sh-thumb_h-20), (sw-20, sh-20), color, 5)

           # ... (前面的程式碼維持不變)

            cv2.imshow(win_name, frame)
            demo_idx = (demo_idx + 1) % len(all_frames)
            
            if cv2.waitKey(1) & 0xFF == 27: 
                stop_speech()
                break
            
            # --- 修改後的結束區塊 ---
            if counters["RIGHT"] >= TARGET_REPS_PER_SIDE:
                # 1. 這裡不使用原本的 speak()，因為 speak 是非同步的會被切斷
                print("訓練完成，正在播放語音...")
                
                # 2. 直接在主程式中初始化並播放 (同步執行)
                # 程式會在這裡「停住」直到語音播報完畢，這期間視窗會維持最後一幀
                temp_engine = pyttsx3.init()
                temp_engine.setProperty('rate', 150)
                temp_engine.say("恭喜完成今天的訓練，你做得非常棒！")
                temp_engine.runAndWait() # 關鍵：確保話說完才跑下一行
                
                # 3. 語音說完了，現在才執行關閉與結束
                print("語音播放結束，關閉程式。")
                break 

        cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":

    import datetime

    def save_rehab_data(counters):
        file_path = "rehab_log.csv"
    # 檢查檔案是否存在，若不存在則寫入標題
        file_exists = os.path.isfile(file_path)
    
        try:
            with open(file_path, "a", encoding="utf-8-sig") as f: # utf-8-sig 讓 Excel 不會亂碼
                if not file_exists:
                    f.write("訓練時間, 左手次數, 右手次數\n")
            
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 根據 0209v1.py 的邏輯，counters 的 Key 是 'LEFT' 和 'RIGHT'
                f.write(f"{timestamp}, {counters.get('LEFT', 0)}, {counters.get('RIGHT', 0)}\n")
            print(f"數據已成功儲存至 {file_path}")
        except Exception as e:
            print(f"儲存失敗：{e}")

# 在 run_trainer() 的最後調用：
# save_rehab_data(counters)
    run_trainer()
