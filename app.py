import threading

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


TARGET_REPS_PER_SIDE = 5

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


class RehabProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.lock = threading.Lock()
        self.current_target = "RIGHT"
        self.counters = {"LEFT": 0, "RIGHT": 0}
        self.stage = "wait"
        self.last_angle = 0.0
        self.status = "請站到鏡頭前，先做右手手肘彎曲。"

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        h, w = image.shape[:2]
        color = (0, 0, 255)
        status = "未偵測到姿勢，請讓上半身進入畫面。"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if self.current_target == "LEFT":
                shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER
                elbow = mp_pose.PoseLandmark.LEFT_ELBOW
                wrist = mp_pose.PoseLandmark.LEFT_WRIST
                side_label = "左手"
            else:
                shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER
                elbow = mp_pose.PoseLandmark.RIGHT_ELBOW
                wrist = mp_pose.PoseLandmark.RIGHT_WRIST
                side_label = "右手"

            if lm[elbow].visibility > 0.7:
                p1 = [lm[shoulder].x, lm[shoulder].y]
                p2 = [lm[elbow].x, lm[elbow].y]
                p3 = [lm[wrist].x, lm[wrist].y]
                user_angle = calculate_angle(p1, p2, p3)
                color = (0, 255, 0)
                status = f"{side_label}角度：{user_angle:.0f} 度"

                if user_angle > 160:
                    self.stage = "stretch"
                    status = f"{side_label}伸直，再慢慢彎曲。"

                if user_angle < 45 and self.stage == "stretch":
                    self.stage = "flex"
                    self.counters[self.current_target] += 1
                    status = f"{side_label}完成第 {self.counters[self.current_target]} 次。"

                    if (
                        self.current_target == "RIGHT"
                        and self.counters["RIGHT"] >= TARGET_REPS_PER_SIDE
                    ):
                        self.current_target = "LEFT"
                        self.stage = "wait"
                        status = "右手完成，請換左手。"

                if self.counters["LEFT"] >= TARGET_REPS_PER_SIDE:
                    status = "訓練完成，做得太棒了！"

                self.last_angle = user_angle
            else:
                status = "手肘不清楚，請調整鏡頭或光線。"

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

        overlay_h = 118
        cv2.rectangle(image, (0, 0), (w, overlay_h), (28, 28, 28), -1)
        cv2.putText(
            image,
            f"RIGHT: {self.counters['RIGHT']} / {TARGET_REPS_PER_SIDE}",
            (24, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"LEFT: {self.counters['LEFT']} / {TARGET_REPS_PER_SIDE}",
            (24, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            status,
            (24, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

        with self.lock:
            self.status = status

        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.set_page_config(page_title="AI 復健辨識", page_icon="🏃", layout="centered")
st.title("AI 復健辨識")
st.caption("請允許瀏覽器使用攝影機。部署到 Streamlit Cloud 後也會使用使用者自己的鏡頭。")

ctx = webrtc_streamer(
    key="rehab",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=RehabProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    with ctx.video_processor.lock:
        counters = dict(ctx.video_processor.counters)
        current_target = ctx.video_processor.current_target
        status = ctx.video_processor.status
else:
    counters = {"LEFT": 0, "RIGHT": 0}
    current_target = "RIGHT"
    status = "按 START 後開始辨識。"

col1, col2, col3 = st.columns(3)
col1.metric("右手", f"{counters['RIGHT']} / {TARGET_REPS_PER_SIDE}")
col2.metric("左手", f"{counters['LEFT']} / {TARGET_REPS_PER_SIDE}")
col3.metric("目前", "左手" if current_target == "LEFT" else "右手")
st.info(status)

st.markdown(
    "流程：先右手伸直再彎曲 5 次，完成後換左手 5 次。"
)
