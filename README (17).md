import threading
from dataclasses import dataclass

import av
import cv2
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.pose as mp_pose
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


@dataclass(frozen=True)
class ExerciseConfig:
    title: str
    subtitle: str
    kind: str
    target_reps: int
    two_sides: bool = False


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def hand_extension(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    distances = [
        np.sqrt(
            (hand_landmarks.landmark[tip].x - wrist.x) ** 2
            + (hand_landmarks.landmark[tip].y - wrist.y) ** 2
        )
        for tip in tips
    ]
    return float(sum(distances) / len(distances))


class RehabProcessor(VideoProcessorBase):
    def __init__(self, config: ExerciseConfig):
        self.config = config
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.lock = threading.Lock()
        self.stage = "wait"
        self.reps = 0
        self.current_target = "RIGHT"
        self.counters = {"LEFT": 0, "RIGHT": 0}
        self.status = "按 START 後開始辨識。"
        self.last_value = "-"

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.config.kind == "hand_open":
            color, status = self._process_hand_open(image, rgb)
        else:
            color, status = self._process_pose(image, rgb)

        self._draw_overlay(image, color, status)

        with self.lock:
            self.status = status

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def _process_hand_open(self, image, rgb):
        results = self.hands.process(rgb)
        color = (0, 0, 255)
        status = "未偵測到手掌，請讓手掌進入畫面。"

        if not results.multi_hand_landmarks:
            return color, status

        for i, landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label.upper()
            if label != self.current_target:
                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
                continue

            extension = hand_extension(landmarks)
            self.last_value = f"{extension:.2f}"
            color = (0, 255, 0)
            status = f"{self._side_text()}手掌開合值：{extension:.2f}"

            if extension < 0.25:
                self.stage = "close"
                status = f"{self._side_text()}握拳後再張開。"

            if extension > 0.35 and self.stage == "close":
                self.stage = "open"
                self.counters[self.current_target] += 1
                status = f"{self._side_text()}完成第 {self.counters[self.current_target]} 次。"
                self._advance_side_if_needed()

            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

        return color, status

    def _process_pose(self, image, rgb):
        results = self.pose.process(rgb)
        color = (0, 0, 255)
        status = "未偵測到姿勢，請讓上半身進入畫面。"

        if not results.pose_landmarks:
            return color, status

        lm = results.pose_landmarks.landmark
        kind = self.config.kind

        if kind == "elbow":
            color, status = self._process_elbow(lm)
        elif kind == "shoulder_circles":
            color, status = self._process_shoulder_circles(lm)
        elif kind == "hand_raise":
            color, status = self._process_hand_raise(lm)
        elif kind == "lateral_raise":
            color, status = self._process_lateral_raise(lm)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return color, status

    def _process_elbow(self, lm):
        if self.current_target == "LEFT":
            shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER
            elbow = mp_pose.PoseLandmark.LEFT_ELBOW
            wrist = mp_pose.PoseLandmark.LEFT_WRIST
        else:
            shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER
            elbow = mp_pose.PoseLandmark.RIGHT_ELBOW
            wrist = mp_pose.PoseLandmark.RIGHT_WRIST

        if lm[elbow].visibility <= 0.7:
            return (0, 0, 255), "手肘不清楚，請調整鏡頭或光線。"

        angle = calculate_angle(
            [lm[shoulder].x, lm[shoulder].y],
            [lm[elbow].x, lm[elbow].y],
            [lm[wrist].x, lm[wrist].y],
        )
        self.last_value = f"{angle:.0f} 度"
        status = f"{self._side_text()}手肘角度：{angle:.0f} 度"

        if angle > 160:
            self.stage = "stretch"
            status = f"{self._side_text()}伸直，再慢慢彎曲。"

        if angle < 45 and self.stage == "stretch":
            self.stage = "flex"
            self.counters[self.current_target] += 1
            status = f"{self._side_text()}完成第 {self.counters[self.current_target]} 次。"
            self._advance_side_if_needed()

        return (0, 255, 0), status

    def _process_shoulder_circles(self, lm):
        sh_l = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        sh_r = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        hip_l = lm[mp_pose.PoseLandmark.LEFT_HIP]
        hip_r = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        rel_height = ((hip_l.y + hip_r.y) / 2) - ((sh_l.y + sh_r.y) / 2)
        self.last_value = f"{rel_height:.2f}"

        if rel_height > 0.41:
            self.stage = "up"
            return (0, 255, 0), "偵測到上提，請放回。"

        if rel_height < 0.39 and self.stage == "up":
            self.stage = "down"
            self.reps += 1
            return (0, 255, 0), f"完成第 {self.reps} 次。"

        return (0, 0, 255), f"肩部高度值：{rel_height:.2f}"

    def _process_hand_raise(self, lm):
        l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        is_raised = (
            l_wrist.visibility > 0.5 and l_wrist.y < l_shoulder.y - 0.05
        ) or (
            r_wrist.visibility > 0.5 and r_wrist.y < r_shoulder.y - 0.05
        )

        if is_raised:
            self.stage = "up"
            self.last_value = "up"
            return (0, 255, 0), "手已舉起，請放下完成一次。"

        both_down = l_wrist.y > l_shoulder.y and r_wrist.y > r_shoulder.y
        if both_down and self.stage == "up":
            self.stage = "down"
            self.reps += 1
            self.last_value = "down"
            return (0, 255, 0), f"完成第 {self.reps} 次。"

        self.last_value = "down"
        return (0, 0, 255), "請向上舉手。"

    def _process_lateral_raise(self, lm):
        l_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
        r_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        points_visible = all(
            point.visibility > 0.7 for point in [l_elbow, r_elbow, l_shoulder, r_shoulder]
        )
        if not points_visible:
            return (0, 0, 255), "肩膀或手肘不清楚，請調整鏡頭。"

        elbows_up = l_elbow.y < l_shoulder.y + 0.12 and r_elbow.y < r_shoulder.y + 0.12
        elbows_down = l_elbow.y > l_shoulder.y + 0.25 and r_elbow.y > r_shoulder.y + 0.25

        if elbows_up:
            self.stage = "up"
            self.last_value = "up"
            return (0, 255, 0), "雙手已側舉，請放下完成一次。"

        if elbows_down and self.stage == "up":
            self.stage = "down"
            self.reps += 1
            self.last_value = "down"
            return (0, 255, 0), f"完成第 {self.reps} 次。"

        self.last_value = "down"
        return (0, 0, 255), "請雙手同時側舉。"

    def _advance_side_if_needed(self):
        if (
            self.config.two_sides
            and self.current_target == "RIGHT"
            and self.counters["RIGHT"] >= self.config.target_reps
        ):
            self.current_target = "LEFT"
            self.stage = "wait"

    def _side_text(self):
        return "左" if self.current_target == "LEFT" else "右"

    def _draw_overlay(self, image, color, status):
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, 0), (w, 118), (28, 28, 28), -1)

        if self.config.two_sides:
            cv2.putText(
                image,
                f"RIGHT: {self.counters['RIGHT']} / {self.config.target_reps}",
                (24, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"LEFT: {self.counters['LEFT']} / {self.config.target_reps}",
                (24, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                image,
                f"REPS: {self.reps} / {self.config.target_reps}",
                (24, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 255),
                3,
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


def run_app(config: ExerciseConfig):
    st.set_page_config(page_title=config.title, layout="centered")
    st.title(config.title)
    st.caption(config.subtitle)

    ctx = webrtc_streamer(
        key=config.kind,
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: RehabProcessor(config),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        with ctx.video_processor.lock:
            status = ctx.video_processor.status
            last_value = ctx.video_processor.last_value
            reps = ctx.video_processor.reps
            counters = dict(ctx.video_processor.counters)
            current_target = ctx.video_processor.current_target
    else:
        status = "按 START 後開始辨識。"
        last_value = "-"
        reps = 0
        counters = {"LEFT": 0, "RIGHT": 0}
        current_target = "RIGHT"

    if config.two_sides:
        col1, col2, col3 = st.columns(3)
        col1.metric("右側", f"{counters['RIGHT']} / {config.target_reps}")
        col2.metric("左側", f"{counters['LEFT']} / {config.target_reps}")
        col3.metric("目前", "左側" if current_target == "LEFT" else "右側")
    else:
        col1, col2 = st.columns(2)
        col1.metric("次數", f"{reps} / {config.target_reps}")
        col2.metric("狀態值", last_value)

    st.info(status)
    st.markdown("請允許瀏覽器使用攝影機。部署到 Streamlit Cloud 後會使用使用者自己的鏡頭。")
