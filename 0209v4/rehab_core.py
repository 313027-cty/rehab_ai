import base64
import html
import time
import threading
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.pose as mp_pose
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS


@dataclass(frozen=True)
class ExerciseConfig:
    action_name: str
    kind: str
    demo_video: str
    target_reps: int = 5
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
            status = self._process_hand_open(image, rgb)
        else:
            status = self._process_pose(image, rgb)

        with self.lock:
            self.status = status

        # streamlit-webrtc mirrors the displayed video element in some browsers.
        # Return a pre-flipped frame so the final on-screen image is not mirrored.
        return av.VideoFrame.from_ndarray(cv2.flip(image, 1), format="bgr24")

    def _process_hand_open(self, image, rgb):
        results = self.hands.process(rgb)

        if self._two_side_complete():
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, landmarks, HAND_CONNECTIONS)
            return "訓練完成，已停止計次。"

        status = "未偵測到手掌，請讓手掌進入畫面。"
        if not results.multi_hand_landmarks:
            return status

        for i, landmarks in enumerate(results.multi_hand_landmarks):
            raw_label = results.multi_handedness[i].classification[0].label.upper()
            label = "LEFT" if raw_label == "RIGHT" else "RIGHT"
            mp_drawing.draw_landmarks(image, landmarks, HAND_CONNECTIONS)

            if label != self.current_target:
                continue

            extension = hand_extension(landmarks)
            self.last_value = f"{extension:.2f}"
            status = f"{self._side_text()}手開合值：{extension:.2f}"

            if extension < 0.25:
                self.stage = "close"
                status = f"{self._side_text()}手握拳後再張開。"

            if (
                extension > 0.35
                and self.stage == "close"
                and self.counters[self.current_target] < self.config.target_reps
            ):
                self.stage = "open"
                self.counters[self.current_target] += 1
                status = f"{self._side_text()}手完成第 {self.counters[self.current_target]} 次。"
                self._advance_side_if_needed()

        return status

    def _process_pose(self, image, rgb):
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return "未偵測到姿勢，請讓上半身進入畫面。"

        lm = results.pose_landmarks.landmark
        if self.config.kind == "elbow":
            status = self._process_elbow(lm)
        elif self.config.kind == "shoulder_circles":
            status = self._process_shoulder_circles(lm)
        elif self.config.kind == "hand_raise":
            status = self._process_hand_raise(lm)
        else:
            status = self._process_lateral_raise(lm)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, POSE_CONNECTIONS)
        return status

    def _process_elbow(self, lm):
        if self._two_side_complete():
            return "訓練完成，已停止計次。"

        if self.current_target == "LEFT":
            shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER
            elbow = mp_pose.PoseLandmark.LEFT_ELBOW
            wrist = mp_pose.PoseLandmark.LEFT_WRIST
        else:
            shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER
            elbow = mp_pose.PoseLandmark.RIGHT_ELBOW
            wrist = mp_pose.PoseLandmark.RIGHT_WRIST

        if lm[elbow].visibility <= 0.7:
            return "手肘不清楚，請調整鏡頭或光線。"

        angle = calculate_angle(
            [lm[shoulder].x, lm[shoulder].y],
            [lm[elbow].x, lm[elbow].y],
            [lm[wrist].x, lm[wrist].y],
        )
        self.last_value = f"{angle:.0f} 度"
        status = f"{self._side_text()}手肘角度：{angle:.0f} 度"

        if angle > 160:
            self.stage = "stretch"
            status = f"{self._side_text()}手伸直，再慢慢彎曲。"

        if (
            angle < 45
            and self.stage == "stretch"
            and self.counters[self.current_target] < self.config.target_reps
        ):
            self.stage = "flex"
            self.counters[self.current_target] += 1
            status = f"{self._side_text()}手完成第 {self.counters[self.current_target]} 次。"
            self._advance_side_if_needed()

        return status

    def _process_shoulder_circles(self, lm):
        if self.reps >= self.config.target_reps:
            return "訓練完成，已停止計次。"

        sh_l = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        sh_r = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        hip_l = lm[mp_pose.PoseLandmark.LEFT_HIP]
        hip_r = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        rel_height = ((hip_l.y + hip_r.y) / 2) - ((sh_l.y + sh_r.y) / 2)
        self.last_value = f"{rel_height:.2f}"

        if rel_height > 0.41:
            self.stage = "up"
            return "偵測到上提，請放回。"

        if rel_height < 0.39 and self.stage == "up":
            self.stage = "down"
            self.reps += 1
            return f"完成第 {self.reps} 次。"

        return f"肩部高度值：{rel_height:.2f}"

    def _process_hand_raise(self, lm):
        if self.reps >= self.config.target_reps:
            return "訓練完成，已停止計次。"

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
            return "手已舉起，請放下完成一次。"

        both_down = l_wrist.y > l_shoulder.y and r_wrist.y > r_shoulder.y
        if both_down and self.stage == "up":
            self.stage = "down"
            self.reps += 1
            self.last_value = "down"
            return f"完成第 {self.reps} 次。"

        self.last_value = "down"
        return "請向上舉手。"

    def _process_lateral_raise(self, lm):
        if self.reps >= self.config.target_reps:
            return "訓練完成，已停止計次。"

        l_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
        r_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        points_visible = all(
            point.visibility > 0.7 for point in [l_elbow, r_elbow, l_shoulder, r_shoulder]
        )
        if not points_visible:
            return "肩膀或手肘不清楚，請調整鏡頭。"

        elbows_up = l_elbow.y < l_shoulder.y + 0.12 and r_elbow.y < r_shoulder.y + 0.12
        elbows_down = l_elbow.y > l_shoulder.y + 0.25 and r_elbow.y > r_shoulder.y + 0.25

        if elbows_up:
            self.stage = "up"
            self.last_value = "up"
            return "雙手已側舉，請放下完成一次。"

        if elbows_down and self.stage == "up":
            self.stage = "down"
            self.reps += 1
            self.last_value = "down"
            return f"完成第 {self.reps} 次。"

        self.last_value = "down"
        return "請雙手同時側舉。"

    def _advance_side_if_needed(self):
        if (
            self.config.two_sides
            and self.current_target == "RIGHT"
            and self.counters["RIGHT"] >= self.config.target_reps
        ):
            self.current_target = "LEFT"
            self.stage = "wait"

    def _two_side_complete(self):
        return (
            self.config.two_sides
            and self.counters["RIGHT"] >= self.config.target_reps
            and self.counters["LEFT"] >= self.config.target_reps
        )

    def _side_text(self):
        return "左" if self.current_target == "LEFT" else "右"


def read_processor_state(ctx):
    if not ctx.video_processor:
        return {
            "status": "按 START 後開始辨識。",
            "last_value": "-",
            "reps": 0,
            "counters": {"LEFT": 0, "RIGHT": 0},
            "current_target": "RIGHT",
        }

    with ctx.video_processor.lock:
        return {
            "status": ctx.video_processor.status,
            "last_value": ctx.video_processor.last_value,
            "reps": ctx.video_processor.reps,
            "counters": dict(ctx.video_processor.counters),
            "current_target": ctx.video_processor.current_target,
        }


def render_state(state, config, metrics_slot, status_slot):
    with metrics_slot.container():
        if config.two_sides:
            col1, col2, col3 = st.columns(3)
            col1.metric("右側", f"{state['counters']['RIGHT']} / {config.target_reps}")
            col2.metric("左側", f"{state['counters']['LEFT']} / {config.target_reps}")
            col3.metric("目前", "左側" if state["current_target"] == "LEFT" else "右側")
        else:
            col1, col2 = st.columns(2)
            col1.metric("計次", f"{state['reps']} / {config.target_reps}")
            col2.metric("狀態", state["last_value"])

    status_slot.info(state["status"])


def render_muted_video(video_path):
    encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    components.html(
        f"""
        <video controls muted playsinline style="width:100%; border-radius:8px;">
          <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
        </video>
        """,
        height=420,
    )


def install_speech_reader():
    components.html(
        """
        <script>
        const spoken = new Set();
        function readSpeechText() {
          try {
            const doc = window.parent.document;
            const node = doc.getElementById("rehab-speech-text");
            const text = node ? node.textContent.trim() : "";
            if (!text || spoken.has(text)) return;
            spoken.add(text);
            const synth = window.parent.speechSynthesis || window.speechSynthesis;
            if (!synth) return;
            synth.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = "zh-TW";
            utterance.rate = 1.05;
            synth.speak(utterance);
          } catch (e) {}
        }
        setInterval(readSpeechText, 700);
        </script>
        """,
        height=0,
    )


def run_app(config: ExerciseConfig):
    st.set_page_config(page_title=config.action_name, layout="centered")
    st.title(config.action_name)

    video_path = Path(__file__).parent / config.demo_video
    if not video_path.exists():
        video_path = Path(__file__).parent / Path(config.demo_video).name

    if video_path.exists():
        render_muted_video(video_path)
    else:
        st.warning("找不到示範影片，請確認 media 資料夾已上傳。")

    install_speech_reader()

    ctx = webrtc_streamer(
        key=config.kind,
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: RehabProcessor(config),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    metrics_slot = st.empty()
    status_slot = st.empty()
    speech_slot = st.empty()
    state = read_processor_state(ctx)
    render_state(state, config, metrics_slot, status_slot)
    speech_slot.markdown(
        '<span id="rehab-speech-text" style="display:none;">按 START 後開始辨識。</span>',
        unsafe_allow_html=True,
    )

    while ctx.state.playing:
        state = read_processor_state(ctx)
        render_state(state, config, metrics_slot, status_slot)
        speech_slot.markdown(
            f'<span id="rehab-speech-text" style="display:none;">{html.escape(state["status"])}</span>',
            unsafe_allow_html=True,
        )
        time.sleep(0.5)
