from rehab_core import ExerciseConfig, run_app


run_app(
    ExerciseConfig(
        action_name="舉手",
        kind="hand_raise",
        demo_video="media/0514v4.mp4",
        target_reps=5,
    )
)
