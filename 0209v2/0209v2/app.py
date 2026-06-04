from rehab_core import ExerciseConfig, run_app


run_app(
    ExerciseConfig(
        action_name="手掌開合",
        kind="hand_open",
        demo_video="media/0514v2.mp4",
        target_reps=5,
        two_sides=True,
    )
)
