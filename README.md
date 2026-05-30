from rehab_core import ExerciseConfig, run_app


run_app(
    ExerciseConfig(
        action_name="手肘彎曲",
        kind="elbow",
        demo_video="media/0514v1.mp4",
        target_reps=5,
        two_sides=True,
    )
)
