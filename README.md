from rehab_core import ExerciseConfig, run_app


run_app(
    ExerciseConfig(
        title="0209v1 AI 復健辨識 - 手肘彎曲",
        subtitle="右手手肘彎曲 5 次後換左手 5 次。",
        kind="elbow",
        target_reps=5,
        two_sides=True,
    )
)
