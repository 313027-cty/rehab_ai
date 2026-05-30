from rehab_core import ExerciseConfig, run_app


run_app(
    ExerciseConfig(
        title="0209v2 AI 復健辨識 - 手掌開合",
        subtitle="右手手掌開合 5 次後換左手 5 次。",
        kind="hand_open",
        target_reps=5,
        two_sides=True,
    )
)
