from rehab_core import ExerciseConfig, run_app


run_app(
    ExerciseConfig(
        title="0209v4 AI 復健辨識 - 舉手",
        subtitle="任一手腕高過肩膀後放下，完成一次，目標 10 次。",
        kind="hand_raise",
        target_reps=10,
    )
)
