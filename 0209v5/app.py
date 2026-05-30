from rehab_core import ExerciseConfig, run_app


run_app(
    ExerciseConfig(
        title="0209v5 AI 復健辨識 - 側平舉",
        subtitle="雙手手肘上舉後放下，完成一次，目標 10 次。",
        kind="lateral_raise",
        target_reps=10,
    )
)
