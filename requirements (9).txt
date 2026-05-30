from rehab_core import ExerciseConfig, run_app


run_app(
    ExerciseConfig(
        title="0209v3 AI 復健辨識 - 肩部循環",
        subtitle="依照原 0209v3 的肩部高度變化邏輯計次，目標 10 次。",
        kind="shoulder_circles",
        target_reps=10,
    )
)
