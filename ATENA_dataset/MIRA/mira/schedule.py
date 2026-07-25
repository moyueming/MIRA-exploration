from types import SimpleNamespace


def _lerp(start, end, progress):
    progress = min(max(float(progress), 0.0), 1.0)
    return float(start) + ((float(end) - float(start)) * progress)


def training_schedule(args, steps_done):
    total_steps = max(float(args.steps), 1.0)
    progress = min(max(float(steps_done) / total_steps, 0.0), 1.0)
    consolidation_start = min(
        max(float(args.consolidation_start), 1e-6),
        1.0 - 1e-6,
    )
    if progress <= consolidation_start:
        lr_scale = 1.0 - (0.5 * progress / consolidation_start)
        consolidation_progress = 0.0
    else:
        consolidation_progress = (
            (progress - consolidation_start) / (1.0 - consolidation_start)
        )
        lr_scale = _lerp(
            0.5,
            float(args.final_lr_scale),
            consolidation_progress,
        )

    return {
        "policy_lr": float(args.lr) * lr_scale,
        "mira_lr": float(args.mira_lr) * lr_scale,
        "alpha": float(args.alpha) * _lerp(
            1.0,
            float(args.final_alpha_scale),
            consolidation_progress,
        ),
        "entropy_coef": float(args.entropy_coef) * _lerp(
            1.0,
            float(args.final_entropy_scale),
            consolidation_progress,
        ),
        "auxiliary_reward_scale": _lerp(
            1.0,
            float(args.final_auxiliary_reward_scale),
            consolidation_progress,
        ),
    }


def runtime_args(args, schedule):
    runtime = SimpleNamespace(**vars(args))
    runtime.lr = float(schedule["policy_lr"])
    runtime.mira_lr = float(schedule["mira_lr"])
    runtime.alpha = float(schedule["alpha"])
    runtime.entropy_coef = float(schedule["entropy_coef"])
    scale = float(schedule["auxiliary_reward_scale"])
    runtime.w_column_coverage = float(args.w_column_coverage) * scale
    runtime.w_group_coverage = float(args.w_group_coverage) * scale
    runtime.w_structure = float(args.w_structure) * scale
    runtime.auxiliary_reward_scale = scale
    return runtime
