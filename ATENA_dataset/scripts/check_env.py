import importlib
import sys
from pathlib import Path


REQUIRED = ["numpy", "pandas", "scipy", "nltk", "zss", "networkx"]
OPTIONAL_TRAIN = ["tensorflow"]


def main():
    missing = []
    for name in REQUIRED:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    train_missing = [name for name in OPTIONAL_TRAIN if importlib.util.find_spec(name) is None]

    if missing:
        print("Missing required benchmark packages:", ", ".join(missing))
        print("Install with: pip install -r ATENA-A-EDA/benchmark/requirements.txt")
        raise SystemExit(1)
    if train_missing:
        print("Missing training packages:", ", ".join(train_missing))
        print("Install project training dependencies, e.g. TensorFlow from ../requirements.txt")
        raise SystemExit(1)

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "ATENA-A-EDA" / "benchmark"))
    from atena_baselines.env import make_env
    from atena_baselines.evaluate import official_metrics

    class Args:
        episode_length = 12
        max_terms_per_column = 10
        reward_mode = "compound"
        w_interestingness = 1.0
        w_diversity = 2.0
        w_coherency = 1.0
        empty_penalty = -1.0
        back_penalty = -0.2
        repeat_penalty = -1.0

    env = make_env("flights", 1, 0, Args())
    state = env.reset()
    assert state.shape[0] == env.state_dim
    done = False
    while not done:
        _, _, done, _ = env.step(env.sample_legal_action())
    metrics = official_metrics("flights", 1, env.actions)
    print("Environment OK")
    print(metrics)


if __name__ == "__main__":
    main()
