from covertype_rl.actions import build_action_space
from covertype_rl.data import load_covertype
from covertype_rl.fixed_sets import ensure_fixed_universe
from covertype_rl.full_a3c import build_parser
from covertype_rl.targets import resolve_target_set


def main():
    parser = build_parser()
    parser.description = "Preprocess the fixed Covertype set universe."
    args = parser.parse_args()

    from pathlib import Path

    root_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = root_dir / csv_path

    data = load_covertype(csv_path, n_bins=args.n_bins)
    target_items, target_path = resolve_target_set(
        data,
        root_dir=root_dir,
        target_set=args.target_set,
        target_seed=args.target_seed,
        target_size=args.target_size,
    )
    actions = build_action_space(n_continuous=data.continuous.shape[1], n_bins=data.n_bins)
    universe, universe_dir = ensure_fixed_universe(
        data=data,
        actions=actions,
        target_items=target_items,
        target_path=target_path,
        root_dir=root_dir,
        n_sets=args.n_sets,
        seed=args.seed,
        min_set_size=args.min_set_size,
        max_set_size=args.max_set_size,
        preprocess_name=args.preprocess_name,
        force=args.force_preprocess,
    )
    print(f"Fixed universe: {universe_dir}")
    print(f"Sets: {universe.n_sets}")
    print(f"State dim: {universe.state_dim}")
    print(f"Action dim: {universe.action_dim}")
    print(f"Target items: {universe.target_set_size}")


if __name__ == "__main__":
    main()
