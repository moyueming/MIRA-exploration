import math
import random
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT_DIR / "ATENA-A-EDA" / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from atena.evaluation.distance import display_distance  # noqa: E402
from atena.simulation.actions import (  # noqa: E402
    AggregationFunction,
    BackAction,
    FilterAction,
    FilterOperator,
    GroupAction,
)
from atena.simulation.dataset import (  # noqa: E402
    CyberDatasetName,
    Dataset,
    DatasetMeta,
    FlightsDatasetName,
    SchemaName,
)
from atena.simulation.tokenization import tokenize_column  # noqa: E402

from .avp_loader import load_avp


def dataset_enum(schema: str, dataset_number: int):
    schema_name = SchemaName(schema.lower())
    if schema_name is SchemaName.FLIGHTS:
        return schema_name, FlightsDatasetName(int(dataset_number))
    if schema_name is SchemaName.CYBER:
        return schema_name, CyberDatasetName(int(dataset_number))
    raise ValueError(f"Unsupported official A-EDA schema: {schema}")


class AtenaActionSpace:
    """Finite action index wrapper over ATENA's parameterized actions."""

    def __init__(
        self,
        dataset: Dataset,
        max_terms_per_column: int = 10,
        seed: int = 0,
        avp_terms: Optional[Mapping[str, Sequence[str]]] = None,
    ):
        self.dataset = dataset
        self.rng = random.Random(int(seed))
        self.actions = [BackAction()]
        self.action_kinds = ["BACK"]
        self.column_term_map: Dict[str, List[str]] = {}
        avp_terms = avp_terms or {}

        for column in self.dataset.columns:
            tokens = tokenize_column(self.dataset.dataset_df, column)
            tokens = sorted(tokens, key=lambda value: str(value))
            if max_terms_per_column > 0:
                # Follow ATENA's bounded filter-term spirit: keep frequent tokens first.
                counts = self.dataset.dataset_df[column].astype(str).value_counts(dropna=False)
                tokens = sorted(tokens, key=lambda value: (-int(counts.get(str(value), 0)), str(value)))
                tokens = tokens[: int(max_terms_per_column)]
            for term in avp_terms.get(str(column), []):
                if term not in tokens:
                    tokens.append(str(term))
            if not tokens:
                tokens = ["<UNK>"]
            self.column_term_map[column] = tokens

            for op in FilterOperator:
                for term in tokens:
                    self.actions.append(FilterAction(column, op, term))
                    self.action_kinds.append("FILTER")

        for grouped_column in self.dataset.columns:
            for aggregated_column in self.dataset.primary_key_columns:
                self.actions.append(
                    GroupAction(grouped_column, aggregated_column, AggregationFunction.COUNT)
                )
                self.action_kinds.append("GROUP")

    def __len__(self):
        return len(self.actions)

    def get(self, index: int):
        return self.actions[int(index)]

    def kind(self, index: int) -> str:
        return self.action_kinds[int(index)]

class AtenaEDAEnv:
    """Official ATENA benchmark environment with task rewards for training."""

    def __init__(
        self,
        schema: str,
        dataset_number: int,
        episode_length: int = 12,
        max_terms_per_column: int = 10,
        seed: int = 0,
        avp="0",
        w_kl: float = 1.5,
        w_compaction: float = 2.0,
        w_official_diversity: float = 2.0,
        w_column_coverage: float = 0.35,
        w_group_coverage: float = 0.30,
        w_structure: float = 0.25,
        empty_penalty: float = -1.0,
        repeat_penalty: float = -1.0,
    ):
        self.schema_name, self.dataset_name = dataset_enum(schema, dataset_number)
        self.dataset_meta = DatasetMeta(self.schema_name, self.dataset_name)
        self.dataset = Dataset(self.dataset_meta)
        from atena.simulation.actions_simulator import ActionsSimulator

        self.simulator = ActionsSimulator(self.dataset)
        self.action_space = AtenaActionSpace(
            self.dataset,
            max_terms_per_column=max_terms_per_column,
            seed=seed,
            avp_terms=load_avp(schema, dataset_number, avp),
        )
        self.episode_length = int(episode_length)
        self.w_kl = float(w_kl)
        self.w_compaction = float(w_compaction)
        self.w_official_diversity = float(w_official_diversity)
        self.w_column_coverage = float(w_column_coverage)
        self.w_group_coverage = float(w_group_coverage)
        self.w_structure = float(w_structure)
        self.empty_penalty = float(empty_penalty)
        self.repeat_penalty = float(repeat_penalty)
        self.rng = np.random.default_rng(int(seed))
        self.state_dim = self._feature_dim()
        self.reset()

    @property
    def action_dim(self) -> int:
        return len(self.action_space)

    def reset(self):
        self.simulator.reset()
        self.actions = []
        self.step_index = 0
        self.displays = list(self.simulator.simulation_state.displays_history)
        self.display_strings = {str(self.displays[-1])}
        self.prev_action_kind = "START"
        self.used_action_columns = set()
        self.used_group_columns = set()
        self.action_kind_history = []
        return self._encode_current_state()

    def legal_action_mask(self) -> np.ndarray:
        mask = np.ones(self.action_dim, dtype=np.float32)
        if len(self.simulator.simulation_state.states_stack) <= 1:
            mask[0] = 0.0
        return mask

    def step(self, action_index: int):
        action_index = int(action_index)
        mask = self.legal_action_mask()
        if action_index < 0 or action_index >= self.action_dim or mask[action_index] <= 0:
            action_index = int(self.rng.choice(np.flatnonzero(mask > 0)))

        state_before = self._encode_current_state()
        action = self.action_space.get(action_index)
        prev_display = self.simulator.simulation_state.displays_history[-1]
        step_info = self.simulator.execute_action(action)
        current_display = step_info.display
        self.actions.append(action)
        self.step_index += 1
        self.displays.append(current_display)

        reward_info = self._compute_rewards(action_index, prev_display, current_display, step_info)
        task_reward = self._combine_reward(reward_info)
        done = self.step_index >= self.episode_length
        next_state = self._encode_current_state()

        return next_state, float(task_reward), done, {
            **reward_info,
            "action_index": action_index,
            "action_kind": self.action_space.kind(action_index),
            "action_repr": repr(action),
            "state_before": state_before,
            "state_after": next_state,
        }

    def preview_step(self, action_index: int) -> Tuple[float, Dict[str, object]]:
        action_index = int(action_index)
        mask = self.legal_action_mask()
        if action_index < 0 or action_index >= self.action_dim or mask[action_index] <= 0:
            raise ValueError(f"Action index {action_index} is not legal in the current state")

        snapshot = self._snapshot_mutable_state()
        try:
            _, reward, _, info = self.step(action_index)
            return float(reward), dict(info)
        finally:
            self._restore_mutable_state(snapshot)

    def _snapshot_mutable_state(self) -> Dict[str, object]:
        simulation_state = self.simulator.simulation_state
        return {
            "states_history": list(simulation_state.states_history),
            "states_stack": list(simulation_state.states_stack),
            "displays_history": list(simulation_state.displays_history),
            "actions": list(self.actions),
            "step_index": int(self.step_index),
            "displays": list(self.displays),
            "display_strings": set(self.display_strings),
            "prev_action_kind": self.prev_action_kind,
            "used_action_columns": set(self.used_action_columns),
            "used_group_columns": set(self.used_group_columns),
            "action_kind_history": list(self.action_kind_history),
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }

    def _restore_mutable_state(self, snapshot: Mapping[str, object]) -> None:
        simulation_state = self.simulator.simulation_state
        simulation_state.states_history = list(snapshot["states_history"])
        simulation_state.states_stack = list(snapshot["states_stack"])
        simulation_state.displays_history = list(snapshot["displays_history"])
        self.actions = list(snapshot["actions"])
        self.step_index = int(snapshot["step_index"])
        self.displays = list(snapshot["displays"])
        self.display_strings = set(snapshot["display_strings"])
        self.prev_action_kind = str(snapshot["prev_action_kind"])
        self.used_action_columns = set(snapshot["used_action_columns"])
        self.used_group_columns = set(snapshot["used_group_columns"])
        self.action_kind_history = list(snapshot["action_kind_history"])
        self.rng.bit_generator.state = deepcopy(snapshot["rng_state"])

    def sample_legal_action(self) -> int:
        valid = np.flatnonzero(self.legal_action_mask() > 0)
        return int(self.rng.choice(valid))

    def _compute_rewards(self, action_index, prev_display, current_display, step_info):
        action_kind = self.action_space.kind(action_index)
        action = self.action_space.get(action_index)
        filtered_df = step_info._filtered_df
        aggregated_df = step_info._aggregated_df
        display_key = str(current_display)

        is_grouped = bool(step_info.state.is_grouped())
        invalid = len(filtered_df) == 0 or (is_grouped and aggregated_df is None)
        repeated = display_key in self.display_strings
        self.display_strings.add(display_key)

        if action_kind != "BACK" and not invalid:
            self.prev_action_kind = action_kind

        if repeated and action_kind != "BACK":
            official_diversity = self.repeat_penalty
        else:
            distances = [
                display_distance(history_display, current_display).display_distance
                for history_display in self.displays[:-1]
            ]
            official_diversity = float(min(distances)) if distances else 1.0

        official_kl, official_compaction = self._official_reward_components(
            action_kind, step_info, invalid
        )
        if invalid:
            official_diversity = self.empty_penalty
        column_coverage, group_coverage, structure = self._action_quality_rewards(
            action_kind,
            action,
            invalid,
        )

        return {
            "official_kl": float(official_kl),
            "official_compaction": float(official_compaction),
            "official_diversity": float(official_diversity),
            "column_coverage": float(column_coverage),
            "group_coverage": float(group_coverage),
            "structure": float(structure),
            "invalid": float(invalid),
            "repeated": float(repeated),
        }

    def _combine_reward(self, reward_info):
        return (
            self.w_kl * reward_info.get("official_kl", 0.0)
            + self.w_compaction * reward_info.get("official_compaction", 0.0)
            + self.w_official_diversity * reward_info.get("official_diversity", 0.0)
            + self.w_column_coverage * reward_info.get("column_coverage", 0.0)
            + self.w_group_coverage * reward_info.get("group_coverage", 0.0)
            + self.w_structure * reward_info.get("structure", 0.0)
        )

    def _action_column(self, action):
        if isinstance(action, FilterAction):
            return str(action.filtered_column)
        if isinstance(action, GroupAction):
            return str(action.grouped_column)
        return ""

    def _action_quality_rewards(self, action_kind: str, action, invalid: bool):
        column = self._action_column(action)
        if invalid:
            self.action_kind_history.append((action_kind, column))
            return 0.0, 0.0, -1.0

        column_coverage = 0.0
        group_coverage = 0.0
        if action_kind in {"FILTER", "GROUP"} and column:
            if column in self.used_action_columns:
                column_coverage = -0.25
            else:
                column_coverage = 1.0
                self.used_action_columns.add(column)

        if action_kind == "GROUP" and column:
            if column in self.used_group_columns:
                group_coverage = 0.10
            else:
                group_coverage = 1.0
                self.used_group_columns.add(column)

        structure = self._structure_reward(action_kind, column)
        self.action_kind_history.append((action_kind, column))
        return column_coverage, group_coverage, structure

    def _structure_reward(self, action_kind: str, column: str):
        history = self.action_kind_history
        if action_kind == "GROUP":
            return 0.20
        if action_kind == "BACK":
            return 0.0
        if len(history) >= 2 and history[-2][0] == "FILTER" and history[-1][0] == "BACK":
            return -1.0 if history[-2][1] == column else -0.60
        if history and history[-1][0] == "BACK":
            return -0.30
        return 0.0

    def _official_reward_components(self, action_kind: str, step_info, invalid: bool):
        if action_kind == "BACK" or invalid:
            return 0.0, 0.0
        if action_kind == "FILTER":
            return self._official_kl(step_info), 0.0
        if action_kind == "GROUP":
            return 0.0, self._official_compaction(step_info)
        return 0.0, 0.0

    def _official_kl(self, step_info):
        previous_df = self._previous_filtered_df()
        current_df = step_info._filtered_df
        if len(previous_df) <= 0 or len(current_df) <= 0:
            return self.empty_penalty
        scores = []
        epsilon = 0.2 / max(len(previous_df), 1)
        for column in self.dataset.columns:
            p_counts = self._value_counts(previous_df[column])
            q_counts = self._value_counts(current_df[column])
            keys = set(p_counts) | set(q_counts)
            p_total = sum(p_counts.values()) + (epsilon * len(keys))
            q_total = sum(q_counts.values()) + (epsilon * len(keys))
            kl = 0.0
            for key in keys:
                p = (p_counts.get(key, 0.0) + epsilon) / p_total
                q = (q_counts.get(key, 0.0) + epsilon) / q_total
                kl += p * math.log(p / q)
            scores.append(kl)
        if not scores:
            return 0.0
        return float(1.0 / (1.0 + math.exp(-((max(scores) / 2.0) - 3.0))))

    def _official_compaction(self, step_info):
        if not step_info.state.is_grouped() or step_info._aggregated_df is None:
            return 0.0
        groups = len(step_info._aggregated_df)
        if groups <= 1:
            return self.empty_penalty
        filtered_rows = max(len(step_info._filtered_df), 1)
        grouped_columns = max(len(step_info.state["grouping"]), 1)
        denominator_epsilon = 1e-5
        compact_display = self._normalized_sigmoid(
            center=0.5,
            steepness=17.0,
            value=1.0 - 1.0 / math.log(8.0 + (groups * grouped_columns) + denominator_epsilon, 8.0),
        )
        compact_data = 1.0 - self._normalized_sigmoid(
            center=0.5,
            steepness=17.0,
            value=1.0 - 1.0 / math.log(7.0 + filtered_rows + denominator_epsilon, 7.0),
        )
        return float(compact_display * compact_data)

    def _previous_filtered_df(self):
        if not self.actions:
            return self.dataset.dataset_df
        previous_actions = self.actions[:-1]
        from atena.simulation.actions_simulator import ActionsSimulator

        simulator = ActionsSimulator(self.dataset)
        if not previous_actions:
            return self.dataset.dataset_df
        steps = simulator.run_actions(previous_actions)
        if not steps:
            return self.dataset.dataset_df
        return steps[-1]._filtered_df

    @staticmethod
    def _value_counts(series):
        counts = Counter()
        for value in series:
            counts[str(value)] += 1
        return counts

    @staticmethod
    def _normalized_sigmoid(center: float, steepness: float, value: float):
        raw = 1.0 / (1.0 + math.exp(-float(steepness) * (float(value) - float(center))))
        low = 1.0 / (1.0 + math.exp(float(steepness) * float(center)))
        high = 1.0 / (1.0 + math.exp(-float(steepness) * (1.0 - float(center))))
        if abs(high - low) <= 1e-12:
            return raw
        return (raw - low) / (high - low)

    def _feature_dim(self):
        columns = len(self.dataset.columns)
        return (columns * 3) + 5 + 3 + 1

    def _encode_current_state(self):
        display = self.simulator.simulation_state.displays_history[-1]
        data_layer = display["data_layer"]
        features: List[float] = []
        for column in self.dataset.columns:
            col = data_layer[column]
            features.extend([
                float(col["unique"]),
                float(col["nulls"]),
                float(col["entropy"]),
            ])

        gran = display["granularity_layer"]
        if gran is None:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([
                float(len(gran["group_attrs"]) / max(len(self.dataset.columns), 1)),
                float(len(gran["agg_attrs"]) / max(len(self.dataset.primary_key_columns), 1)),
                float(gran["inverse_ngroups"]),
                float(gran["site_std"]),
                float(gran["inverse_size_mean"]),
            ])

        action_one_hot = [0.0, 0.0, 0.0]
        if self.prev_action_kind == "BACK":
            action_one_hot[0] = 1.0
        elif self.prev_action_kind == "FILTER":
            action_one_hot[1] = 1.0
        elif self.prev_action_kind == "GROUP":
            action_one_hot[2] = 1.0
        features.extend(action_one_hot)
        features.append(float(self.step_index / max(self.episode_length, 1)))
        return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=5.0, neginf=-5.0)


def make_env(schema, dataset_number, seed, args):
    return AtenaEDAEnv(
        schema=schema,
        dataset_number=dataset_number,
        episode_length=args.episode_length,
        max_terms_per_column=args.max_terms_per_column,
        seed=seed,
        avp=getattr(args, "avp", "0"),
        w_kl=getattr(args, "w_kl", 1.5),
        w_compaction=getattr(args, "w_compaction", 2.0),
        w_official_diversity=getattr(args, "w_official_diversity", 2.0),
        w_column_coverage=getattr(args, "w_column_coverage", 0.35),
        w_group_coverage=getattr(args, "w_group_coverage", 0.30),
        w_structure=getattr(args, "w_structure", 0.25),
        empty_penalty=args.empty_penalty,
        repeat_penalty=args.repeat_penalty,
    )
