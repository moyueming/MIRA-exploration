from . import EXPECTED_DATASETS, METRICS, mean_metric_rows, validate_dataset_keys


def deterministic_summary(saved_rows, corpus_row):
    result = {name: float(corpus_row[name]) for name in METRICS}
    result["Precision"] = _mean(saved_rows, "Precision")
    result["EDA-Sim"] = _mean(saved_rows, "EDA-Sim")
    return result


def random_summary(corpus_rows):
    return mean_metric_rows(corpus_rows)


def ordered_sessions(sessions):
    validate_dataset_keys(sessions)
    ordered = []
    for key in sorted(EXPECTED_DATASETS):
        actions = sessions[key]
        if len(actions) != 12:
            raise ValueError(f"{key} expected 12 actions, got {len(actions)}")
        ordered.append((key, actions))
    return ordered


def apply_saved_scalar_metrics(runtime_row, saved_rows):
    result = {name: float(runtime_row[name]) for name in METRICS}
    result["Precision"] = _mean(saved_rows, "Precision")
    result["EDA-Sim"] = _mean(saved_rows, "EDA-Sim")
    return result


def _mean(rows, metric):
    if not rows:
        raise ValueError(f"cannot average zero {metric} rows")
    return sum(float(row[metric]) for row in rows) / len(rows)
