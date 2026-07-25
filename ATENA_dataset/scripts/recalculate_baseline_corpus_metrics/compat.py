import pandas as pd


def enable_legacy_pandas():
    try:
        pd.set_option("future.infer_string", False)
    except (KeyError, ValueError):
        pass
    if not hasattr(pd.Series, "iteritems"):
        pd.Series.iteritems = pd.Series.items
