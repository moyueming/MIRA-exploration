from typing import Dict, List


AVP_VALUE = "0"


def avp_enabled(avp) -> bool:
    return False


def load_avp(schema: str, dataset_number: int, avp) -> Dict[str, List[str]]:
    return {}


def avp_manifest(schema: str, dataset_number: int, avp) -> Dict[str, object]:
    return {
        "requested": AVP_VALUE,
        "available": False,
        "active": False,
        "schema": schema.lower(),
        "dataset": int(dataset_number),
        "source_kind": None,
        "source_path": None,
        "terms": {},
        "sha256": None,
    }
