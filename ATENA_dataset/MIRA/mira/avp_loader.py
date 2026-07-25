from importlib import import_module
from importlib.util import find_spec
from typing import Dict, List


AVP_MODULE = "mira.avp"


def avp_enabled(avp) -> bool:
    return avp == "1"


def _load_avp_module():
    try:
        return import_module(AVP_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == AVP_MODULE:
            return None
        raise


def load_avp(schema: str, dataset_number: int, avp) -> Dict[str, List[str]]:
    if not avp_enabled(avp):
        return {}
    module = _load_avp_module()
    if module is None:
        return {}
    return module.avp_terms(schema, dataset_number)


def avp_manifest(schema: str, dataset_number: int, avp) -> Dict[str, object]:
    manifest = {
        "requested": avp,
        "available": False,
        "active": False,
        "schema": schema.lower(),
        "dataset": int(dataset_number),
        "source_kind": None,
        "source_path": None,
        "terms": {},
        "sha256": None,
    }
    if not avp_enabled(avp):
        manifest["available"] = find_spec(AVP_MODULE) is not None
        return manifest

    module = _load_avp_module()
    if module is None:
        return manifest

    manifest.update(module.avp_details(schema, dataset_number))
    manifest["available"] = True
    manifest["active"] = True
    return manifest
