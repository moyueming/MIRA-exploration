import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mira import avp
from mira import env as standalone_env
from mira.avp_loader import load_avp


EXPECTED_AVP_DIGESTS = {
    ("cyber", 1): "8f30b54cd5be6263ba246126f2f7c75b6d42013b5458893b0c5af2f93018346e",
    ("cyber", 2): "34f2af5ba9337709fc89ea96f7f83833f8df49065edf006c0de2f90e9c224315",
    ("cyber", 3): "118d7382096ed9736b8a843ce920096f86f823fbfa726fe3b93e10fa6b765a0f",
    ("cyber", 4): "729715974f3052b3482269b41e1d64aecd2351697f6497d2e89b202bee73c776",
    ("flights", 1): "f870d5b93c8c9e2d4f6a1bdb93ec829833895b7eb624efbe35dd2a4e9b42f732",
    ("flights", 2): "252912ca8b24693a322cd8b808973eaa5fe83375984a595b58941e5df50ec5cf",
    ("flights", 3): "8099b041c0dd7490c8f0cc34bd4f6648d61a7fe4676bb6f92a3b0dd1fe53c638",
    ("flights", 4): "5d5b3a794aa1399e5f695e8959376c9bc293016a4b9cbf78fddd07f5742982b9",
}


class FakeDataset:
    columns = ["ip_dst", "highest_layer"]
    primary_key_columns = ["packet_number"]
    dataset_meta = SimpleNamespace(
        schema=SimpleNamespace(value="cyber"),
        dataset_name=SimpleNamespace(value=1),
    )

    class FakeSeries:
        def __init__(self, values):
            self.values = list(values)

        def astype(self, _dtype):
            return self

        def value_counts(self, dropna=False):
            del dropna
            counts = {}
            for value in self.values:
                counts[str(value)] = counts.get(str(value), 0) + 1
            return counts

        def iteritems(self):
            return iter(enumerate(self.values))

    class FakeFrame:
        def __init__(self):
            self.columns = {
                "ip_dst": ["popular", "popular", "82.108.87.7"],
                "highest_layer": ["TCP", "TCP", "ARP"],
            }

        def __len__(self):
            return 3

        def __getitem__(self, column):
            return FakeDataset.FakeSeries(self.columns[column])

    dataset_df = FakeFrame()


class AvpContractTests(unittest.TestCase):
    def test_enabled_avp_matches_frozen_vocabulary_digests(self):
        for (schema, dataset), expected_digest in EXPECTED_AVP_DIGESTS.items():
            with self.subTest(schema=schema, dataset=dataset):
                self.assertEqual(avp.avp_details(schema, dataset)["sha256"], expected_digest)
                self.assertEqual(load_avp(schema, dataset, "1"), avp.avp_terms(schema, dataset))

    def test_avp_terms_are_appended_after_frequency_cap(self):
        space = standalone_env.AtenaActionSpace(
            FakeDataset(),
            max_terms_per_column=1,
            avp_terms={"ip_dst": ["82.108.87.7", "popular"]},
        )

        self.assertEqual(space.column_term_map["ip_dst"], ["popular", "82.108.87.7"])
        self.assertTrue(any(
            isinstance(action, standalone_env.FilterAction)
            and str(action.filtered_column) == "ip_dst"
            and action.filter_operator is standalone_env.FilterOperator.CONTAINS
            and str(action.filter_term) == "82.108.87.7"
            for action in space.actions
        ))

    def test_avp_details_disclose_source_and_digest(self):
        details = avp.avp_details("cyber", 1)

        self.assertEqual(details["schema"], "cyber")
        self.assertEqual(details["dataset"], 1)
        self.assertEqual(details["source_kind"], "official_evaluator_reference")
        self.assertTrue(details["source_path"].endswith("references/cyber/dataset1.py"))
        self.assertEqual(len(details["sha256"]), 64)
        self.assertEqual(details["terms"], avp.avp_terms("cyber", 1))


if __name__ == "__main__":
    unittest.main()
