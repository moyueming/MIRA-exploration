#!/usr/bin/env python3
"""Download and prepare the canonical UCI Covertype CSV."""

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "covertype.csv"
DEFAULT_SHA256 = "a07902ee1c9d3231c6655f23e6f75a6797d0ba26a2359f533c2c0e65d05c9bd4"
HEADER = (
    "Elevation,Aspect,Slope,Horizontal_Distance_To_Hydrology,"
    "Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,"
    "Hillshade_9am,Hillshade_Noon,Hillshade_3pm,"
    "Horizontal_Distance_To_Fire_Points,"
    + ",".join("Wilderness_Area{}".format(index) for index in range(1, 5))
    + ","
    + ",".join("Soil_Type{}".format(index) for index in range(1, 41))
    + ",Cover_Type"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "MIRA-dataset-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def prepare_dataset(archive_path, output_path, expected_sha256):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".part")

    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.rsplit("/", 1)[-1] == "covtype.data"]
            if len(members) != 1:
                raise ValueError("Archive must contain exactly one covtype.data file")
            with archive.open(members[0]) as source, partial_path.open("wb") as output:
                output.write((HEADER + "\n").encode("ascii"))
                shutil.copyfileobj(source, output)

        actual_sha256 = sha256_file(partial_path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise ValueError(
                "Checksum mismatch: expected {}, got {}".format(expected_sha256, actual_sha256)
            )
        partial_path.replace(output_path)
        return actual_sha256
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Covertype ZIP URL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV path")
    parser.add_argument(
        "--expected-sha256",
        default=DEFAULT_SHA256,
        help="Expected SHA-256 of the prepared CSV",
    )
    return parser


def main():
    args = build_parser().parse_args()
    with tempfile.TemporaryDirectory(prefix="mira-covertype-") as temp_dir:
        archive_path = Path(temp_dir) / "covertype.zip"
        print("Downloading {}".format(args.url))
        download_archive(args.url, archive_path)
        checksum = prepare_dataset(archive_path, args.output.resolve(), args.expected_sha256)
    print("Wrote {}".format(args.output.resolve()))
    print("SHA-256 {}".format(checksum))


if __name__ == "__main__":
    main()
