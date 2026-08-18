"""Fetch CORD-v2 ground-truth JSON (text columns only — never the images).

The dataset's parquet files embed receipt images, making whole-file reads
~500 MB; column pruning via pyarrow reads just the ``ground_truth`` strings
(a few MB). Output: JSONL with one {"gt_parse": ..., "valid_line": ...} per line, the
input the ENTERPRISE_EVAL_SPEC's from_cord_gt adapter consumes.

    python scripts/fetch_cord.py --split train --limit 100 --out cord_train.jsonl

License: CORD is CC BY 4.0 (https://github.com/clovaai/cord); this script
downloads from the Hugging Face parquet mirror of naver-clova-ix/cord-v2.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/datasets/naver-clova-ix/cord-v2/parquet/default"


def parquet_urls(split: str) -> list[str]:
    with urllib.request.urlopen(f"{API}/{split}") as response:
        urls = json.load(response)
    if not isinstance(urls, list) or not urls:
        raise RuntimeError(f"no parquet files listed for split {split!r}")
    return urls


def fetch_ground_truth(split: str, limit: int, cache_dir: Path) -> list[dict]:
    """Download each parquet whole, prune to ground_truth, delete the file.

    Remote range-reads fail here: HF's CDN issues single-use signed URLs
    whose policy pins the exact byte range of the resolving request, so
    seekable remote access is off the table. The files embed receipt images
    (train ~500 MB each, validation/test ~240 MB), so each is downloaded,
    column-pruned locally (ground_truth is ~30 KB total), and removed
    before the next one — peak disk usage is one file.
    """

    import pyarrow.parquet as pq

    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for url in parquet_urls(split):
        if len(records) >= limit:
            break
        local = cache_dir / f"cord_{split}_{Path(url).name}"
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, local)  # noqa: S310 - pinned host
        try:
            table = pq.read_table(local, columns=["ground_truth"])
        finally:
            local.unlink(missing_ok=True)
        for value in table.column("ground_truth").to_pylist():
            payload = json.loads(value)
            if "gt_parse" not in payload:
                continue
            records.append(
                {
                    "gt_parse": payload["gt_parse"],
                    "valid_line": payload.get("valid_line", []),
                }
            )
            if len(records) >= limit:
                break
    if not records:
        raise RuntimeError("no ground_truth records extracted")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train",
                        choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default="/tmp/cord_cache")
    args = parser.parse_args()

    records = fetch_ground_truth(args.split, args.limit, Path(args.cache_dir))
    out = Path(args.out)
    with out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
