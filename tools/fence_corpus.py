#!/usr/bin/env python3
"""Build the fence corpus: every raw model output this project has banked,
one record per output, with provenance and the labels the source
artifact itself carries.

    python3 tools/fence_corpus.py            # build + manifest
    python3 tools/fence_corpus.py --check    # rebuild in memory, compare SHA

Preregistration: docs/research/ADDENDUM_U_PROTOCOL.md (frozen 2026-09-04).

Rules the builder enforces rather than documents:
  * an artifact is included only if its `predictions` map holds a STRING
    per document (raw text); a dict/list/None value aborts the build;
  * family / size / adapted / corpus come from the registry below; regime,
    k, decoder, dtype and model id come from the artifact's own fields,
    and any contradiction between the two aborts the build;
  * registry entries whose file is absent are listed in the manifest as
    absent -- never silently dropped;
  * the output is deterministic (sorted artifacts, sorted document ids)
    so its SHA-256 is a stable fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
DEFAULT_TAG = "2026-09-04"


def paths(tag: str = DEFAULT_TAG) -> tuple[pathlib.Path, pathlib.Path]:
    """(corpus, manifest) for a dated corpus. The first corpus (2026-09-04,
    four families) stays as banked; a later tag is a second corpus that
    includes what has landed since, published beside the first."""
    return (EXP / f"fence_corpus_{tag}.jsonl", EXP / f"fence_corpus_{tag}.manifest.json")


CORPUS, MANIFEST = paths()

# (relative path, family, size, adapted, corpus, expected-regime-source)
# The last column names WHICH artifact field carries the regime so the
# builder reads it from there and never guesses from a filename.
REGISTRY = [
    # Addendum M/N/R: Qwen scale rungs and fence-dose arms on waybills
    ("waybill_scale_rung_0.5b_kshot_2026-08-25.json", "qwen2.5", "0.5B", False, "waybill", "mode"),
    ("waybill_scale_rung_0.5b_schema_2026-08-25.json", "qwen2.5", "0.5B", False, "waybill", "mode"),
    ("waybill_fence_dose_k1_2026-08-25.json", "qwen2.5", "0.5B", False, "waybill", "mode"),
    ("waybill_fence_dose_k1schema_2026-08-25.json", "qwen2.5", "0.5B", False, "waybill", "mode"),
    ("waybill_fence_dose_k3_2026-08-25.json", "qwen2.5", "0.5B", False, "waybill", "mode"),
    ("waybill_scale_rung_1.5b_kshot_RAW_2026-08-25.json", "qwen2.5", "1.5B", False, "waybill", "mode"),
    ("waybill_scale_rung_1.5b_kshot_bf16_2026-08-25.json", "qwen2.5", "1.5B", False, "waybill", "mode"),
    ("waybill_scale_rung_1.5b_schema_REPRO_2026-08-25.json", "qwen2.5", "1.5B", False, "waybill", "mode"),
    ("waybill_scale_rung_3b_kshot_bf16_2026-08-25.json", "qwen2.5", "3B", False, "waybill", "mode"),
    ("waybill_scale_rung_3b_schema_2026-08-25.json", "qwen2.5", "3B", False, "waybill", "mode"),
    ("waybill_scale_rung_3b_schema_bf16_control_2026-08-25.json", "qwen2.5", "3B", False, "waybill", "mode"),
    # Addendum Q / ladder E5: adapted 3B on waybills
    ("waybill_adapted_3b_2026-08-25.json", "qwen2.5", "3B", True, "waybill", "serving"),
    ("ladder_e5_3b_adapted_kshot_2026-08-31.json", "qwen2.5", "3B", True, "waybill", "serving"),
    # Ladder E6-E9 on CORD
    ("ladder_e6_cord_prompted_2026-08-31.json", "qwen2.5", "3B", False, "cord", "arm"),
    ("ladder_e6_cord_adapted_2026-08-31.json", "qwen2.5", "3B", True, "cord", "arm"),
    ("ladder_e7_cord_prompted_2026-09-02.json", "qwen2.5", "3B", False, "cord", "arm"),
    ("ladder_e7_cord_adapted_2026-09-02.json", "qwen2.5", "3B", True, "cord", "arm"),
    ("ladder_e8_cord_prompted_2026-09-03.json", "qwen2.5", "3B", False, "cord", "arm"),
    ("ladder_e8_cord_adapted_2026-09-03.json", "qwen2.5", "3B", True, "cord", "arm"),
    ("ladder_e9_cord_prompted_greedy_2026-09-03.json", "qwen2.5", "3B", False, "cord", "arm"),
    ("ladder_e9_cord_prompted_2026-09-03.json", "qwen2.5", "3B", False, "cord", "arm"),
    ("ladder_e9_cord_adapted_2026-09-03.json", "qwen2.5", "3B", True, "cord", "arm"),
    # Addendum S cells on CORD
    ("cord_fence_tax_cells/0.5b_schema.json", "qwen2.5", "0.5B", False, "cord", "regime"),
    ("cord_fence_tax_cells/0.5b_kshot.json", "qwen2.5", "0.5B", False, "cord", "regime"),
    ("cord_fence_tax_cells/1.5b_schema.json", "qwen2.5", "1.5B", False, "cord", "regime"),
    ("cord_fence_tax_cells/1.5b_kshot.json", "qwen2.5", "1.5B", False, "cord", "regime"),
    # Addendum T cells on CORD
    ("cord_fence_tax_families_cells/smollm2-1.7b_schema.json", "smollm2", "1.7B", False, "cord", "regime"),
    ("cord_fence_tax_families_cells/smollm2-1.7b_kshot.json", "smollm2", "1.7B", False, "cord", "regime"),
    ("cord_fence_tax_families_cells/granite-2b_schema.json", "granite", "2B", False, "cord", "regime"),
    ("cord_fence_tax_families_cells/granite-2b_kshot.json", "granite", "2B", False, "cord", "regime"),
    ("cord_fence_tax_families_cells/phi3-mini_schema.json", "phi3", "3.8B", False, "cord", "regime"),
    ("cord_fence_tax_families_cells/phi3-mini_kshot.json", "phi3", "3.8B", False, "cord", "regime"),
    ("cord_fence_tax_families_cells/falcon3-1b_schema.json", "falcon3", "1B", False, "cord", "regime"),
    ("cord_fence_tax_families_cells/falcon3-1b_kshot.json", "falcon3", "1B", False, "cord", "regime"),
]

_FAMILY_OF_MODEL = {
    "Qwen/Qwen2.5-0.5B-Instruct": ("qwen2.5", "0.5B"),
    "Qwen/Qwen2.5-1.5B-Instruct": ("qwen2.5", "1.5B"),
    "Qwen/Qwen2.5-3B-Instruct": ("qwen2.5", "3B"),
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": ("smollm2", "1.7B"),
    "ibm-granite/granite-3.1-2b-instruct": ("granite", "2B"),
    "microsoft/Phi-3-mini-4k-instruct": ("phi3", "3.8B"),
    "tiiuae/Falcon3-1B-Instruct": ("falcon3", "1B"),
}


class CorpusError(RuntimeError):
    pass


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _labels(rec: dict, source_field: str, rel: str) -> dict:
    """Regime / k / decoder / dtype from the artifact's own fields."""
    if source_field == "mode":
        mode = rec["mode"]
        k = int(rec.get("n_demonstrations", 0))
        if mode == "schema":
            regime, k = "schema", 0
        elif mode == "kshot":
            regime = "kshot"
        elif mode == "schema_kshot":
            regime = "schema_kshot"
        else:
            raise CorpusError(f"{rel}: unknown mode {mode!r}")
        decoder = "greedy"
    elif source_field == "serving":
        serving = rec["serving"]
        if serving.startswith("document-only"):
            regime, k = "document_only", 0
        elif "k=20" in serving:
            regime, k = "kshot", 20
        else:
            raise CorpusError(f"{rel}: unrecognised serving {serving[:60]!r}")
        decoder = "greedy"
    elif source_field == "arm":
        arm = rec["arm"]
        k = int(rec.get("k", 20))
        regime = "kshot"
        if "constrained" in arm or "constrained" in str(rec.get("decoder", "")):
            decoder = "constrained"
        elif rec.get("decoder") == "greedy" or str(rec.get("decode", "")).startswith("greedy"):
            decoder = "greedy"
        else:
            raise CorpusError(f"{rel}: cannot tell the decoder from arm={arm!r} "
                              f"decoder={rec.get('decoder')!r} decode={rec.get('decode')!r}")
    elif source_field == "regime":
        regime = rec["regime"]
        if regime not in ("schema", "kshot"):
            raise CorpusError(f"{rel}: unknown regime {regime!r}")
        k = int(rec.get("k", 0))
        decoder = "greedy"
        if not str(rec.get("decode", "")).startswith("greedy"):
            raise CorpusError(f"{rel}: decode field is not greedy: {rec.get('decode')!r}")
    else:  # pragma: no cover
        raise CorpusError(f"{rel}: unknown label source {source_field!r}")
    dtype = rec.get("dtype")
    if dtype is None:
        decode = str(rec.get("decode", ""))
        dtype = "float32" if "float32" in decode else ("bfloat16" if "bfloat16" in decode else None)
    if dtype not in ("float32", "bfloat16"):
        raise CorpusError(f"{rel}: dtype unknown ({dtype!r})")
    return {"regime": regime, "k": k, "decoder": decoder, "dtype": dtype}


def build(only: set | None = None) -> tuple[list[dict], dict]:
    """Build from every registry artifact present on disk, or -- with `only`,
    the set of artifact paths a banked manifest lists as present -- from
    exactly those, so an earlier corpus stays reproducible after later
    artifacts land (an artifact in `only` that is missing is an error)."""
    records: list[dict] = []
    present, absent = [], []
    for rel, family, size, adapted, corpus, source_field in REGISTRY:
        path = EXP / rel
        if only is not None and rel not in only:
            absent.append(rel)
            continue
        if not path.exists():
            if only is not None:
                raise CorpusError(f"{rel}: listed present in the manifest but missing on disk")
            absent.append(rel)
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        preds = rec.get("predictions")
        if not isinstance(preds, dict) or not preds:
            raise CorpusError(f"{rel}: no predictions map")
        model = rec.get("model")
        if _FAMILY_OF_MODEL.get(model) != (family, size):
            raise CorpusError(f"{rel}: model {model!r} contradicts registry {family}/{size}")
        labels = _labels(rec, source_field, rel)
        sha = _sha256_file(path)
        stem = pathlib.Path(rel).with_suffix("").as_posix()
        n = 0
        for doc_id in sorted(preds):
            text = preds[doc_id]
            if not isinstance(text, str):
                raise CorpusError(f"{rel}: prediction for {doc_id} is {type(text).__name__}, "
                                  "not raw text -- refusing to build from a parsed object")
            records.append({
                "id": f"{stem}:{doc_id}",
                "artifact": rel, "artifact_sha256": sha, "doc_id": doc_id,
                "model": model, "family": family, "size": size,
                "adapted": adapted, "corpus": corpus, **labels, "text": text,
            })
            n += 1
        present.append({"artifact": rel, "sha256": sha, "n": n, "family": family,
                        "size": size, "adapted": adapted, "corpus": corpus, **labels})
    if not records:
        raise CorpusError("no artifacts present -- nothing to build")
    manifest = {
        "what": "Fence corpus: every raw model output banked in this repository, "
                "one record per output, provenance and the source artifact's own "
                "labels. Built by tools/fence_corpus.py; preregistered in "
                "docs/research/ADDENDUM_U_PROTOCOL.md.",
        "n_records": len(records),
        "artifacts_present": present,
        "artifacts_absent": absent,
        "families_present": sorted({r["family"] for r in records}),
        "by_slice": _slice_counts(records),
    }
    return records, manifest


def _slice_key(r: dict) -> str:
    return "|".join([r["family"], r["size"], "adapted" if r["adapted"] else "prompted",
                     r["corpus"], r["regime"], f"k={r['k']}", r["decoder"]])


def _slice_counts(records: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in records:
        out[_slice_key(r)] = out.get(_slice_key(r), 0) + 1
    return dict(sorted(out.items()))


def serialize(records: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="rebuild from the banked manifest's artifact list and compare SHA")
    parser.add_argument("--tag", default=DEFAULT_TAG,
                        help="date tag of the corpus to build or check (default: the first corpus)")
    args = parser.parse_args()
    corpus_path, manifest_path = paths(args.tag)
    if args.check:
        if not corpus_path.exists() or not manifest_path.exists():
            print(f"no banked corpus {args.tag} to check against"); return 2
        listed = {a["artifact"] for a in json.loads(manifest_path.read_text())["artifacts_present"]}
        records, manifest = build(only=listed)
    else:
        records, manifest = build()
    text = serialize(records)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest["corpus_sha256"] = sha
    manifest["tag"] = args.tag
    if args.check:
        banked = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        if banked != sha:
            print(f"DRIFT: banked {banked[:12]} != rebuilt {sha[:12]} "
                  f"(absent now: {manifest['artifacts_absent']})"); return 1
        print(f"corpus {args.tag} current: {len(records)} records, sha256 {sha[:12]}"); return 0
    corpus_path.write_text(text, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"built {len(records)} records from {len(manifest['artifacts_present'])} artifacts "
          f"({len(manifest['artifacts_absent'])} absent: {manifest['artifacts_absent']}); "
          f"families {manifest['families_present']}; sha256 {sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
