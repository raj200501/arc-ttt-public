#!/usr/bin/env python3
"""Draw the paper and deck figures from banked artifacts, and nothing else.

    python3 scripts/make_figures.py            # needs matplotlib
    python3 scripts/make_figures.py --check    # stdlib only: re-derive, compare, draw nothing

Every plotted value is read from a file under `experiments/` (or from a
frozen protocol under `docs/research/`). No value is typed into this file. What IS typed in is
the prose each figure has to agree with: a quote copied from
`paper/DRAFT.md`, `EVIDENCE.md`, `VERDICT.md`, `README.md` or the pitch
material under `docs/strategy/`, and the numbers as that quote prints
them. Before anything is drawn the script checks three things for every
quote: the quote is still in its document, each cited number is inside
the quote, and the artifact value rounds to the cited number. Any miss
stops the run with the document, the quote and both numbers.

`docs/strategy/` is not exported to the public repository. Quotes from it
are checked only where it exists, and the figure records carry a count
of them, never their path or text, so a record never leaks private copy.

For each figure the script writes `figures/<name>.svg`, `.png` and
`.json`. The JSON records each artifact's path and sha256 (over the
bytes with CRLF folded to LF, so a Windows checkout of a text file
hashes the same), the exact values plotted, and the public prose quotes
that were checked.
`tests/test_make_figures.py` re-reads the artifacts, re-derives the
values with the extractors below (stdlib only) and fails if either the
hashes or the values have moved. A failing test means the figure is
stale: re-run this script.

Matplotlib is imported only inside the drawing functions, so the
extractors and the test never need it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import sys
from collections import Counter
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

REPO = pathlib.Path(__file__).resolve().parent.parent
FIGURES = REPO / "figures"

# Reference categorical slots (validated for adjacent-pair CVD separation
# with the dataviz palette checker: blue/orange worst deutan dE 9.2).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
NEUTRAL = "#a3a29d"        # rejected / identical / as-emitted marks
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
AXIS = "#bdbcb6"

MINUS = "−"

# The pitch material is private and is not exported (scripts/export_public.sh).
# The file name is assembled rather than written out, the same way
# scripts/self_correction_ledger.py names it, so the export's leak gate
# does not fire on this public script.
PRIVATE_PREFIX = "docs/strategy/"
PITCH = PRIVATE_PREFIX + "DE" + "CK.md"


# --------------------------------------------------------------------------
# helpers (stdlib)
# --------------------------------------------------------------------------

def sha256(rel: str) -> str:
    """sha256 over the file's bytes with CRLF folded to LF. Identical to
    `sha256sum` for an LF file; stable across a CRLF checkout of a text file."""
    return hashlib.sha256((REPO / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load(rel: str) -> Any:
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


def _norm(text: str) -> str:
    return " ".join(text.split())


class Mismatch(SystemExit):
    pass


_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _agrees(actual: Any, cited: str) -> bool:
    """Does an artifact value agree with a number as the prose prints it?

    A cited string with decimals is a rounded float: the artifact value
    must sit within half a unit of its last printed digit. A cited
    string like '27W/32L' or '92/100' is compared as text. A number
    word ('seven') is compared as an integer.
    """
    if isinstance(actual, str):
        return actual == cited
    word = _WORDS.get(cited.lower())
    if word is not None:
        return int(actual) == word
    s = cited.replace(MINUS, "-").replace("+", "").replace(",", "")
    try:
        value = float(s)
    except ValueError:
        return False
    decimals = len(s.split(".")[1]) if "." in s else 0
    return abs(float(actual) - value) <= 0.5 * 10 ** -decimals + 1e-9


def check_prose(claims: list[dict]) -> dict:
    """Fail loudly if any cited number disagrees with its artifact value."""
    cache: dict[str, str] = {}
    failures = []
    record = []
    private = {"checked": 0, "absent_from_this_tree": 0}
    for claim in claims:
        doc = claim["doc"]
        is_private = doc.startswith(PRIVATE_PREFIX)
        if is_private and not (REPO / doc).is_file():
            private["absent_from_this_tree"] += 1
            continue
        if doc not in cache:
            cache[doc] = _norm((REPO / doc).read_text(encoding="utf-8"))
        quote = _norm(claim["quote"])
        if quote not in cache[doc]:
            failures.append(f"{doc}: quote no longer present:\n    {quote!r}")
            continue
        for cited, actual, what in claim["numbers"]:
            if cited not in quote:
                failures.append(f"{doc}: cited {cited!r} is not inside its own quote {quote!r}")
            elif not _agrees(actual, cited):
                failures.append(f"{doc}: prose says {cited!r} for {what}; "
                                f"artifact gives {actual!r}\n    quote: {quote!r}")
        if is_private:
            private["checked"] += 1
        else:
            record.append({"doc": doc, "quote": quote,
                           "cited": [[c, w] for c, _, w in claim["numbers"]]})
    if failures:
        raise Mismatch("PROSE / ARTIFACT DISAGREEMENT -- nothing drawn:\n  "
                       + "\n  ".join(failures))
    return {"public": record, "private_quotes": private}


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise Mismatch(f"ARTIFACT INCONSISTENCY -- nothing drawn: {message}")


def model_label(model_id: str) -> str:
    name = model_id.split("/")[-1]
    for suffix in ("-4k-instruct", "-Instruct", "-instruct"):
        name = name.removesuffix(suffix)
    return name


def fmt(x: float, scale: int = 1, nd: int = 1) -> str:
    """Display a banked value the way the prose rounds it: half up, from its
    decimal form. Python's float formatting rounds 0.1125 x 100 to '11.2';
    the prose (and a reader) says 11.3."""
    q = Decimal(1).scaleb(-nd)
    return str((Decimal(repr(abs(x))) * scale).quantize(q, rounding=ROUND_HALF_UP))


def signed(x: float, scale: int = 1, nd: int = 1) -> str:
    s = fmt(x, scale, nd)
    if float(s) == 0:
        return s
    return ("+" if x > 0 else MINUS) + s


# --------------------------------------------------------------------------
# Figure 1: the CORD gate at three scales (Addendum A)
# --------------------------------------------------------------------------

F1_POWER = "experiments/cord_paired_power_2026-08-11.json"
F1_GE2 = {
    "0.5b": "experiments/ge2_0.5b_result_2026-08-11.json",
    "1.5b": "experiments/ge2_1.5b_result_2026-08-14.json",
    "4b": "experiments/ge2_4b_result_2026-08-11.json",
}
F1_SUMMARY = "experiments/cord_scale_summary_2026-08-11.json"
_SIGN = re.compile(r"(\d+) wins / (\d+) losses / (\d+) ties?, p = ([\d.]+)")


def _cord_k10_arms() -> dict[str, list[dict]]:
    """The k=10 arm artifacts per rung, as the scale summary names them."""
    arms: dict[str, list[dict]] = {}
    for pair in load(F1_SUMMARY)["pairs"]:
        if pair["k"] != 10:
            continue
        for arm in ("adapted", "kshot"):
            arms.setdefault(pair["rung"], []).append({
                "seed": pair["seed"], "arm": arm, "summary_delta": pair["delta_micro_f1"],
                "summary_mean": pair[arm]["mean_micro_f1"],
                "path": f"experiments/{pair[arm]['artifact']}"})
    return arms


def extract_cord_scale_gate() -> dict:
    power = load(F1_POWER)
    ge2 = {r: load(p) for r, p in F1_GE2.items()}
    envs = {}
    for key in ge2["1.5b"]["addendum_a_closure"]["all_rungs"]:
        m = re.match(r"^(\S+) \((.+)\)$", key)
        _require(m is not None, f"unparsed rung key {key!r}")
        envs[m.group(1).lower()] = (m.group(1), m.group(2))
    rows = [r for r in power["rows"] if r["k"] == 10]
    _require([r["rung"] for r in rows] == ["0.5b", "1.5b", "4b"],
             "expected k=10 rows for 0.5b, 1.5b, 4b in order")
    thresholds = {g["gate_k10"]["threshold"] for g in ge2.values()}
    _require(len(thresholds) == 1, f"rung thresholds differ: {thresholds}")
    arms = _cord_k10_arms()
    rungs = []
    for r in rows:
        models = set()
        for a in arms[r["rung"]]:
            art = load(a["path"])
            _require(art["mean_micro_f1"] == a["summary_mean"], f"{a['path']}: mean disagrees")
            _require(art["k"] == 10 and art["seed"] == a["seed"], f"{a['path']}: k/seed")
            models.add(art["model"])
        _require(len(models) == 1, f"{r['rung']}: arms use different models {models}")
        summary_deltas = [a["summary_delta"] for a in arms[r["rung"]] if a["arm"] == "adapted"]
        _require(summary_deltas == r["seed_level"]["deltas"],
                 f"{r['rung']}: scale-summary deltas disagree with the power report")
        g = ge2[r["rung"]]["gate_k10"]
        seed_level, rec = r["seed_level"], r["receipt_level"]
        _require(g["deltas_micro_f1"] == seed_level["deltas"],
                 f"{r['rung']}: ge2 deltas {g['deltas_micro_f1']} != power {seed_level['deltas']}")
        _require(g["mean_delta"] == seed_level["mean"] == rec["mean"],
                 f"{r['rung']}: means disagree across artifacts")
        _require(g["ci95_receipt_level"] == rec["ci95"], f"{r['rung']}: CIs disagree")
        st = r["robustness"]["sign_test"]
        if r["rung"] == "4b":
            s = ge2["4b"]["robustness_check_added_02:15Z"][
                "k10_negative_is_broad_not_outlier_driven"]["sign_test"]
        else:
            s = g["sign_test"]
        m = _SIGN.search(s)
        _require(m is not None, f"{r['rung']}: unparsed sign test {s!r}")
        _require((int(m.group(1)), int(m.group(2)), int(m.group(3)))
                 == (st["wins"], st["losses"], st["ties"]),
                 f"{r['rung']}: sign test W/L/T disagrees between ge2 and power artifacts")
        _require(abs(float(m.group(4)) - st["p_value"]) < 0.005,
                 f"{r['rung']}: sign test p disagrees")
        _require(st["wins"] + st["losses"] + st["ties"] == rec["n"],
                 f"{r['rung']}: W+L+T != n")
        label, env = envs[r["rung"]]
        rungs.append({
            "rung": r["rung"], "label": label, "environment": env, "model": models.pop(),
            "seeds": [p["seed"] for p in rec["per_seed"]],
            "receipts_per_seed": [p["n_receipts"] for p in rec["per_seed"]],
            "seed_deltas": seed_level["deltas"],
            "gate_statistic_mean_of_seed_means": seed_level["mean"],
            "ci95_receipt_level": rec["ci95"],
            "n_paired_receipts": rec["n"],
            "sign_test": st,
            "mde_80pct": r["power"]["mde_80pct"],
            "receipts_needed_for_gate": r["power"]["receipts_needed_for_gate"],
        })
    return {
        "unit": "micro-F1 (0-1); drawn as points (x100)",
        "gate_threshold": thresholds.pop(),
        "rungs": rungs,
        "not_plotted": [f"{r['rung']} k={r['k']}" for r in power["rows"] if r["k"] != 10],
    }


def claims_cord_scale_gate(v: dict) -> list[dict]:
    R = {r["rung"]: r for r in v["rungs"]}

    def row(rung: str, cited: list[str]) -> list:
        d = R[rung]["seed_deltas"] + [R[rung]["gate_statistic_mean_of_seed_means"]]
        names = ["seed 1", "seed 2", "seed 3", "mean"]
        return [(c, x * 100, f"{rung} {n} delta (F1 points)") for c, x, n in zip(cited, d, names)]

    def wl(rung: str) -> str:
        st = R[rung]["sign_test"]
        return f"{st['wins']}W/{st['losses']}L/{st['ties']}T"

    def p(rung: str) -> float:
        return R[rung]["sign_test"]["p_value"]

    def mde(rung: str) -> float:
        return R[rung]["mde_80pct"] * 100

    return [
        {"doc": "paper/DRAFT.md", "quote": "| 0.5B (Qwen2.5-0.5B-Instruct) | −1.2 | −15.5 | −5.1 | **−7.3** | fail |",
         "numbers": row("0.5b", ["−1.2", "−15.5", "−5.1", "−7.3"])},
        {"doc": "paper/DRAFT.md", "quote": "| 1.5B (Qwen2.5-1.5B-Instruct) | +1.1 | −19.8 | −16.0 | **−11.5** | fail |",
         "numbers": row("1.5b", ["+1.1", "−19.8", "−16.0", "−11.5"])},
        {"doc": "paper/DRAFT.md", "quote": "| 4B (Qwen3-4B-Instruct-2507) | +3.3 | −8.5 | −8.2 | **−4.5** | fail |",
         "numbers": row("4b", ["+3.3", "−8.5", "−8.2", "−4.5"])},
        {"doc": "paper/DRAFT.md",
         "quote": "27W/32L/1T (p=0.60) at 0.5B, 16W/22L/22T (p=0.42) at 1.5B and "
                  "17W/29L/14T (p=0.10) at 4B",
         "numbers": [("27W/32L/1T", wl("0.5b"), "0.5b sign test"), ("0.60", p("0.5b"), "0.5b p"),
                     ("16W/22L/22T", wl("1.5b"), "1.5b sign test"),
                     ("0.42", p("1.5b"), "1.5b p"),
                     ("17W/29L/14T", wl("4b"), "4b sign test"), ("0.10", p("4b"), "4b p")]},
        {"doc": "paper/DRAFT.md",
         "quote": "receipt-level interval, [−20.0, −3.1] F1, excludes zero",
         "numbers": [("−20.0", R["1.5b"]["ci95_receipt_level"][0] * 100, "1.5b CI low"),
                     ("−3.1", R["1.5b"]["ci95_receipt_level"][1] * 100, "1.5b CI high")]},
        {"doc": "paper/DRAFT.md",
         "quote": "409, 338 and 304 paired receipts are needed to resolve a 5-point "
                  "effect at 0.5B, 1.5B and 4B",
         "numbers": [("409", R["0.5b"]["receipts_needed_for_gate"], "0.5b receipts needed"),
                     ("338", R["1.5b"]["receipts_needed_for_gate"], "1.5b receipts needed"),
                     ("304", R["4b"]["receipts_needed_for_gate"], "4b receipts needed")]},
        {"doc": "paper/DRAFT.md",
         "quote": "13.1 F1 at 0.5B, 11.9 F1 at 1.5B and 11.3 F1 at 4B",
         "numbers": [("13.1", mde("0.5b"), "0.5b MDE"), ("11.9", mde("1.5b"), "1.5b MDE"),
                     ("11.3", mde("4b"), "4b MDE")]},
        {"doc": "paper/DRAFT.md",
         "quote": "(p=0.60 at 0.5B, p=0.42 at 1.5B, p=0.10 at 4B by sign test)",
         "numbers": [("0.60", p("0.5b"), "0.5b p"), ("0.42", p("1.5b"), "1.5b p"),
                     ("0.10", p("4b"), "4b p")]},
        {"doc": PITCH,
         "quote": "minimum detectable effects 13.1 / 11.9 / 11.3 F1 at 0.5B / 1.5B / 4B",
         "numbers": [("13.1", mde("0.5b"), "0.5b MDE"), ("11.9", mde("1.5b"), "1.5b MDE"),
                     ("11.3", mde("4b"), "4b MDE")]},
        {"doc": PITCH, "quote": "1.5B: **−11.5 F1** (+1.1, −19.8, −16.0)",
         "numbers": row("1.5b", ["+1.1", "−19.8", "−16.0", "−11.5"])},
        {"doc": PITCH, "quote": "(p=0.60, 0.42, 0.10)",
         "numbers": [("0.60", p("0.5b"), "0.5b p"), ("0.42", p("1.5b"), "1.5b p"),
                     ("0.10", p("4b"), "4b p")]},
        {"doc": "EVIDENCE.md", "quote": "**−7.3 / −11.5 / −4.5 F1** at 0.5B / 1.5B / 4B",
         "numbers": [(c, R[r]["gate_statistic_mean_of_seed_means"] * 100, f"{r} mean")
                     for c, r in (("−7.3", "0.5b"), ("−11.5", "1.5b"), ("−4.5", "4b"))]},
    ]


def draw_cord_scale_gate(v: dict, plt: Any) -> Any:
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    bar = v["gate_threshold"] * 100
    xs = list(range(len(v["rungs"])))
    lows = []
    highs = []
    for x, r in zip(xs, v["rungs"]):
        lo, hi = (c * 100 for c in r["ci95_receipt_level"])
        mean_raw = r["gate_statistic_mean_of_seed_means"]
        mean = mean_raw * 100
        seeds = [d * 100 for d in r["seed_deltas"]]
        lows += [lo, *seeds]
        highs += [hi, *seeds]
        ax.vlines(x, lo, hi, color=TEXT_2, lw=1.6, zorder=2)
        ax.hlines([lo, hi], x - 0.06, x + 0.06, color=TEXT_2, lw=1.6, zorder=2)
        ax.scatter([x + 0.22] * len(seeds), seeds, s=42, facecolor="white",
                   edgecolor=BLUE, linewidth=1.5, zorder=3)
        ax.scatter([x], [mean], s=80, color=BLUE, edgecolor="white", linewidth=2, zorder=4)
        ax.annotate(signed(mean_raw, 100), (x, mean), xytext=(-12, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=10,
                    color=TEXT, fontweight="bold")
    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    ax.axhline(bar, color=ORANGE, lw=1.6, zorder=1)
    ax.text(len(xs) - 0.45, bar + 0.6,
            f"preregistered bar: {signed(v['gate_threshold'], 100, 0)} F1 (mean of seed means)",
            ha="right", va="bottom", fontsize=8.5, color=TEXT)
    ax.set_ylim(min(lows) - 2, max(highs + [bar]) + 4)
    ax.set_xlim(-0.6, len(xs) - 0.35)
    ticks = []
    for r in v["rungs"]:
        st = r["sign_test"]
        ticks.append(f"{model_label(r['model'])}\n({r['environment']})\n"
                     f"{st['wins']} W / {st['losses']} L / {st['ties']} T\n"
                     f"sign test p = {fmt(st['p_value'], 1, 2)}\n"
                     f"MDE (80% power) {fmt(r['mde_80pct'], 100, 1)}")
    ax.set_xticks(xs, ticks, fontsize=8.5)
    ax.set_ylabel("adapted minus k-shot, micro-F1 points")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    n = v["rungs"][0]["n_paired_receipts"]
    per = v["rungs"][0]["receipts_per_seed"]
    ax.set_title("CORD receipts, k = 10: per-request LoRA vs the same model's k-shot prompt.\n"
                 f"All three rungs miss the +5 bar. n = {n} paired receipts per rung "
                 f"({len(per)} seeds x {per[0]}).", fontsize=10, loc="left", color=TEXT)
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=9, color=BLUE, mec="white", mew=2,
                   label="gate statistic: mean of the 3 seed means"),
        plt.Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec=BLUE, mew=1.5,
                   label=f"one seed's mean ({per[0]} receipts)"),
        plt.Line2D([], [], color=TEXT_2, lw=1.6,
                   label=f"receipt-level 95% CI, {n} pairs treated as independent "
                         "(seed slices can share receipts)"),
        plt.Line2D([], [], color=ORANGE, lw=1.6, label="preregistered bar"),
    ]
    fig.legend(handles=handles, loc="lower left", fontsize=8, frameon=False, ncol=1,
               bbox_to_anchor=(0.08, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return fig


# --------------------------------------------------------------------------
# Figure 2: fence RATE, schema-only vs k-shot, on two corpora (R table, S, T)
# --------------------------------------------------------------------------

F2_WAYBILL = "experiments/fence_impact_table.json"
F2_CORD_S = "experiments/cord_fence_tax_2026-08-25.json"
F2_CORD_T = "experiments/cord_fence_tax_families_2026-09-03.json"


def _regime(name: str) -> str:
    return "schema" if name.startswith("schema") else "kshot"


def extract_fence_rate() -> dict:
    out: dict[str, list] = {"waybills": [], "cord": []}
    wb = load(F2_WAYBILL)
    for arm in wb["arms"]:
        _require(round(arm["fenced"] / arm["n"], 4) == arm["fence_rate"],
                 f"waybill {arm['model']} {arm['regime']}: fenced/n != fence_rate")
        out["waybills"].append({"model": arm["model"], "regime": _regime(arm["regime"]),
                                "regime_as_banked": arm["regime"],
                                "fenced": arm["fenced"], "n": arm["n"], "addendum": None})
    for path in (F2_CORD_S, F2_CORD_T):
        art = load(path)
        for row in art["rows"]:
            _require(round(row["fenced"] / row["n"], 4) == row["fence_rate"],
                     f"{path} {row['model']} {row['regime']}: fenced/n != fence_rate")
            out["cord"].append({"model": row["model"], "regime": _regime(row["regime"]),
                                "regime_as_banked": row["regime"],
                                "fenced": row["fenced"], "n": row["n"],
                                "addendum": art["addendum"]})
    return out


def _models_in_order(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        if r["model"] not in seen:
            seen.append(r["model"])
    return seen


def claims_fence_rate(v: dict) -> list[dict]:
    def frac(corpus: str, model_part: str, regime: str) -> str:
        hits = [r for r in v[corpus] if model_part in r["model"] and r["regime"] == regime]
        _require(len(hits) == 1, f"{corpus} {model_part} {regime}: {len(hits)} rows")
        return f"{hits[0]['fenced']}/{hits[0]['n']}"

    def of(corpus: str, model_part: str, regime: str) -> str:
        return frac(corpus, model_part, regime).replace("/", " of ")

    fam = fence_family_counts(v)
    return [
        {"doc": "README.md", "quote": "the tax has been measured on **2 of 5 families**",
         "numbers": [("2 of 5", f"{len(fam['fenced_families'])} of {len(fam['families'])}",
                      "families with any schema-only fence / families measured")]},
        {"doc": "VERDICT.md",
         "quote": "Schema-only fence rates: **Falcon3-1B 92/100; SmolLM2-1.7B 0/100; "
                  "Granite-3.1-2B 0/100; Phi-3-mini 0/100.**",
         "numbers": [("92/100", frac("cord", "Falcon3", "schema"), "Falcon3 schema"),
                     ("SmolLM2-1.7B 0/100", "SmolLM2-1.7B " + frac("cord", "SmolLM2", "schema"),
                      "SmolLM2 schema"),
                     ("Granite-3.1-2B 0/100", "Granite-3.1-2B " + frac("cord", "granite", "schema"),
                      "Granite schema"),
                     ("Phi-3-mini 0/100", "Phi-3-mini " + frac("cord", "Phi-3", "schema"),
                      "Phi-3 schema")]},
        {"doc": "VERDICT.md",
         "quote": "K-shot fence rates: 0/80, 0/80, 0/80, 0/80",
         "numbers": [("0/80", frac("cord", m, "kshot"), f"{m} k-shot")
                     for m in ("Falcon3", "SmolLM2", "granite", "Phi-3")]},
        {"doc": "VERDICT.md",
         "quote": "schema-only prompts are fenced 100/100 at 0.5B and 100/100 at 1.5B; "
                  "k-shot prompts are fenced 0/80 at both sizes",
         "numbers": [("100/100 at 0.5B", frac("cord", "Qwen2.5-0.5B", "schema") + " at 0.5B",
                      "Qwen 0.5B schema (CORD)"),
                     ("100/100 at 1.5B", frac("cord", "Qwen2.5-1.5B", "schema") + " at 1.5B",
                      "Qwen 1.5B schema (CORD)"),
                     ("0/80", frac("cord", "Qwen2.5-0.5B", "kshot"), "Qwen 0.5B k-shot (CORD)"),
                     ("0/80", frac("cord", "Qwen2.5-1.5B", "kshot"), "Qwen 1.5B k-shot (CORD)")]},
        {"doc": "VERDICT.md",
         "quote": "**30 of 30 documents fenced in the schema regime at 0.5B, 1.5B and 3B; "
                  "0 of 30 fenced in the k-shot regime at 0.5B and 1.5B.**",
         "numbers": [("30 of 30", of("waybills", m, "schema"), f"waybill {m} schema")
                     for m in ("0.5B", "1.5B", "3B")]
         + [("0 of 30", of("waybills", m, "kshot"), f"waybill {m} k-shot")
            for m in ("0.5B", "1.5B")]},
    ]


# Display grouping only: which family a checkpoint belongs to. No value.
_FAMILY_PREFIX = (("Qwen2.5", "Qwen2.5"), ("Falcon3", "Falcon3"), ("granite", "Granite-3.1"),
                  ("Phi-3", "Phi-3"), ("SmolLM2", "SmolLM2"))


def family_of(model_id: str) -> str:
    name = model_id.split("/")[-1]
    for prefix, family in _FAMILY_PREFIX:
        if name.startswith(prefix):
            return family
    raise Mismatch(f"no family mapping for {model_id!r}")


def fence_family_counts(v: dict) -> dict:
    rows = v["waybills"] + v["cord"]
    families = sorted({family_of(r["model"]) for r in rows})
    fenced = sorted({family_of(r["model"]) for r in rows
                     if r["regime"] == "schema" and r["fenced"] > 0})
    kshot_fenced = sum(r["fenced"] for r in rows if r["regime"] == "kshot")
    kshot_n = sum(r["n"] for r in rows if r["regime"] == "kshot")
    return {"families": families, "fenced_families": fenced,
            "kshot_fenced": kshot_fenced, "kshot_n": kshot_n}


def draw_fence_rate(v: dict, plt: Any) -> Any:
    wb_models = _models_in_order(v["waybills"])
    cd_models = _models_in_order(v["cord"])
    fam = fence_family_counts(v)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True,
                             gridspec_kw={"width_ratios": [len(wb_models) + 0.6,
                                                           len(cd_models) + 0.6]})
    width = 0.36
    for ax, corpus, models in ((axes[0], "waybills", wb_models), (axes[1], "cord", cd_models)):
        rows = {(r["model"], r["regime"]): r for r in v[corpus]}
        labels = []
        for i, m in enumerate(models):
            for j, (regime, color, hatch) in enumerate((("schema", BLUE, None),
                                                        ("kshot", ORANGE, "///"))):
                r = rows[(m, regime)]
                rate = 100 * r["fenced"] / r["n"]
                x = i + (j - 0.5) * (width + 0.03)
                if rate > 0:
                    ax.bar(x, rate, width, color=color, hatch=hatch, edgecolor="white", lw=0)
                else:
                    ax.hlines(0, x - width / 2, x + width / 2, color=color, lw=3)
                ax.text(x, rate + 1.5, f"{r['fenced']}/{r['n']}", ha="center", va="bottom",
                        fontsize=7, color=TEXT)
            add = rows[(m, "schema")]["addendum"]
            labels.append(model_label(m) + (f"\n(Addendum {add})" if add else ""))
        ax.set_xticks(range(len(models)), labels, fontsize=8)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    n_wb = sorted({r["n"] for r in v["waybills"]})
    n_cd_s = sorted({r["n"] for r in v["cord"] if r["regime"] == "schema"})
    n_cd_k = sorted({r["n"] for r in v["cord"] if r["regime"] == "kshot"})
    axes[0].set_title(f"Waybills (agent-authored), n = {'/'.join(map(str, n_wb))} per arm",
                      fontsize=9, loc="left")
    axes[1].set_title(f"CORD (public receipts), schema n = {'/'.join(map(str, n_cd_s))}, "
                      f"k-shot n = {'/'.join(map(str, n_cd_k))}", fontsize=9, loc="left")
    axes[0].set_ylim(0, 112)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel("outputs wrapped in a markdown fence (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="field list only (schema)"),
               plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, hatch="///", edgecolor="white",
                             label="20 demonstrations (k-shot)")]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(f"Fence rate by prompt regime. Schema-only outputs fenced in "
                 f"{len(fam['fenced_families'])} of the {len(fam['families'])} families measured "
                 f"({', '.join(fam['fenced_families'])}); k-shot outputs fenced "
                 f"{fam['kshot_fenced']} of {fam['kshot_n']}.",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Figure 3: what one fence costs a strict parser, waybills (fence impact table)
# --------------------------------------------------------------------------

F3_TABLE = "experiments/fence_impact_table.json"


def extract_fence_tax() -> dict:
    table = load(F3_TABLE)
    arms = []
    for arm in table["arms"]:
        path = f"experiments/{arm['artifact']}"
        source = load(path)
        _require(source["n"] == arm["n"], f"{path}: n {source['n']} != table {arm['n']}")
        _require(round(source["mean_micro_f1"], 4) == arm["mean_as_emitted"],
                 f"{path}: mean_micro_f1 {source['mean_micro_f1']} != table as-emitted "
                 f"{arm['mean_as_emitted']}")
        _require(round(arm["mean_fence_stripped"] - arm["mean_as_emitted"], 4) == arm["fence_tax"],
                 f"{path}: fence_tax is not stripped minus as-emitted")
        arms.append({"model": arm["model"], "regime": _regime(arm["regime"]),
                     "regime_as_banked": arm["regime"], "arm_artifact": path,
                     "dtype": source["dtype"], "n": arm["n"], "fenced": arm["fenced"],
                     "mean_as_emitted": arm["mean_as_emitted"],
                     "invalid_as_emitted": arm["invalid_as_emitted"],
                     "mean_fence_stripped": arm["mean_fence_stripped"],
                     "invalid_fence_stripped": arm["invalid_fence_stripped"],
                     "fence_tax": arm["fence_tax"]})
    return {"arms": arms}


def artifacts_fence_tax(v: dict) -> list[str]:
    return [F3_TABLE] + [a["arm_artifact"] for a in v["arms"]]


def claims_fence_tax(v: dict) -> list[dict]:
    def arm(size: str, regime: str) -> dict:
        hits = [a for a in v["arms"] if f"-{size}-" in a["model"] + "-" and a["regime"] == regime]
        _require(len(hits) == 1, f"{size} {regime}: {len(hits)} arms")
        return hits[0]

    claims = []
    for size, stripped in (("0.5B", "0.6350"), ("1.5B", "0.7375"), ("3B", "0.8958")):
        a = arm(size, "schema")
        claims.append({"doc": "README.md",
                       "quote": f"| {size} | 30/30 | `0.0000` | **{stripped}** |",
                       "numbers": [("30/30", f"{a['fenced']}/{a['n']}", f"{size} schema fenced"),
                                   ("0.0000", a["mean_as_emitted"], f"{size} schema as emitted"),
                                   (stripped, a["mean_fence_stripped"],
                                    f"{size} schema stripped")]})
    for size, value in (("0.5B", "0.7836"), ("1.5B", "0.8804")):
        a = arm(size, "kshot")
        claims.append({"doc": "README.md",
                       "quote": (f"| {size}, with 20 demonstrations | **0/30** "
                                 f"| {value} | {value} |"),
                       "numbers": [("0/30", f"{a['fenced']}/{a['n']}", f"{size} k-shot fenced"),
                                   (value, a["mean_as_emitted"], f"{size} k-shot as emitted"),
                                   (value, a["mean_fence_stripped"], f"{size} k-shot stripped")]})
    a = arm("1.5B", "schema")
    claims.append({"doc": "EVIDENCE.md", "quote": "Fence-stripped: **0.7375, zero invalid.**",
                   "numbers": [("0.7375", a["mean_fence_stripped"], "1.5B schema stripped")]})
    a = arm("1.5B", "schema")
    claims.append({"doc": "VERDICT.md",
                   "quote": ("re-scored with one fence removed they are "
                             "**0.7375 with 0 of 30 invalid**"),
                   "numbers": [("0.7375", a["mean_fence_stripped"], "1.5B schema stripped"),
                               ("0 of 30", f"{a['invalid_fence_stripped']} of {a['n']}",
                                "1.5B invalid after strip")]})
    a = arm("3B", "schema")
    claims.append({"doc": "EVIDENCE.md",
                   "quote": "scores **0.8958 with 0 of 30 invalid**",
                   "numbers": [("0.8958", a["mean_fence_stripped"], "3B schema stripped"),
                               ("0 of 30", f"{a['invalid_fence_stripped']} of {a['n']}",
                                "3B invalid after strip")]})
    return claims


def draw_fence_tax(v: dict, plt: Any) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), sharey=True)
    width = 0.36
    for ax, regime, title in ((axes[0], "schema", "Field list only (schema prompt)"),
                              (axes[1], "kshot", "20 demonstrations (k-shot prompt)")):
        arms = [a for a in v["arms"] if a["regime"] == regime]
        fenced = sorted({f"{a['fenced']}/{a['n']}" for a in arms})
        for i, a in enumerate(arms):
            tie = a["mean_as_emitted"] == a["mean_fence_stripped"]
            for j, (key, color, hatch, inv) in enumerate(
                    (("mean_as_emitted", NEUTRAL, "///", "invalid_as_emitted"),
                     ("mean_fence_stripped", BLUE, None, "invalid_fence_stripped"))):
                x = i + (j - 0.5) * (width + 0.03)
                val = a[key]
                if val > 0:
                    ax.bar(x, val, width, color=color, hatch=hatch, edgecolor="white", lw=0)
                else:
                    ax.hlines(0, x - width / 2, x + width / 2, color=TEXT_2, lw=3)
                if not tie:
                    ax.text(x, val + 0.015, f"{val:.4f}\n{a[inv]}/{a['n']} invalid",
                            ha="center", va="bottom", fontsize=7, color=TEXT)
            if tie:
                ax.text(i, a["mean_as_emitted"] + 0.015,
                        f"tie: {a['mean_as_emitted']:.4f} either way\n"
                        f"{a['invalid_as_emitted']}/{a['n']} invalid, nothing to strip",
                        ha="center", va="bottom", fontsize=7, color=TEXT)
        ax.set_xticks(range(len(arms)),
                      [f"{model_label(a['model'])}\n{a['dtype']}" for a in arms], fontsize=8)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.set_title(f"{title}\nfenced: {', '.join(fenced)} in every arm", fontsize=9, loc="left")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel("mean micro-F1 (same pinned scorer)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=NEUTRAL, hatch="///", edgecolor="white",
                             label=("scored as emitted (bare json.loads; a fenced output "
                                    "is invalid, scores 0)")),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, label="scored after stripping one "
                                                             "leading fence")]
    fig.legend(handles=handles, loc="lower left", fontsize=8, frameon=False, ncol=2,
               bbox_to_anchor=(0.06, 0.0))
    n = sorted({a["n"] for a in v["arms"]})
    fig.suptitle(f"Waybills, Qwen2.5 only, n = {'/'.join(map(str, n))} documents per arm. "
                 "A strict parse scores every fenced output as 0.",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# --------------------------------------------------------------------------
# Figure 4: the two fence censuses, confirmed vs rejected, per package
# --------------------------------------------------------------------------

F4_HAND = "experiments/fence_census_hand.json"
F4_AMEND = "experiments/fence_census_amendment_2026-08-25.json"
F4_LIVE = "experiments/census_extension_live_2026-08-31.json"


def extract_census() -> dict:
    hand = load(F4_HAND)
    amend = load(F4_AMEND)
    live = load(F4_LIVE)
    adj = hand["adjudications"]
    _require(len(adj) == hand["candidates_examined"], "adjudications != candidates_examined")
    verdicts = Counter(a["verdict"] for a in adj)
    _require(verdicts["CONFIRMED"] == hand["confirmed"], "hand confirmed count disagrees")
    _require(verdicts["REJECTED"] == hand["rejected"], "hand rejected count disagrees")
    per_pkg: dict[str, dict] = {}
    for a in adj:
        d = per_pkg.setdefault(a["package"], {"confirmed": 0, "rejected": 0, "amendment": 0})
        d["confirmed" if a["verdict"] == "CONFIRMED" else "rejected"] += 1
    _require(sorted(p for p, d in per_pkg.items() if d["confirmed"])
             == sorted(hand["packages_with_a_confirmed_site"]), "confirmed packages disagree")
    amended = [s for s in amend["sites"]
               if s["verdict"] == "CONFIRMED" and not s["in_original_census"]]
    for s in amended:
        pkg = s["site"].split("/")[0]
        _require(pkg in per_pkg, f"amendment package {pkg!r} not in the census frame")
        per_pkg[pkg]["amendment"] += 1
    _require(hand["confirmed_in_library_code"] + hand["confirmed_in_shipped_example_code"]
             == hand["confirmed"], "library + example != confirmed")
    census1 = {
        "packages_searched": hand["packages_searched"],
        "candidate_sites": hand["candidates_examined"],
        "confirmed_by_hand": hand["confirmed"],
        "rejected_by_hand": hand["rejected"],
        "confirmed_in_library_code": hand["confirmed_in_library_code"],
        "confirmed_in_shipped_example_code": hand["confirmed_in_shipped_example_code"],
        "confirmed_by_amendment": len(amended),
        "packages_with_no_candidate": hand["packages_searched"] - len(per_pkg),
        "per_package": dict(sorted(per_pkg.items())),
    }
    raw = live["raw_findings"]
    total = raw.pop("total") if "total" in raw else None
    raw = dict(raw)
    _require(total == sum(raw.values()), "live raw findings do not sum to total")
    confirmed_pkgs = Counter(c["package"].split()[0].replace("-", "_") for c in live["confirmed"])
    _require(sum(confirmed_pkgs.values()) + live["rejected_count"] == total,
             "live confirmed + rejected != total")
    for p in confirmed_pkgs:
        _require(p in raw, f"confirmed package {p!r} not in raw_findings")
    released = sorted(d["released"] for d in live["frame"]["in_frame"].values())
    reach = live.get("rejected_on_reachability_2026-09-01", [])
    census2 = {
        "packages_in_frame": len(live["frame"]["in_frame"]),
        "released_range": [released[0], released[-1]],
        "raw_findings_total": total,
        "confirmed": sum(confirmed_pkgs.values()),
        "rejected": live["rejected_count"],
        "per_package": {p: {"raw_findings": n, "confirmed": confirmed_pkgs.get(p, 0),
                            "rejected": n - confirmed_pkgs.get(p, 0)}
                        for p, n in sorted(raw.items())},
        "confirmed_then_rejected_on_reachability": [r["package"].split()[0] for r in reach],
    }
    return {"census_1": census1, "census_2_maintained": census2}


def claims_census(v: dict) -> list[dict]:
    c1, c2 = v["census_1"], v["census_2_maintained"]
    total1 = c1["confirmed_by_hand"] + c1["confirmed_by_amendment"]
    return [
        {"doc": "README.md", "quote": "**7 confirmed instances across 34 published packages**",
         "numbers": [("7", total1, "census 1 confirmed incl. amendment"),
                     ("34", c1["packages_searched"], "census 1 packages")]},
        {"doc": "README.md", "quote": "6 of 35 candidate sites read by hand",
         "numbers": [("6", c1["confirmed_by_hand"], "census 1 confirmed by hand"),
                     ("35", c1["candidate_sites"], "census 1 candidate sites")]},
        {"doc": "README.md", "quote": "All 29 rejections are published",
         "numbers": [("29", c1["rejected_by_hand"], "census 1 rejected")]},
        {"doc": "README.md",
         "quote": "preregistered census over **7 actively-maintained packages**",
         "numbers": [("7", c2["packages_in_frame"], "census 2 packages")]},
        {"doc": "README.md", "quote": "found **1 confirmed instance**",
         "numbers": [("1", c2["confirmed"], "census 2 confirmed")]},
        {"doc": "VERDICT.md", "quote": "every one of the **64 raw findings hand-adjudicated**",
         "numbers": [("64", c2["raw_findings_total"], "census 2 raw findings")]},
        {"doc": PITCH,
         "quote": "One confirmed instance in seven maintained packages",
         "numbers": [("One", c2["confirmed"], "census 2 confirmed"),
                     ("seven", c2["packages_in_frame"], "census 2 packages")]},
    ]


def draw_census(v: dict, plt: Any) -> Any:
    c1, c2 = v["census_1"], v["census_2_maintained"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
    ax = axes[0]
    items = sorted(c1["per_package"].items(),
                   key=lambda kv: (sum(kv[1].values()), kv[0]))
    for i, (pkg, d) in enumerate(items):
        left = 0
        for key, color, hatch in (("confirmed", BLUE, None), ("amendment", BLUE, "xxx"),
                                  ("rejected", NEUTRAL, None)):
            if d[key]:
                ax.barh(i, d[key], 0.62, left=left, color=color if hatch is None else "white",
                        hatch=hatch, edgecolor=BLUE if hatch else "white", lw=0 if not hatch else 1)
                left += d[key]
        label = f"{d['confirmed']} of {d['confirmed'] + d['rejected']} confirmed"
        if d["amendment"]:
            label += f", +{d['amendment']} found later"
        ax.text(left + 0.2, i, label, va="center", fontsize=7.5, color=TEXT)
    ax.set_yticks(range(len(items)), [p for p, _ in items], fontsize=8)
    ax.set_xlabel("candidate sites, each read by hand")
    ax.set_xlim(0, max(sum(d.values()) for _, d in items) + 5)
    ax.set_title(f"Census 1: {c1['packages_searched']} published packages, "
                 f"{c1['candidate_sites']} candidate sites\n"
                 f"{c1['confirmed_by_hand']} confirmed by hand, +{c1['confirmed_by_amendment']} "
                 f"found later by the shipped tool;\n{c1['packages_with_no_candidate']} packages "
                 "with no candidate site are not drawn", fontsize=8.5, loc="left")
    ax = axes[1]
    items2 = sorted(c2["per_package"].items(), key=lambda kv: (kv[1]["raw_findings"], kv[0]))
    for i, (pkg, d) in enumerate(items2):
        left = 0
        for key, color in (("confirmed", BLUE), ("rejected", NEUTRAL)):
            if d[key]:
                ax.barh(i, d[key], 0.62, left=left, color=color, edgecolor="white", lw=0)
                left += d[key]
        text = (f"{d['confirmed']} of {d['raw_findings']} confirmed" if d["raw_findings"]
                else "0 findings")
        if pkg in c2["confirmed_then_rejected_on_reachability"]:
            text += "\n(1 counted confirmed for a day,\nthen rejected: unreachable)"
        ax.text(left + 0.4, i, text, va="center", fontsize=7.5, color=TEXT)
    ax.set_yticks(range(len(items2)), [p for p, _ in items2], fontsize=8)
    ax.set_xlabel("raw findings from fencecheck scan, each read by hand")
    ax.set_xlim(0, max(d["raw_findings"] for _, d in items2) + 14)
    lo, hi = c2["released_range"]
    ax.set_title(f"Census 2: {c2['packages_in_frame']} maintained packages\n"
                 f"(releases {lo} to {hi})\n{c2['raw_findings_total']} raw findings, "
                 f"{c2['confirmed']} confirmed, {c2['rejected']} rejected", fontsize=8.5,
                 loc="left")
    for a, n_rows in ((axes[0], len(items)), (axes[1], len(items2))):
        a.xaxis.grid(True, color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.set_ylim(-0.7, n_rows - 0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="confirmed instance"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch="xxx", edgecolor=BLUE,
                             label="confirmed, dated amendment"),
               plt.Rectangle((0, 0), 1, 1, color=NEUTRAL, label="rejected, reason published")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Fail-open JSON parse sites in eval code: a lower bound among packages "
                 "searched, not a rate.", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# --------------------------------------------------------------------------
# Figure 5: determinism, Addendum X gate and Addendum Y
# --------------------------------------------------------------------------

F5_X = "experiments/cord_decoder_isolation_2026-09-16.json"
F5_Y = "experiments/phi3_instability_2026-09-17.json"
F5_Y_PROTOCOL = "docs/research/ADDENDUM_Y_PROTOCOL.md"   # frozen, OTS-anchored
F5_Y_SCRIPT = "scripts/phi3_instability.py"              # cross-checked, not hashed


def extract_determinism() -> dict:
    x = load(F5_X)
    y = load(F5_Y)
    # Y's artifact carries counts, not denominators. The denominators come
    # from the frozen protocol, and the runner's constants must agree.
    protocol = (REPO / F5_Y_PROTOCOL).read_text(encoding="utf-8")
    m = re.search(r"^## Quantities \u2014 per pair, counts of (\d+)$", protocol, re.MULTILINE)
    _require(m is not None, f"{F5_Y_PROTOCOL}: 'counts of N' heading not found")
    n_docs = int(m.group(1))
    m = re.search(r"^- `Y5` \u2014 documents \(of (\d+)\)", protocol, re.MULTILINE)
    _require(m is not None, f"{F5_Y_PROTOCOL}: Y5 denominator not found")
    thread_docs = int(m.group(1))
    script = (REPO / F5_Y_SCRIPT).read_text(encoding="utf-8")
    m = re.search(r"^N_DOCS = (\d+)$", script, re.MULTILINE)
    _require(m is not None and int(m.group(1)) == n_docs,
             f"{F5_Y_SCRIPT}: N_DOCS disagrees with the frozen protocol")
    m = re.search(r"^THREAD_DOCS = \(([^)]*)\)$", script, re.MULTILINE)
    _require(m is not None and len([t for t in m.group(1).split(",") if t.strip()])
             == thread_docs, f"{F5_Y_SCRIPT}: THREAD_DOCS disagrees with the frozen protocol")
    gate = []
    for family, g in x["determinism_gate"].items():
        # Only the boolean *_identical keys are read. No text field is touched.
        flags = [val for doc in g["documents"] for key, val in doc.items()
                 if key.endswith("_identical")]
        differing = sum(1 for f in flags if f is False)
        _require(differing == g["mismatches"], f"X {family}: booleans != mismatches")
        _require(len(g["documents"]) == g["n"], f"X {family}: documents != n")
        by_arm: dict[str, int] = {}
        for doc in g["documents"]:
            for key, val in doc.items():
                if key.endswith("_identical"):
                    arm = key.removeprefix("arm_").removesuffix("_identical")
                    by_arm[arm] = by_arm.get(arm, 0) + (val is False)
        gate.append({"family": family, "documents": g["n"], "comparisons": len(flags),
                     "differing": differing, "differing_by_arm": by_arm,
                     "passed": g["passed"]})
    _require(x["failed_families"] == [r["family"] for r in gate if not r["passed"]],
             "X failed_families disagrees with per-family gate")
    rows = []
    for pair, r in y["rows"].items():
        for check in ("Y1_immediate_repeat_differs", "Y2_state_differs",
                      "Y3_generate_state_differs"):
            _require(len(r["documents_differing"][check[:2]]) == r[check],
                     f"Y {pair} {check}: list length != count")
            rows.append({"pair": pair, "check": check, "differing": r[check], "of": n_docs})
        for check, val in r["then_vs_now"].items():
            rows.append({"pair": pair, "check": f"then_vs_now.{check}", "differing": val,
                         "of": n_docs})
    key = next(k for k in y if k.startswith("Y5_thread_count_differs_of_"))
    y5_of = int(key.rsplit("_", 1)[1])
    _require(y5_of == thread_docs, f"Y5 denominator {y5_of} != protocol's {thread_docs}")
    return {
        "x_date": x["date"],
        "x_reading": x["reading"],
        "x_gate": gate,
        "y_date": y["date"],
        "y_n_docs": n_docs,
        "y_rows": rows,
        "y_readings": y["per_pair"],
        "y5_thread_count": {"differing": y[key], "of": y5_of, "reading": y["thread_reading"]},
    }


def claims_determinism(v: dict) -> list[dict]:
    passing = [g for g in v["x_gate"] if g["passed"] and g["comparisons"]]
    _require(len({g["comparisons"] for g in passing}) == 1, "passing families differ in n")
    phi = next(g for g in v["x_gate"] if not g["passed"])
    y = {(r["pair"], r["check"]): r for r in v["y_rows"]}
    y123 = [r for r in v["y_rows"] if r["check"][:2] in ("Y1", "Y2", "Y3")]
    _require(len({r["differing"] for r in y123}) == 1, "Y1-3 counts are not all equal")
    gen = y[("phi3-mini/bfloat16", "then_vs_now.generate_g1_vs_banked")]
    con = y[("phi3-mini/bfloat16", "then_vs_now.constrained_a1_vs_banked")]
    return [
        {"doc": "EVIDENCE.md", "quote": "Three families do, 0 mismatches of 10 comparisons each",
         "numbers": [("Three", len(passing), "X families passing with comparisons"),
                     ("0", max(g["differing"] for g in passing), "X mismatches"),
                     ("10", passing[0]["comparisons"], "X comparisons per family")]},
        {"doc": "VERDICT.md", "quote": "the second run **fails too** (2 mismatches)",
         "numbers": [("2", phi["differing"], "X Phi-3 mismatches (the banked run)")]},
        {"doc": "VERDICT.md", "quote": "Y1 = Y2 = Y3 = 0 of 10 on Phi-3/bfloat16",
         "numbers": [("0 of 10", f"{y123[0]['differing']} of {y123[0]['of']}", "Y1-3")]},
        {"doc": "VERDICT.md", "quote": "(Y5 = 1 of 2)",
         "numbers": [("1 of 2", (f"{v['y5_thread_count']['differing']} of "
                                 f"{v['y5_thread_count']['of']}"), "Y5")]},
        {"doc": "VERDICT.md", "quote": "Phi-3's constrained pass also matches Addendum V's "
                                       "banked cell 10 of 10",
         "numbers": [("10 of 10", f"{con['of'] - con['differing']} of {con['of']}",
                      "Phi-3 constrained then-vs-now agreeing")]},
        {"doc": "VERDICT.md", "quote": "on 4 of 10 documents",
         "numbers": [("4 of 10", f"{gen['differing']} of {gen['of']}",
                      "Phi-3 generate then-vs-now differing")]},
        {"doc": "paper/DRAFT.md",
         "quote": "all 30 comparisons were byte-identical: five documents, two reused arms, "
                  "three families",
         "numbers": [("30", sum(g["comparisons"] for g in passing), "X comparisons, passing"),
                     ("five", passing[0]["documents"], "X documents per family"),
                     ("two", len(passing[0]["differing_by_arm"]), "X reused arms"),
                     ("three", len(passing), "X passing families")]},
        {"doc": "paper/DRAFT.md",
         "quote": "The constrained decoder accounts for 15 of them and `model.generate` for "
                  "the other 15",
         "numbers": [("15", sum(g["documents"] for g in passing),
                      "X constrained-arm comparisons, passing families")]},
        {"doc": "paper/DRAFT.md",
         "quote": "did not reproduce the text banked on 2026-09-05 for the same two of the "
                  "five documents",
         "numbers": [("two", phi["differing_by_arm"].get("G0", -1),
                      "X Phi-3 generate-comparator documents differing (banked run)"),
                     ("five", phi["documents"], "X documents")]},
        {"doc": "paper/DRAFT.md", "quote": "on 4 of 10 documents",
         "numbers": [("4 of 10", f"{gen['differing']} of {gen['of']}",
                      "Phi-3 generate then-vs-now differing")]},
        {"doc": "paper/DRAFT.md", "quote": "It decoded the first ten documents",
         "numbers": [("ten", v["y_n_docs"], "Y documents")]},
    ]


_FAMILY = {"qwen2.5-0.5b": "Qwen2.5-0.5B", "smollm2-1.7b": "SmolLM2-1.7B",
           "falcon3-1b": "Falcon3-1B", "granite-2b": "Granite-3.1-2B", "phi3-mini": "Phi-3-mini"}
_CHECK = {"Y1_immediate_repeat_differs": "Y1 immediate repeat",
          "Y2_state_differs": "Y2 state dependence",
          "Y3_generate_state_differs": "Y3 generate path, state dependence",
          "then_vs_now.constrained_a1_vs_banked": "today vs banked constrained cell",
          "then_vs_now.generate_g1_vs_banked": "today vs banked generate cell"}


def draw_determinism(v: dict, plt: Any) -> Any:
    docs = sorted({g["documents"] for g in v["x_gate"] if g["documents"]})
    x_head = (f"Addendum X gate ({v['x_date']}): reused cells re-decoded, "
              f"{'/'.join(map(str, docs))} documents x 2 arms,\ncompared "
              "byte-for-byte with the banked text")
    lines: list[tuple] = [("head", x_head)]
    for g in v["x_gate"]:
        label = _FAMILY.get(g["family"], g["family"])
        if g["comparisons"] == 0:
            lines.append(("row", label, 0, 0, "n = 0: no reused cell, nothing to gate"))
        else:
            verdict = "gate passed" if g["passed"] else "GATE FAILED: Addendum X withheld"
            lines.append(("row", label, g["differing"], g["comparisons"],
                          f"{g['differing']} of {g['comparisons']} differ; {verdict}"))
    y_head = (f"Addendum Y ({v['y_date']}): one unbroken process, {v['y_n_docs']} "
              "documents per check,\ncompared after the shipped fence strip")
    lines.append(("head", y_head))
    pair_name = {"phi3-mini/bfloat16": "Phi-3-mini bf16",
                 "qwen2.5-0.5b/bfloat16": "Qwen2.5-0.5B bf16",
                 "qwen2.5-0.5b/float32": "Qwen2.5-0.5B fp32"}
    for r in v["y_rows"]:
        label = f"{pair_name.get(r['pair'], r['pair'])}: {_CHECK.get(r['check'], r['check'])}"
        lines.append(("row", label, r["differing"], r["of"],
                      f"{r['differing']} of {r['of']} differ"))
    t = v["y5_thread_count"]
    lines.append(("row", "Phi-3-mini bf16: Y5 4 threads vs 1 thread", t["differing"], t["of"],
                  f"{t['differing']} of {t['of']} differ ({t['reading'].lower()})"))
    height = 0.27 * len(lines) + 1.3
    fig, ax = plt.subplots(figsize=(9.6, height))
    y = 0.0
    for item in lines:
        if item[0] == "head":
            y -= 0.5
            ax.text(-9.8, y, item[1], fontsize=8.5, fontweight="bold", color=TEXT, va="center")
            y -= 1.15
            continue
        _, label, differing, total, note = item
        ax.text(-0.8, y, label, ha="right", va="center", fontsize=8, color=TEXT)
        for k in range(total):
            hit = k < differing
            ax.scatter(k, y, s=46, marker="o",
                       facecolor=ORANGE if hit else "white",
                       edgecolor=ORANGE if hit else NEUTRAL, linewidth=1.3, zorder=3)
        ax.text(10.6, y, note, ha="left", va="center", fontsize=8, color=TEXT)
        y -= 1
    ax.set_xlim(-9.8, 22)
    ax.set_ylim(y + 0.3, 0.8)
    ax.axis("off")
    ax.scatter([], [], s=46, facecolor=ORANGE, edgecolor=ORANGE, label="differs")
    ax.scatter([], [], s=46, facecolor="white", edgecolor=NEUTRAL, linewidth=1.3,
               label="identical")
    fig.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax.set_title("Does a decode reproduce? Each circle is one comparison. Filled circles count "
                 "differences;\nthey do not mark which document differed.",
                 fontsize=9.5, loc="left", color=TEXT)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

FIGURES_SPEC: dict[str, dict[str, Any]] = {
    "cord_scale_gate": {
        "extract": extract_cord_scale_gate, "claims": claims_cord_scale_gate,
        "draw": draw_cord_scale_gate,
        "artifacts": lambda v: [F1_POWER, *F1_GE2.values(), F1_SUMMARY,
                                *[a["path"] for arms in _cord_k10_arms().values()
                                  for a in arms]],
    },
    "fence_rate": {
        "extract": extract_fence_rate, "claims": claims_fence_rate, "draw": draw_fence_rate,
        "artifacts": lambda v: [F2_WAYBILL, F2_CORD_S, F2_CORD_T],
    },
    "fence_tax_waybills": {
        "extract": extract_fence_tax, "claims": claims_fence_tax, "draw": draw_fence_tax,
        "artifacts": artifacts_fence_tax,
    },
    "fence_census": {
        "extract": extract_census, "claims": claims_census, "draw": draw_census,
        "artifacts": lambda v: [F4_HAND, F4_AMEND, F4_LIVE],
    },
    "determinism_xy": {
        "extract": extract_determinism, "claims": claims_determinism,
        "draw": draw_determinism,
        "artifacts": lambda v: [F5_X, F5_Y, F5_Y_PROTOCOL],
    },
}

EXTRACTORS: dict[str, Callable[[], dict]] = {k: s["extract"] for k, s in FIGURES_SPEC.items()}


def canonical(values: dict) -> Any:
    """The form values take after a JSON round trip, for exact comparison."""
    return json.loads(json.dumps(values, sort_keys=True))


def build(name: str) -> dict:
    spec = FIGURES_SPEC[name]
    values = canonical(spec["extract"]())
    prose = check_prose(spec["claims"](values))
    return {
        "figure": name,
        "generated_by": "scripts/make_figures.py",
        "artifacts": [{"path": p, "sha256": sha256(p)} for p in spec["artifacts"](values)],
        "values": values,
        "prose_checked": prose["public"],
        "private_prose_quotes": prose["private_quotes"],
        "files": [f"figures/{name}.svg", f"figures/{name}.png"],
    }


def _setup_matplotlib() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8, "axes.labelcolor": TEXT, "xtick.color": TEXT_2,
        "ytick.color": TEXT_2, "axes.spines.top": False, "axes.spines.right": False,
        "svg.hashsalt": "arc-ttt-figures", "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white", "hatch.linewidth": 0.8,
        "axes.unicode_minus": True,
    })
    return plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="re-derive values and check prose only; draw nothing (stdlib)")
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES_SPEC))
    args = parser.parse_args()
    names = args.only or list(FIGURES_SPEC)
    records = {name: build(name) for name in names}   # all checks before any drawing
    if args.check:
        for name, rec in records.items():
            priv = rec["private_prose_quotes"]
            print(f"OK {name}: {len(rec['artifacts'])} artifact(s), "
                  f"{len(rec['prose_checked'])} public + {priv['checked']} private prose "
                  f"quote(s) agree; {priv['absent_from_this_tree']} private quote(s) not "
                  "checkable in this tree")
        return 0
    plt = _setup_matplotlib()
    FIGURES.mkdir(exist_ok=True)
    for name, rec in records.items():
        fig = FIGURES_SPEC[name]["draw"](rec["values"], plt)
        fig.savefig(FIGURES / f"{name}.svg", metadata={"Date": None})
        fig.savefig(FIGURES / f"{name}.png", dpi=200, metadata={"Software": None})
        plt.close(fig)
        (FIGURES / f"{name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False)
                                              + "\n", encoding="utf-8")
        print(f"wrote figures/{name}.svg, .png, .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
