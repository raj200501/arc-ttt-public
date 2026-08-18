"""Render CORD before/after-adaptation dumps into a self-contained HTML page.

Inputs: the two --dump-texts artifacts (adapted + baseline arms) from
scripts/enterprise_smoke.py, sharing eval indices. For each receipt the
page shows the OCR text, then baseline vs adapted extractions with every
field checked against ground truth. Nothing hand-edited.

Usage:
    python demo/render_cord_demo.py adapted.json baseline.json out.html
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path


def parse_fields(text: str) -> dict[str, str] | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return {str(k): str(v) for k, v in data.items()}


def fields_table(prediction: str, gold: dict[str, str]) -> tuple[str, int, int]:
    predicted = parse_fields(prediction)
    rows = []
    hits = 0
    for key in gold:
        gold_value = gold[key]
        if predicted is None:
            status, shown = "miss", "—"
        elif key not in predicted:
            status, shown = "miss", "—"
        elif predicted[key] == gold_value:
            status, shown = "hit", predicted[key]
            hits += 1
        else:
            status, shown = "wrong", predicted[key]
        rows.append(
            f"<tr class='{status}'><td>{html.escape(key)}</td>"
            f"<td>{html.escape(shown)}</td>"
            f"<td class='gold'>{html.escape(gold_value)}</td></tr>"
        )
    extra = 0
    if predicted:
        for key in predicted:
            if key not in gold:
                extra += 1
                rows.append(
                    f"<tr class='wrong'><td>{html.escape(key)}</td>"
                    f"<td>{html.escape(predicted[key])}</td>"
                    f"<td class='gold'>(not in receipt)</td></tr>"
                )
    table = (
        "<table><tr><th>field</th><th>extracted</th><th>ground truth</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    return table, hits, len(gold)


def render(adapted_path: Path, baseline_path: Path, out_path: Path) -> None:
    adapted = json.loads(adapted_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    baseline_by_index = {r["index"]: r for r in baseline["results"] if "prediction" in r}

    sections = []
    for result in adapted["results"]:
        if "prediction" not in result:
            continue
        index = result["index"]
        base = baseline_by_index.get(index)
        if base is None:
            continue
        gold = parse_fields(result["gold"]) or {}
        base_table, base_hits, total = fields_table(base["prediction"], gold)
        adapt_table, adapt_hits, _ = fields_table(result["prediction"], gold)
        ocr = html.escape(result["input_text"][:900])
        sections.append(f"""
<section>
  <h2>Receipt <code>#{index}</code> <span class="scorepair">baseline F1 {base["micro_f1"]:.2f} → adapted <b>{result["micro_f1"]:.2f}</b></span></h2>
  <details><summary>the OCR text the model reads</summary><pre>{ocr}</pre></details>
  <div class="compare">
    <div class="arm"><h3>Before — same model, k-shot prompt ({base_hits}/{total} fields)</h3>{base_table}</div>
    <div class="arm"><h3>After — 4.5 min adaptation ({adapt_hits}/{total} fields)</h3>{adapt_table}</div>
  </div>
</section>""")

    page = f"""<meta charset='utf-8'>
<title>Adaptation demo — receipts, before and after</title>
<style>
  :root {{ --paper:#f7f8f7; --ink:#16181d; --muted:#5b6472; --line:#d8dcd9;
          --good:#1a7a2e; --goodbg:#e9f7ec; --bad:#b32d22; --badbg:#fdeeec;
          --card:#ffffff; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --paper:#101216; --ink:#e6e8eb; --muted:#99a1ac; --line:#2a2f36;
    --good:#37d654; --goodbg:#12291a; --bad:#ff6b5e; --badbg:#2e1613;
    --card:#171a20; }} }}
  :root[data-theme="dark"] {{
    --paper:#101216; --ink:#e6e8eb; --muted:#99a1ac; --line:#2a2f36;
    --good:#37d654; --goodbg:#12291a; --bad:#ff6b5e; --badbg:#2e1613;
    --card:#171a20; }}
  body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 2rem auto;
         background: var(--paper); color: var(--ink); padding: 0 1rem; }}
  h1 {{ font-size: 1.6rem; text-wrap: balance; }}
  h2 {{ margin-top: 2.6rem; }} h3 {{ font-size: .95rem; margin: .2rem 0 .6rem; }}
  .scorepair {{ font-size: .85rem; color: var(--muted); font-weight: 400; margin-left: .6rem; }}
  .scorepair b {{ color: var(--ink); }}
  .compare {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .arm {{ flex: 1 1 380px; background: var(--card); border: 1px solid var(--line);
         border-radius: 6px; padding: .8rem 1rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .82rem; }}
  th, td {{ text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--line);
           font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
  th {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .08em;
       color: var(--muted); }}
  tr.hit td {{ background: var(--goodbg); }}
  tr.wrong td, tr.miss td {{ background: var(--badbg); }}
  td.gold {{ color: var(--muted); }}
  details {{ margin: .6rem 0 1rem; }} summary {{ cursor: pointer; color: var(--muted);
           font-size: .85rem; }}
  pre {{ white-space: pre-wrap; font-size: .75rem; background: var(--card);
        border: 1px solid var(--line); border-radius: 6px; padding: .8rem;
        max-height: 260px; overflow: auto; }}
  footer {{ margin-top: 3rem; font-size: .8rem; color: var(--muted); }}
</style>
<h1>Same small model. New weights in 271 seconds.</h1>
<p>Receipts from the public CORD dataset (CC BY 4.0). "Before" is a 0.5B
model prompted with 10 example receipts; "after" is the <em>same model
with the same 10 examples</em> following 271 seconds of per-task
adaptation on commodity hardware — both arms share model, examples, and
held-out receipts, so
any pretraining contamination cancels and the delta is attributable to
adaptation. This page renders the <strong>seed-0 arm</strong>: micro-F1
0.661 → 0.788 (+12.7) on its 20 held-out receipts. Across all 4 paired
arms measured (seeds/k varied), adaptation at this dev scale is
net-neutral — mean −1.3 F1, range −6.5 to +12.7; seed 0 is the favorable
draw, shown here to illustrate the <em>mechanism</em>. Full variance
artifacts in the experiment registry. Green rows match ground truth
exactly; red rows are wrong or missing.</p>
{''.join(sections)}
<footer>Generated by demo/render_cord_demo.py from the run artifacts in
experiments/ (cord_smoke_2026-08-08.json, cord_smoke_baseline_2026-08-08.json);
per-receipt dumps reproduce those runs with identical seed and examples.
Nothing hand-edited.</footer>
"""
    out_path.write_text(page)
    print(f"wrote {out_path} ({len(sections)} receipts)")


if __name__ == "__main__":
    render(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
