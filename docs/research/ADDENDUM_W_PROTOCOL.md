# Addendum W — the instrument's undercount, measured and closed (preregistration)

**Frozen 2026-09-08, before the extended scope ran on any banked
output.** Addendum R's erratum documents that `tools/fencecheck.py`'s
`strip_fence` removes ONE LEADING fence and nothing else, so the
instrument under-credits: a fence after prose, or a bare object inside
prose, is counted as unparseable. Addenda S, T and U carried that
undercount deliberately (a reference that never over-credits), and U's
substance checks found the shapes it misses at single-digit counts.
This addendum makes the wider scope an explicit, opt-in mode of the
shipped tool, measures on the whole corpus what the undercount was
worth, and leaves every frozen reading on the leading-fence scope
untouched.

## The engineering

`tools/fencecheck.py` gains `extract_json(text, scope)`:

- `scope="leading"` — exactly `strip_fence` (unchanged; every existing
  reading uses it).
- `scope="any"` — in this order: (1) a leading fence, as before; else
  (2) the first fenced block anywhere in the text (prose before it);
  else (3) a bare object inside prose: the first `{…}` span that parses
  strictly to an object AND whose surrounding text holds no other
  brace and no `"key":` pattern (the `exact_object_present` rule of
  Addendum U's substance check — one complete object, prose around
  it); else (4) the text unchanged. It returns `(body, kind)` with
  `kind ∈ {none, leading_fence, fence_after_prose, bare_object_in_prose}`.

`score` gains `--scope any`, reporting both scopes side by side. `scan`
is unchanged. The stdlib-only, single-file property is kept.

## Quantities — on `fence_corpus_2026-09-05` (2,130 outputs, five families)

Per record, the reference object under each scope (the fail-closed
`parse_json_object` after extraction):

- **newly credited** — records with no object under `leading` and an
  object under `any`, by kind and by slice;
- **divergence** — records with an object under both scopes that
  differ (must be zero: scope `any` only widens where `leading` found
  nothing);
- **recovered score** — for newly credited CORD records, field-level
  micro-F1 of the credited object against gold, versus the 0 they
  scored under the undercount; banked per record;
- **fence rate under `any`** — the S and T schema-only cells'
  fence rates recomputed with `kind != none` as "fenced", beside the
  banked leading-scope rates, so the R erratum's "undercount, never
  overcount" statement is quantified.

## Frozen readings

- **W1 — divergence.** Zero → the extension is a pure widening; any
  divergence is a bug in the extension, published and fixed before the
  addendum is read.
- **W2 — size of the undercount.** `newly credited / n_noref_leading`
  < 0.10 → "the undercount was under a tenth of the outputs the
  instrument rejected, on this corpus"; ≥ 0.10 → stated at size, and
  the Addendum S and T rows gain a dated note carrying the `any`-scope
  rates beside the frozen ones.
- **W3 — fence rates.** If any S/T schema-only cell's fence rate moves
  by ≥ 0.05 under `any`, the T row's exception sentences are re-checked
  against the T thresholds with the `any` rates and both results are
  published; the frozen reading is not re-taken.

**Prediction, written before the run:** newly credited between 5 and
15 of 2,130, most of them `bare_object_in_prose`; zero divergences;
no S/T fence rate moves by ≥ 0.05.

## What this cannot show

The corpus is this project's own banked output; the shapes the wider
scope catches are the ones this project's models produced. A wider
scope is a policy choice for the reader, which is why it is opt-in and
why no frozen reading is re-taken under it.

Artifact: `experiments/instrument_undercount_2026-09-08.json`. Runner:
`scripts/instrument_undercount.py`.

## Errata — 2026-09-08, after the run

1. **W3 was first tested on floats.** `0.97 - 0.92` is `0.0499…` in
   floating point, and the first run printed STABLE for a move that is
   exactly 0.05 (Falcon3 schema-only, 92 → 97 of 100). The threshold is
   now tested on counts (`20 × |Δ| ≥ n`) and the reading fires MOVES; the
   T re-check with the wider rate is banked and leaves Falcon3 at (a).
   Pinned by a test at 92 → 97 and 92 → 96.
2. **"kind ≠ none counts as fenced" conflates a bare object in prose with
   a fence.** The letter is applied as frozen (97 for Falcon3, one of
   them a bare object) and the fences-only count (96) is banked beside
   it; both are printed in the W3 sentence.
3. **The prediction missed low.** Written before the run: 5–15 newly
   credited of 2,130. Measured: 3 (two fences after prose, one bare
   object in prose). Zero divergences and no ≥ 0.05 fences-only move
   held; the letter's W3 fired on the bare-object count. Published as
   written.
4. **Closure convention.** `strip_fence` counts an unclosed leading
   fence as fenced; the after-prose block rule requires a closing
   fence. Falcon3's four outputs credited only by the letter open a
   fence and never close it, so under the leading scope's own
   convention the cell would read 100/100; the 96-vs-97 distinction in
   erratum 2 is a tool-convention artifact. No reading changes. The
   T re-check now uses the wider-scope k-shot rate as well (0/80 for
   every family under both scopes, banked).
5. **The bare-object rule is U's rule plus a no-key-pattern condition**
   — stricter than the text said, in the non-flattering direction.
6. **The human-readable `score --scope any` output did not print the
   wider-scope fields** until a review ran it without `--json`; fixed
   the same day and shown in the row.
