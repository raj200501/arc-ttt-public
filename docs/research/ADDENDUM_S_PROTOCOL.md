# Addendum S — protocol fixed before any arm runs

**Date: 2026-09-01.** The Addendum S preregistration row in `VERDICT.md`
(frozen 2026-08-25) states the frame: fence RATES on CORD, models
`Qwen2.5-0.5B-Instruct` and `Qwen2.5-1.5B-Instruct`, regimes
schema-only vs k-shot, shipped scorer, greedy, CPU, readings (a)/(b)/(c)
with thresholds 0.50 and 0.10. This note fixes what that row left
unspecified, before any S data exists. Where this note deviates from
the row's letter, the deviation and its reason are stated here, dated,
rather than discovered by a reader.

## The schema-only instruction (verbatim, both models)

CORD gold is nested and per-receipt keys vary, so the waybill wording
("exactly these keys, no others" over a flat list) cannot transfer
verbatim. The instruction below keeps the waybill sentence shape and
derives the key vocabulary from the corpus gold — the same source the
waybill arm used (`scale_rung_arm.py` reads the field list from gold):

```
Extract the receipt fields from the document and return them as a
single JSON object. Use only these keys:
- menu: a list of objects; keys among: cnt, discountprice, nm, num, price, sub, unitprice, vatyn
- sub_total: an object; keys among: discount_price, etc, service_price, subtotal_price, tax_price
- total: an object; keys among: cashprice, changeprice, creditcardprice, emoneyprice, menuqty_cnt, menutype_cnt, total_etc, total_price
Omit any key the document does not support.
```

The vocabulary is asserted in the runner against a fresh derivation
from `demo/cord_validation.jsonl` at run time; drift refuses to run.
`void_menu` and `void_total` appear in zero of the 100 gold parses and
are therefore not offered.

## Demonstrations and evaluation sets

- **Schema arm: all 100 receipts** — the row's letter.
- **k-shot arm: k=20, demonstrations = the E6 train split**
  (`experiments/ladder_e6_cord_split/train.jsonl`, SHA-banked, chosen
  by seed-1 shuffle before E6 ran), **evaluated over the 80 non-demo
  receipts**. This deviates from the row's "over 100 receipts": a
  receipt cannot serve as a demonstration of itself, and every prior
  k-shot arm in this tree (waybill and `cord_scale_run.py` alike) keeps
  demonstrations disjoint from evaluation. The fence thresholds are
  RATES and apply unchanged at n=80.
- Prompt construction: k-shot through `run_challenge.build_task` +
  `text_task_to_messages`, the same constructors as every banked k-shot
  arm; schema as a single user turn via the chat template.

## Decode and classification

Greedy, `max_new_tokens=512`, `max_seq=8192`, float32 (the banked 0.5B
and 1.5B convention on this box), per-document checkpoints. Fence
classification by the SHIPPED `tools/fencecheck.py` `strip_fence` —
leading-fence scope, the same instrument as Addendum R, with the same
known undercount property R's erratum documents.

## How readings combine across two models

The row defines readings per pair (f_schema, f_kshot) and lists two
models without saying how they combine. Fixed now, in the
non-flattering direction:

- **(a) IT REPLICATES** may be claimed only if (a) fires at BOTH
  sizes.
- **(c) IT DOES NOT REPLICATE** at EITHER size triggers the row's full
  re-scoping commitment (the word *industry* comes out; *on our
  corpus* goes back in), even if the other size shows (a).
- Any other combination publishes as (b)/mixed: all four rates, no
  headline movement.

Readings are applied by arithmetic in the runner
(`scripts/cord_fence_tax.py`), never re-derived from the shape of the
data (the Addendum R lesson). Artifact: the row's frozen pointer
`experiments/cord_fence_tax_2026-08-25.json`, with honest internal run
dates (the pointer name is the preregistration's, the run is 09-01).
Per-cell raw outputs bank incrementally under
`experiments/cord_fence_tax_cells/`.
