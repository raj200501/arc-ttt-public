#!/usr/bin/env python3
"""Addendum X: the decoder, isolated -- one loop, the constraint toggled.

    PYTHONPATH=src python3 scripts/cord_decoder_isolation.py --bank-configs
    PYTHONPATH=src python3 scripts/cord_decoder_isolation.py --determinism qwen2.5-0.5b
    PYTHONPATH=src python3 scripts/cord_decoder_isolation.py --constrained granite-2b
    PYTHONPATH=src python3 scripts/cord_decoder_isolation.py --cell qwen2.5-0.5b
    PYTHONPATH=src python3 scripts/cord_decoder_isolation.py --generate-neutral qwen2.5-0.5b
    PYTHONPATH=src python3 scripts/cord_decoder_isolation.py --read

Preregistration: docs/research/ADDENDUM_X_PROTOCOL.md (frozen 2026-09-16,
before the plain arm ran on any family).

Addendum V compared a constrained arm against `model.generate` cells and
so varied three things at once -- the constraint, the decoding path, and
(on one family) the checkpoint's `generation_config` defaults. It could
attribute; it could not measure. This runner separates them:

  arm C  constrained  : V's banked cells, reused under a determinism gate
                        (re-run here for a family whose prompt is not
                        reproducible -- see RENDER_DATE below)
  arm P  plain        : THE SAME FUNCTION with enforce=False   [new]
  arm G  neutralised  : model.generate with the config's modifiers off [new]
  arm G0 defaults     : V's comparator cells, already banked

Every cell is checkpointed per document and keyed by config, so a reboot
resumes rather than restarts. The reader withholds until every
preregistered cell exists AND the determinism gate has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import cord_fence_tax as cft  # noqa: E402
import cord_constrained_families as v  # noqa: E402

N_DOCS = v.N_DOCS
TOP_K = v.TOP_K
MAX_NEW_TOKENS = cft.MAX_NEW_TOKENS
MAX_SEQ = cft.MAX_SEQ
DETERMINISM_DOCS = 5          # frozen in the protocol, before the run
V_CELLS = v.CELLS_DIR
CELLS = v.CELLS                # family -> (model_id, dtype, comparator cell)
OUT_DIR = REPO / "experiments" / "cord_decoder_isolation_cells"
OUT = REPO / "experiments" / "cord_decoder_isolation_2026-09-16.json"
CONFIGS = REPO / "experiments" / "generation_configs_2026-09-16.json"
WORK = REPO / "work" / "x"
V_ARTIFACT = REPO / "experiments" / "cord_constrained_families_2026-09-08.json"

# --------------------------------------------------------------------------
# a prompt that does not depend on what day it is
# --------------------------------------------------------------------------
# Granite's chat template builds its own system message and puts
# strftime_now('%B %d, %Y') into it, so the prompt this repository has been
# sending that checkpoint changed every single day. Addendum V's constrained
# cell and Addendum T's comparator cell for Granite were produced five days
# apart and therefore did NOT carry the same prompt -- a confound nobody
# noticed, inside the family V reported as byte-identical between its arms.
# Found by audit before this addendum froze.
#
# Every Addendum X arm renders that family's system message from this frozen
# string instead, so X's own prompts are identical across its arms and
# reproducible by a stranger on any future day. The string is verified at run
# time to reproduce the template's own output exactly, differing only in the
# date, and the run refuses to start if it stops doing so.
RENDER_DATE = "September 16, 2026"
FROZEN_SYSTEM = {
    "granite-2b": ("Knowledge Cutoff Date: April 2024.\nToday's Date: "
                   + RENDER_DATE + ".\nYou are Granite, developed by IBM. "
                   "You are a helpful AI assistant."),
}
CLOCK_CALLS = ("strftime_now", "datetime.now", "date.today", "utcnow")
_DATE_IN_TEMPLATE = re.compile(r"[A-Z][a-z]+ \d{1,2}, \d{4}")

# The keys that make `generate` do something other than plain greedy.
# Fixed here, before the run: a family gets arm G if and only if its
# checkpoint's generation_config sets one of these to a non-neutral value
# (or its prompt is not reproducible -- see needs_generate_arm).
MODIFIER_KEYS = (
    "repetition_penalty", "encoder_repetition_penalty", "temperature",
    "top_p", "top_k", "typical_p", "epsilon_cutoff", "eta_cutoff",
    "no_repeat_ngram_size", "min_length", "min_new_tokens", "length_penalty",
    "diversity_penalty", "do_sample", "num_beams", "penalty_alpha",
    "exponential_decay_length_penalty", "suppress_tokens",
    "begin_suppress_tokens", "forced_bos_token_id", "forced_eos_token_id",
    "renormalize_logits", "bad_words_ids", "sequence_bias", "guidance_scale",
)
NEUTRAL = {"repetition_penalty": 1.0, "encoder_repetition_penalty": 1.0,
           "temperature": 1.0, "top_p": 1.0, "typical_p": 1.0,
           "length_penalty": 1.0, "diversity_penalty": 0.0,
           "no_repeat_ngram_size": 0, "min_length": 0, "min_new_tokens": None,
           "do_sample": False, "num_beams": 1, "epsilon_cutoff": 0.0,
           "eta_cutoff": 0.0, "penalty_alpha": None, "renormalize_logits": None,
           "suppress_tokens": None, "begin_suppress_tokens": None,
           "forced_bos_token_id": None, "forced_eos_token_id": None,
           "exponential_decay_length_penalty": None, "bad_words_ids": None,
           "sequence_bias": None, "guidance_scale": None, "top_k": 50}


def v_regressions() -> dict:
    """What Addendum V published, read from V's own artifact rather than
    retyped here. X3 narrows one of V's sentences if the number moves, so
    the number being narrowed must have exactly one referent."""
    rows = json.loads(V_ARTIFACT.read_text(encoding="utf-8"))["rows"]
    return {r["family"]: r["regressions"] for r in rows}


def modifiers_of(cfg: dict, baseline: dict | None = None) -> dict:
    """The modifiers a checkpoint actually ships.

    `GenerationConfig.to_dict()` fills in the library's own defaults, so a
    naive read reports every key as set and every family as needing arm G.
    Two rules fix that, and both are conservative in the direction that
    makes us run MORE arms rather than fewer: a value of `None` is unset,
    and a value equal to the library's default for that key is what
    `generate` would do anyway. `baseline` is `GenerationConfig().to_dict()`
    when available; the frozen NEUTRAL table backs it up so this function is
    testable without transformers installed. `None` entries in a supplied
    baseline are dropped rather than allowed to mask a real setting.
    """
    base = dict(NEUTRAL)
    if baseline:
        base.update({k: val for k, val in baseline.items()
                     if k in MODIFIER_KEYS and val is not None})
    out = {}
    for key in MODIFIER_KEYS:
        if key not in cfg:
            continue
        val = cfg[key]
        if val is None:                      # unset
            continue
        if key in base and val == base[key]:  # what generate does regardless
            continue
        out[key] = val
    return out


def clock_calls_in(template) -> list:
    """Which clock functions a chat template calls. A template that reads
    the clock renders a different prompt every day, so a cell banked from it
    is not reproducible and cannot be reused as a comparator."""
    return [c for c in CLOCK_CALLS if c in (template or "")]


def needs_generate_arm(cfg: dict, baseline=None, clock_dependent: bool = False) -> bool:
    """Arm G runs if the config carries a modifier that can change greedy
    output, OR if the family's prompt is clock-dependent -- because then the
    banked arm G0 was rendered under a different prompt and cannot serve as
    the path comparator. `top_k` alone cannot change greedy output (no
    sampling), so it does not trigger."""
    if clock_dependent:
        return True
    mods = modifiers_of(cfg, baseline)
    mods.pop("top_k", None)
    return bool(mods)


def _environment() -> dict:
    """Banked on every cell this addendum writes, so a difference between two
    cells can be attributed rather than guessed. Addendum V's cells carry
    none of this, which is why its arms could not be compared on anything
    but their text."""
    import platform
    import torch
    import transformers
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "render_date_pinned_for_clock_dependent_templates": RENDER_DATE,
    }


# --------------------------------------------------------------------------
# banking the scope -- before any arm runs, so it cannot follow the results
# --------------------------------------------------------------------------

def bank_configs(force: bool = False) -> int:
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    baseline = GenerationConfig().to_dict()
    families = {}
    for family, (model_id, dtype_name, _comp) in CELLS.items():
        gcfg = GenerationConfig.from_pretrained(model_id).to_dict()
        tok = AutoTokenizer.from_pretrained(model_id)
        clock = clock_calls_in(getattr(tok, "chat_template", None))
        mods = modifiers_of(gcfg, baseline)
        families[family] = {
            "model": model_id,
            "dtype": dtype_name,
            "modifiers": mods,
            "chat_template_clock_calls": clock,
            "prompt_is_reproducible": not clock,
            "arm_C_reused_from_V": not clock,
            "arm_G_runs": needs_generate_arm(gcfg, baseline, bool(clock)),
            "architectures": getattr(AutoConfig.from_pretrained(model_id),
                                     "architectures", None),
        }
        print(f"{family:14s} modifiers={mods or '{}'}  clock={clock or '[]'}  "
              f"arm_G={'YES' if families[family]['arm_G_runs'] else 'no'}  "
              f"arm_C={'reused' if not clock else 'RE-RUN'}")
    record = {
        "what": "Each family's shipped generation_config and chat-template clock "
                "dependence, read before any Addendum X arm ran. Which arms run on "
                "which family is decided by this record -- by the checkpoints, not "
                "by the results.",
        "protocol": "docs/research/ADDENDUM_X_PROTOCOL.md",
        "modifier_keys_considered": list(MODIFIER_KEYS),
        "clock_calls_considered": list(CLOCK_CALLS),
        "library_defaults_used_as_the_baseline": {
            k: baseline[k] for k in MODIFIER_KEYS if k in baseline},
        "why_a_baseline_is_needed": (
            "A modifier is a value that differs from what generate would do with "
            "no config at all. Two things can hide that: a checkpoint key set to "
            "None (unset, not a modifier) and a key set to the library's own "
            "default. On the transformers pinned here GenerationConfig().to_dict() "
            "returns None for every key in this list, so the None rule does the "
            "work and the frozen NEUTRAL table below is what actually supplies the "
            "defaults; on a library version that reports real defaults the "
            "baseline takes over. Both are banked so a reader can tell which "
            "applied."),
        "neutral_table_used": {k: NEUTRAL[k] for k in MODIFIER_KEYS if k in NEUTRAL},
        "why_top_k_alone_does_not_count": (
            "top_k is applied by a sampling warper. With do_sample=False no warper "
            "runs, so a top_k in the config cannot change greedy output. It is "
            "reported and excluded from the trigger."),
        "why_clock_dependence_changes_the_arms": (
            "A chat template that calls the clock renders a different prompt every "
            "day. Addendum V's constrained cell and Addendum T's comparator cell "
            "for such a family were produced on different days and so did not "
            "carry the same prompt. Neither banked cell can be reused, so arm C is "
            "re-run here under a pinned prompt and arm G is run to give X2 a "
            "prompt-identical generate arm."),
        "environment": _environment(),
        "families": families,
    }
    if CONFIGS.exists() and not force:
        existing = json.loads(CONFIGS.read_text(encoding="utf-8"))["families"]
        watched = ("modifiers", "chat_template_clock_calls", "arm_G_runs",
                   "arm_C_reused_from_V")
        drift = sorted(f for f in families
                       if any(existing.get(f, {}).get(k) != families[f][k] for k in watched))
        if drift:
            raise SystemExit(
                "the banked scope record disagrees with the checkpoints as they are "
                f"now, in: {', '.join(drift)}. This record IS the preregistered scope "
                "of the addendum; rewriting it silently would let the scope follow "
                "the checkpoints, or the results. Re-run with --force-configs only "
                "to bank a scope no arm has yet run under, and say so in the "
                "protocol's errata.")
        print("scope record unchanged; nothing rewritten.")
        return 0
    CONFIGS.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"\nbanked: {CONFIGS.relative_to(REPO)}")
    return 0


def _scope() -> dict:
    if not CONFIGS.exists():
        raise SystemExit("bank the generation configs first (--bank-configs): the "
                         "scope rule must be fixed before any arm runs.")
    return json.loads(CONFIGS.read_text(encoding="utf-8"))["families"]


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

def _messages(family: str, text: str) -> list:
    """The Addendum S/T/V schema-only prompt. For a family whose template
    reads the clock, the system message the template would have built is
    supplied explicitly from the frozen string, so the prompt does not change
    with the calendar."""
    user = {"role": "user", "content": cft.SCHEMA_INSTRUCTION + "\n\n" + text}
    frozen = FROZEN_SYSTEM.get(family)
    return ([{"role": "system", "content": frozen}] if frozen else []) + [user]


def _assert_frozen_system_reproduces(tokenizer, family: str) -> None:
    """The frozen system message must be exactly what the template builds,
    differing only in the date. Checked at run time so a template change
    fails the run instead of quietly altering the prompt."""
    if family not in FROZEN_SYSTEM:
        return
    probe = [{"role": "user", "content": "PROBE"}]
    auto = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
    frozen = tokenizer.apply_chat_template(
        [{"role": "system", "content": FROZEN_SYSTEM[family]}] + probe,
        tokenize=False, add_generation_prompt=True)
    found = _DATE_IN_TEMPLATE.search(auto)
    if not found:
        raise SystemExit(f"{family}: expected a date in the auto-rendered template; found none")
    if auto.replace(found.group(0), RENDER_DATE) != frozen:
        raise SystemExit(
            f"{family}: the frozen system message no longer reproduces the chat "
            "template's own output. The template changed, so the prompt this "
            "addendum sends would differ from the one it documents. Refusing.")


def _prompt_ids(tokenizer, family: str, text: str):
    """Returns the ids and a digest of the rendered prompt. The digest is
    banked per document per arm so that prompt identity BETWEEN the arms X
    compares is checked by the reader rather than assumed by a docstring."""
    rendered = tokenizer.apply_chat_template(
        _messages(family, text), tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, return_tensors="pt").input_ids
    if ids.shape[1] > MAX_SEQ:
        raise SystemExit(f"prompt exceeded {MAX_SEQ}")
    return ids, hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


def _load(family: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id, dtype_name, _comp = CELLS[family]
    torch.set_num_threads(4)
    torch.manual_seed(1)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _assert_frozen_system_reproduces(tokenizer, family)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype_name))
    model.eval()
    return model, tokenizer, model_id, dtype_name


def _checkpoint(path: pathlib.Path, config_key: str) -> dict:
    done: dict = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row.get("config") != config_key:
            raise SystemExit(f"checkpoint {path} belongs to {row.get('config')!r}; refusing")
        done[row["id"]] = row
    return done


def _bank(path: pathlib.Path, record: dict) -> None:
    record.setdefault("environment", _environment())
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(f"banked: {path.name}")


# --------------------------------------------------------------------------
# the decoding arms -- one loop, one boolean, plus generate
# --------------------------------------------------------------------------

def _decode_cell(family: str, *, enforce: bool, cell_path: pathlib.Path,
                 tag: str, what: str) -> int:
    """Arms C and P share this body. The ONLY difference between them is the
    `enforce` argument handed to the one decoder function."""
    from arcttt.constrained_json import constrained_greedy_generate
    cft._assert_vocab()
    if cell_path.exists():
        print(f"cell banked already: {cell_path.name}")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    holdout = v._holdout()
    model, tokenizer, model_id, dtype_name = _load(family)

    config_key = (f"{model_id}|schema|{dtype_name}|enforce={enforce}|topk={TOP_K}"
                  f"|mnt={MAX_NEW_TOKENS}|seq={MAX_SEQ}|n={N_DOCS}|render={RENDER_DATE}|X")
    ckpt_path = WORK / f"{family}_{tag}.ckpt.jsonl"
    done = _checkpoint(ckpt_path, config_key)
    if done:
        print(f"[X:{family}:{tag}] resuming {len(done)}/{len(holdout)}", flush=True)

    predictions, seconds, decode, tokens, digests = {}, {}, {}, {}, {}
    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        for i, row in enumerate(holdout):
            if row["id"] in done:
                d = done[row["id"]]
                predictions[row["id"]], seconds[row["id"]] = d["raw"], d["seconds"]
                decode[row["id"]], tokens[row["id"]] = d["decode"], d["token_ids"]
                digests[row["id"]] = d["prompt_sha256_16"]
                continue
            ids, digest = _prompt_ids(tokenizer, family, row["text"])
            began = time.monotonic()
            res = constrained_greedy_generate(
                model, tokenizer, ids, max_new_tokens=MAX_NEW_TOKENS,
                top_k=TOP_K, enforce=enforce)
            took = round(time.monotonic() - began, 1)
            if not enforce and (res.constrained_steps or res.fallbacks):
                raise SystemExit(
                    f"{family}/{row['id']}: enforce=False recorded "
                    f"{res.constrained_steps} constrained steps and {res.fallbacks} "
                    "fallbacks. The unconstrained arm must never take either "
                    "branch; the flag is not a toggle.")
            acct = {"fallbacks": res.fallbacks, "constrained_steps": res.constrained_steps,
                    "steps": res.steps, "stopped_on": res.stopped_on}
            predictions[row["id"]], seconds[row["id"]] = res.text, took
            decode[row["id"]], tokens[row["id"]] = acct, res.token_ids
            digests[row["id"]] = digest
            ckpt.write(json.dumps({"config": config_key, "id": row["id"], "raw": res.text,
                                   "seconds": took, "decode": acct,
                                   "token_ids": res.token_ids,
                                   "prompt_sha256_16": digest}, ensure_ascii=False) + "\n")
            ckpt.flush()
            print(f"[X:{family}:{tag}] {row['id']} {i + 1}/{len(holdout)} {took}s "
                  f"steps={res.steps} stop={res.stopped_on}", flush=True)

    _bank(cell_path, {
        "what": what,
        "protocol": "docs/research/ADDENDUM_X_PROTOCOL.md (frozen 2026-09-16)",
        "arm": "C" if enforce else "P",
        "model": model_id, "dtype": dtype_name, "regime": "schema",
        "n": len(holdout), "k": 0, "schema_instruction": cft.SCHEMA_INSTRUCTION,
        "system_message": FROZEN_SYSTEM.get(family),
        "decoder": {"kind": "constrained greedy" if enforce else "same loop, constraint off",
                    "enforce": enforce, "top_k": TOP_K,
                    "max_new_tokens": MAX_NEW_TOKENS, "allow_leading_fence": True,
                    "stop": "eos | complete | max_new_tokens"},
        "resumed_from_checkpoint": len(done),
        "per_document_seconds": seconds, "per_document_decode": decode,
        "per_document_token_ids": tokens, "per_document_prompt_sha256_16": digests,
        "predictions": predictions,
    })
    return 0


def run_plain(family: str) -> int:
    return _decode_cell(
        family, enforce=False, cell_path=OUT_DIR / f"{family}_plain.json", tag="plain",
        what=f"Addendum X arm P: {CELLS[family][0]}, the SAME decoder loop as the "
             "constrained arm with enforce=False -- top-1 always taken, EOS never "
             "suppressed, identical cap, fence tolerance and stop rule.")


def run_constrained(family: str) -> int:
    """Arm C, re-run rather than reused. Only for a family whose prompt is
    not reproducible from the banked cells (a clock-dependent template)."""
    if _scope()[family]["arm_C_reused_from_V"]:
        raise SystemExit(
            f"{family}'s prompt is reproducible, so the protocol reuses Addendum V's "
            "constrained cell under the determinism gate rather than re-running it.")
    return _decode_cell(
        family, enforce=True, cell_path=OUT_DIR / f"{family}_constrained.json",
        tag="constrained",
        what=f"Addendum X arm C: {CELLS[family][0]}, the constrained decoder re-run "
             "under the pinned prompt. NOT reused from Addendum V, because this "
             "checkpoint's chat template reads the clock, so V's cell and T's "
             "comparator were rendered on different days under different prompts.")


def run_generate_neutral(family: str) -> int:
    import torch
    from transformers import GenerationConfig
    cft._assert_vocab()
    scope = _scope()
    if not scope[family]["arm_G_runs"]:
        raise SystemExit(
            f"{family} ships no generation modifier and its prompt is reproducible, "
            "so arm G is arm G0 by construction and the protocol does not run it.")
    cell_path = OUT_DIR / f"{family}_generate_neutral.json"
    if cell_path.exists():
        print(f"cell banked already: {cell_path.name}")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    holdout = v._holdout()
    model, tokenizer, model_id, dtype_name = _load(family)

    # Start from the CHECKPOINT's own config and neutralise only the banked
    # modifiers, so everything else it ships -- eos_token_id lists, bos, pad --
    # survives. Building a fresh GenerationConfig instead would silently drop
    # those and make arm G differ from arm G0 in more than the modifiers.
    before = GenerationConfig.from_pretrained(model_id).to_dict()
    neutral = GenerationConfig.from_pretrained(model_id)
    changed = {}
    for key in scope[family]["modifiers"]:
        if key in NEUTRAL:
            setattr(neutral, key, NEUTRAL[key])
            changed[key] = {"was": before.get(key), "now": NEUTRAL[key]}
    neutral.max_new_tokens = MAX_NEW_TOKENS
    neutral.do_sample = False
    neutral.num_beams = 1
    if neutral.pad_token_id is None:
        neutral.pad_token_id = tokenizer.pad_token_id
    model.generation_config = neutral

    config_key = (f"{model_id}|schema|{dtype_name}|generate_neutralised"
                  f"|mnt={MAX_NEW_TOKENS}|seq={MAX_SEQ}|n={N_DOCS}"
                  f"|render={RENDER_DATE}|X")
    ckpt_path = WORK / f"{family}_generate_neutral.ckpt.jsonl"
    done = _checkpoint(ckpt_path, config_key)
    if done:
        print(f"[X:{family}:neutral] resuming {len(done)}/{len(holdout)}", flush=True)

    predictions, seconds, tokens, digests = {}, {}, {}, {}
    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        for i, row in enumerate(holdout):
            if row["id"] in done:
                d = done[row["id"]]
                predictions[row["id"]], seconds[row["id"]] = d["raw"], d["seconds"]
                tokens[row["id"]], digests[row["id"]] = d["token_ids"], d["prompt_sha256_16"]
                continue
            ids, digest = _prompt_ids(tokenizer, family, row["text"])
            began = time.monotonic()
            with torch.no_grad():
                out = model.generate(input_ids=ids,
                                     attention_mask=torch.ones_like(ids),
                                     generation_config=neutral)
            took = round(time.monotonic() - began, 1)
            gen = out[0][ids.shape[1]:].tolist()
            text = tokenizer.decode(gen, skip_special_tokens=True).strip()
            predictions[row["id"]], seconds[row["id"]] = text, took
            tokens[row["id"]], digests[row["id"]] = gen, digest
            ckpt.write(json.dumps({"config": config_key, "id": row["id"], "raw": text,
                                   "seconds": took, "token_ids": gen,
                                   "prompt_sha256_16": digest}, ensure_ascii=False) + "\n")
            ckpt.flush()
            print(f"[X:{family}:neutral] {row['id']} {i + 1}/{len(holdout)} {took}s "
                  f"tokens={len(gen)}", flush=True)

    _bank(cell_path, {
        "what": f"Addendum X arm G: {model_id}, model.generate(do_sample=False) with the "
                "checkpoint's own generation_config loaded and ONLY its banked modifiers "
                "set to neutral values. Arm G0 ran the same call with those modifiers "
                "in force.",
        "protocol": "docs/research/ADDENDUM_X_PROTOCOL.md (frozen 2026-09-16)",
        "arm": "G", "model": model_id, "dtype": dtype_name, "regime": "schema",
        "n": len(holdout), "k": 0, "schema_instruction": cft.SCHEMA_INSTRUCTION,
        "system_message": FROZEN_SYSTEM.get(family),
        "modifiers_neutralised": changed,
        "effective_generation_config": neutral.to_dict(),
        "resumed_from_checkpoint": len(done),
        "per_document_seconds": seconds, "per_document_token_ids": tokens,
        "per_document_prompt_sha256_16": digests, "predictions": predictions,
    })
    return 0


# --------------------------------------------------------------------------
# the determinism gate on the reused arms
# --------------------------------------------------------------------------

def run_determinism(family: str) -> int:
    import torch
    from arcttt.constrained_json import constrained_greedy_generate
    cft._assert_vocab()
    scope = _scope()
    gate_path = OUT_DIR / f"{family}_determinism.json"
    if gate_path.exists():
        print(f"gate banked already: {gate_path.name}")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not scope[family]["arm_C_reused_from_V"]:
        _bank(gate_path, {
            "what": f"Addendum X determinism gate: NOT APPLICABLE to {family}. Its chat "
                    "template reads the clock, so the banked cells were rendered under "
                    "prompts this run cannot reproduce. Arm C is re-run instead of "
                    "reused and there is nothing here to gate.",
            "protocol": "docs/research/ADDENDUM_X_PROTOCOL.md (frozen 2026-09-16)",
            "model": CELLS[family][0], "n": 0, "mismatches": 0, "passed": True,
            "not_applicable_because": scope[family]["chat_template_clock_calls"],
            "documents": [],
        })
        return 0
    banked_cell = V_CELLS / f"{family}_schema_constrained.json"
    comparator = CELLS[family][2]
    if not banked_cell.exists():
        raise SystemExit(f"Addendum V cell missing: {banked_cell}")
    banked_c = json.loads(banked_cell.read_text())["predictions"]
    banked_g0 = json.loads(comparator.read_text())["predictions"]
    holdout = v._holdout()[:DETERMINISM_DOCS]
    WORK.mkdir(parents=True, exist_ok=True)
    config_key = (f"{CELLS[family][0]}|gate|{CELLS[family][1]}|n={DETERMINISM_DOCS}"
                  f"|mnt={MAX_NEW_TOKENS}|render={RENDER_DATE}|X")
    ckpt_path = WORK / f"{family}_gate.ckpt.jsonl"
    done = _checkpoint(ckpt_path, config_key)
    if len(done) >= len(holdout):
        rows = [done[r["id"]]["row"] for r in holdout]
        mismatches = sum(not r["arm_C_identical"] for r in rows) + \
            sum(not r["arm_G0_identical"] for r in rows)
        model = tokenizer = None
        model_id, dtype_name = CELLS[family][0], CELLS[family][1]
    else:
        model, tokenizer, model_id, dtype_name = _load(family)
        if done:
            print(f"[X:{family}:gate] resuming {len(done)}/{len(holdout)}", flush=True)
        rows, mismatches = [], 0
        with open(ckpt_path, "a", encoding="utf-8") as ckpt:
            for row in holdout:
                if row["id"] in done:
                    r = done[row["id"]]["row"]
                    rows.append(r)
                    mismatches += (not r["arm_C_identical"]) + (not r["arm_G0_identical"])
                    continue
                ids, _digest = _prompt_ids(tokenizer, family, row["text"])
                res = constrained_greedy_generate(model, tokenizer, ids,
                                                  max_new_tokens=MAX_NEW_TOKENS, top_k=TOP_K)
                # Arm G0 is reused too, so it is gated on the same documents, with
                # the comparator's own call reproduced verbatim (config in force).
                with torch.no_grad():
                    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id)
                g0_text = tokenizer.decode(out[0][ids.shape[1]:],
                                           skip_special_tokens=True).strip()
                same_c = res.text == banked_c[row["id"]]
                same_g0 = g0_text == banked_g0[row["id"]]
                mismatches += (not same_c) + (not same_g0)
                r = {"id": row["id"], "arm_C_identical": same_c,
                     "arm_G0_identical": same_g0,
                     "rerun_C_text": None if same_c else res.text,
                     "rerun_G0_text": None if same_g0 else g0_text}
                rows.append(r)
                ckpt.write(json.dumps({"config": config_key, "id": row["id"], "row": r},
                                      ensure_ascii=False) + "\n")
                ckpt.flush()
                print(f"[X:{family}:gate] {row['id']} C={'OK' if same_c else 'DIFFERS'} "
                      f"G0={'OK' if same_g0 else 'DIFFERS'}", flush=True)

    _bank(gate_path, {
        "what": f"Addendum X determinism gate: the first {DETERMINISM_DOCS} documents of "
                f"{family} re-decoded on BOTH reused arms -- the constrained arm "
                "(Addendum V's cell) and the generate comparator (Addendum S/T's cell) -- "
                "and compared byte-for-byte. Both arms are reused rather than re-run; this "
                "is the check that makes the reuse legitimate, and it is a sample, not a "
                "proof.",
        "protocol": "docs/research/ADDENDUM_X_PROTOCOL.md (frozen 2026-09-16)",
        "model": model_id, "dtype": dtype_name, "n": len(rows),
        "mismatches": mismatches, "passed": mismatches == 0, "documents": rows,
    })
    if mismatches:
        print(f"GATE FAILED for {family}: {mismatches} mismatch(es). The reading "
              "withholds; the non-determinism is the result.")
    return 0


# --------------------------------------------------------------------------
# frozen readings -- arithmetic only
# --------------------------------------------------------------------------

def x1_reading(i_plain: int, i_constrained: int, regressions: int,
               identical: int, n: int) -> str:
    """X1, per family, in the protocol's order; first match wins."""
    if identical == n:
        return "CONSTRAINT INERT"
    if i_constrained > i_plain or regressions >= 3:
        return f"CONSTRAINT HURTS (invalid {i_plain} -> {i_constrained}, regressions {regressions})"
    if i_plain <= 1:
        # Nothing to remove. Without this branch `i_constrained <= i_plain // 2`
        # is satisfied by 0 <= 0 and the family reads REMOVES where the
        # constraint removed nothing -- the flattering direction, and the guard
        # Addendum V had ("UNTESTABLE AT SIZE") that this reading first dropped.
        return "UNTESTABLE AT SIZE (plain invalid <= 1)"
    if i_constrained <= i_plain // 2 and regressions == 0:
        return "CONSTRAINT REMOVES"
    if i_constrained < i_plain and regressions == 0:
        return "CONSTRAINT HELPS"
    if i_constrained < i_plain and 1 <= regressions <= 2:
        return f"CONSTRAINT MIXED (regressions {regressions})"
    return "CONSTRAINT NEUTRAL"


def x1_combine(per_family: dict) -> str:
    n_fam = len(per_family)
    removes = sorted(f for f, r in per_family.items() if r.startswith("CONSTRAINT REMOVES"))
    hurts = sorted(f for f, r in per_family.items() if r.startswith("CONSTRAINT HURTS"))
    inert = sorted(f for f, r in per_family.items() if r.startswith("CONSTRAINT INERT"))
    untestable = sorted(f for f, r in per_family.items() if r.startswith("UNTESTABLE"))
    if hurts:
        if not removes:
            return (f"X1 FAILS IN {', '.join(hurts)}: CONSTRAINT REMOVES fires in 0 of "
                    f"{n_fam} families, so there is no sentence to narrow -- the claim "
                    f"is withdrawn, not scoped. Per-family: {json.dumps(per_family)}")
        return (f"X1 EXCEPTION IN {', '.join(hurts)}: named at full size; the sentence "
                f"becomes 'on {len(removes)} of the {n_fam} families tested' -- never "
                f"'across families'. Per-family: {json.dumps(per_family)}")
    if len(removes) >= 4:
        return (f"X1 HOLDS: the constraint removes invalid JSON in {len(removes)} of "
                f"{n_fam} families against its own decoding path, and no family is an "
                f"exception. Per-family: {json.dumps(per_family)}")
    return (f"X1 MIXED: CONSTRAINT REMOVES in {len(removes)} of {n_fam} families "
            f"({len(inert)} inert -- the constraint never fired there, which is not a "
            f"success; {len(untestable)} untestable at size -- nothing to remove, "
            f"counted neither way). All counts publish, no headline. "
            f"Per-family: {json.dumps(per_family)}")


def x2_reading(divergences: int, basis: str = "token ids") -> str:
    """The frozen vocabulary stays the prefix of the string; where the test
    was run on decoded text rather than token ids the string says so, so a
    weaker result can never be read as the stronger one."""
    verdict = ("PATH REPRODUCES" if divergences == 0
               else f"PATH DIVERGES ({divergences})")
    if basis == "token ids":
        return verdict
    return verdict + " (WEAKER TEST: decoded text, not token ids)"


def x3_reading(explained: int, v_regressions: int, reproduces: bool = True) -> str:
    """Does V's published attribution survive? The branch that costs us is
    written first, and a regression count that does not reproduce preempts
    both -- a disagreement about the number is read before the attribution."""
    if not reproduces:
        return ("WITHHELD: V's published regression count for this family does not "
                "reproduce from its own banked cells. That disagreement is read "
                "before the attribution and publishes first.")
    if explained < v_regressions:
        return (f"V'S ATTRIBUTION IS TOO STRONG: {explained} of {v_regressions} "
                f"regressions are explained by the generation defaults; the residual "
                f"{v_regressions - explained} is not. V's sentence narrows by erratum.")
    return (f"V'S ATTRIBUTION HOLDS: all {v_regressions} regressions are explained by "
            "the generation defaults rather than by the constraint.")


def _prefix_divergence(a, b, a_stopped_on: str = "complete"):
    """Where the plain arm (a) and the generate arm (b) part company.

    The two loops stop differently -- ours as soon as the root closes and
    parses, `generate` only at EOS or the cap -- so the plain arm is EXPECTED
    to be the shorter one, and "a is a prefix of b" is agreement, not
    divergence.

    Two cases the bare prefix rule gets wrong, both in the direction that
    flatters us: the generate arm stopping FIRST is never expected; and if
    the plain arm stopped because the model asked to stop (`eos`), the
    generate arm saw the same logits and must stop in the same place, so an
    empty plain arm against a long generate arm is not agreement.

    Returns None for agreement, else the first index at which they part.
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(b) < len(a):          # generate stopped FIRST: never expected
        return len(b)
    if a_stopped_on == "eos" and len(b) != len(a):
        return len(a)            # ours took EOS here; generate kept going
    return None


def _text_prefix_divergence(a: str, b: str, a_stopped_on: str = "complete"):
    """The same rule on decoded text, for the families where no token ids
    were banked for the generate arm. Weaker, and named as weaker."""
    return _prefix_divergence(a, b, a_stopped_on)


def _arm_c_path(family: str, scope: dict) -> pathlib.Path:
    return (V_CELLS / f"{family}_schema_constrained.json"
            if scope[family]["arm_C_reused_from_V"]
            else OUT_DIR / f"{family}_constrained.json")


def read() -> int:
    from arcttt.scoring import field_micro_f1, parse_json_object
    from arcttt.text_task import TextTaskFormatError
    fc = cft._fc()
    scope = _scope()

    # The gate is checked FIRST, and it is terminal. A failed gate voids the
    # reuse the whole design rests on, so no later cell can rescue the reading
    # and there is nothing to be gained by decoding more of them. Ordering the
    # two checks this way changes no reading -- both withhold -- it only stops
    # the runner from spending hours on cells it has already been told not to
    # read.
    gates = {f: json.loads((OUT_DIR / f"{f}_determinism.json").read_text())
             for f in CELLS if (OUT_DIR / f"{f}_determinism.json").exists()}
    failed = sorted(f for f, g in gates.items() if not g["passed"])
    if failed:
        record = {
            "what": "Addendum X: WITHHELD. The determinism gate failed, so the reuse "
                    "of the banked arms is void and none of X1, X2 or X3 is read. "
                    "The gate failure is the result this addendum returns.",
            "protocol": "docs/research/ADDENDUM_X_PROTOCOL.md (frozen 2026-09-16)",
            "date": "2026-09-16",
            "environment": _environment(),
            "reading": "WITHHELD BY THE DETERMINISM GATE",
            "failed_families": failed,
            "determinism_gate": {f: {"n": g["n"], "mismatches": g["mismatches"],
                                     "passed": g["passed"],
                                     "documents": g["documents"]}
                                 for f, g in gates.items()},
            "cells_that_exist": sorted(p.name for p in OUT_DIR.glob("*.json")),
            "cells_not_run": sorted(
                f"{f}_{suffix}.json"
                for f in CELLS
                for suffix in ("plain", "generate_neutral")
                if (suffix != "generate_neutral" or scope[f]["arm_G_runs"])
                and not (OUT_DIR / f"{f}_{suffix}.json").exists()),
            "why_the_remaining_cells_were_not_run": (
                "A failed gate is terminal by the frozen protocol: it voids the reuse "
                "every contrast depends on. Decoding the remaining cells could not "
                "change the withholding, so the compute was not spent. What was and "
                "was not run is listed above rather than left to be inferred."),
            "disclosure_the_operator_owes_the_reader": (
                "While the run was in progress the per-family X1 quantities for the "
                "families that passed the gate were computed by hand, to decide which "
                "cells to schedule next on a box that is reclaimed on idle. They are "
                "NOT published, the reader does not compute them, and no sentence "
                "anywhere in this repository rests on them. This note exists so that "
                "a successor addendum reading the same cells cannot be mistaken for "
                "an author who already knew the answer and wrote a protocol to fit."),
            "what_this_does_not_say": (
                "Nothing here reads on the constraint, the decoding path or the "
                "generation defaults. The banked cells for the families that passed "
                "the gate are raw data for a successor addendum, not results of this "
                "one, and this artifact deliberately does not compute their readings."),
            "how_to_recompute": "PYTHONPATH=src python3 scripts/cord_decoder_isolation.py --read",
        }
        OUT.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("WITHHELD BY THE DETERMINISM GATE: " + ", ".join(failed) +
              ". A reused arm does not reproduce, and that non-determinism is the "
              "result. See the banked gate cells.")
        print(f"banked: {OUT.relative_to(REPO)}")
        return 1

    missing = []
    for family in CELLS:
        for path in (OUT_DIR / f"{family}_plain.json",
                     OUT_DIR / f"{family}_determinism.json",
                     _arm_c_path(family, scope)):
            if not path.exists():
                missing.append(path.name)
        if scope[family]["arm_G_runs"] and not (OUT_DIR / f"{family}_generate_neutral.json").exists():
            missing.append(f"{family}_generate_neutral.json")
    if missing:
        print("WITHHELD: missing cells " + ", ".join(sorted(missing)) +
              " -- the reading is stated once every preregistered cell exists.")
        return 1

    gold = {json.loads(l)["id"]: json.loads(l)["gold"] for l in
            (cft.SPLIT_DIR / "gold.jsonl").read_text().splitlines() if l.strip()}
    for r in cft._read_jsonl(cft.SPLIT_DIR / "train.jsonl"):
        gold[r["id"]] = r["gold"]

    def classify(text):
        body, fenced = fc.strip_fence(text)
        try:
            return parse_json_object(body), fenced
        except TextTaskFormatError:
            return None, fenced

    rows, x1_per_family, x2_per_family = [], {}, {}
    x3 = None
    prompt_mismatches = []
    for family, (model_id, dtype_name, comparator) in CELLS.items():
        plain = json.loads((OUT_DIR / f"{family}_plain.json").read_text())
        const = json.loads(_arm_c_path(family, scope).read_text())
        g0 = json.loads(comparator.read_text())
        ids = list(plain["predictions"])

        # Prompt identity between the arms X compares: checked, not assumed.
        if "per_document_prompt_sha256_16" in const:
            prompt_mismatches += [
                f"{family}/{d} (C vs P)" for d in ids
                if const["per_document_prompt_sha256_16"][d]
                != plain["per_document_prompt_sha256_16"][d]]

        # ---- X1: the constraint, isolated (C against P, the same loop)
        i_p = i_c = regressions = identical = 0
        wins = losses = ties = 0
        deltas, differing_docs, zero_step_differs, nonzero_step_identical = [], [], [], []
        fen_p = fen_c = 0
        for doc_id in ids:
            op, fp = classify(plain["predictions"][doc_id])
            oc, fcd = classify(const["predictions"][doc_id])
            i_p += op is None
            i_c += oc is None
            fen_p += fp
            fen_c += fcd
            regressions += (op is not None and oc is None)
            same = plain["predictions"][doc_id] == const["predictions"][doc_id]
            identical += same
            steps_here = const["per_document_decode"][doc_id]["constrained_steps"]
            if not same:
                differing_docs.append(doc_id)
                if steps_here == 0:
                    zero_step_differs.append(doc_id)
            elif steps_here > 0:
                # equally impossible in the other direction: the validator
                # rejected a top-1 token here, so the arms cannot be identical
                nonzero_step_identical.append(doc_id)
            sp = field_micro_f1(op, gold[doc_id]) if op is not None else 0.0
            sc = field_micro_f1(oc, gold[doc_id]) if oc is not None else 0.0
            d = sc - sp
            deltas.append(d)
            wins += d > 1e-9
            losses += d < -1e-9
            ties += abs(d) <= 1e-9
        x1 = x1_reading(i_p, i_c, regressions, identical, len(ids))
        x1_per_family[family] = x1

        # The prediction written before the run, checked PER DOCUMENT. Summing
        # a family's constrained steps makes the check vacuous wherever the
        # validator fired at all -- Falcon3 fired on 7 of 50, so a family-wide
        # sum would have excused its other 43 documents.
        c_steps = sum(a["constrained_steps"] for a in const["per_document_decode"].values())
        accounting_ok = not zero_step_differs and not nonzero_step_identical

        # ---- X2: the implementation path (P against G, or G0 where G == G0)
        has_g = scope[family]["arm_G_runs"]
        if has_g:
            g = json.loads((OUT_DIR / f"{family}_generate_neutral.json").read_text())
            other, other_name = g["predictions"], "G"
            prompt_mismatches += [
                f"{family}/{d} (P vs G)" for d in ids
                if g["per_document_prompt_sha256_16"][d]
                != plain["per_document_prompt_sha256_16"][d]]
            div = {d: at for d in ids
                   if (at := _prefix_divergence(
                       plain["per_document_token_ids"][d],
                       g["per_document_token_ids"][d],
                       plain["per_document_decode"][d]["stopped_on"])) is not None}
            comparison, basis = "P vs G (token prefix)", "token ids"
        else:
            other = {d: g0["predictions"][d].strip() for d in ids}
            other_name = "G0"
            div = {d: at for d in ids
                   if (at := _text_prefix_divergence(
                       plain["predictions"][d], other[d],
                       plain["per_document_decode"][d]["stopped_on"])) is not None}
            comparison, basis = "P vs G0 (text prefix; G == G0, no modifiers)", "decoded text"
        i_other = sum(classify(other[d])[0] is None for d in ids)
        x2 = x2_reading(len(div), basis)
        x2_per_family[family] = x2

        mean_d = round(sum(deltas) / len(deltas), 4)
        p = v._sign_test(wins, losses) if mean_d >= 0 else v._sign_test(losses, wins)
        rows.append({
            "family": family, "model": model_id, "dtype": dtype_name, "n": len(ids),
            "arm_C_source": str(_arm_c_path(family, scope).relative_to(REPO)),
            "x1_invalid_plain": i_p, "x1_invalid_constrained": i_c,
            "x1_regressions_plain_valid_constrained_invalid": regressions,
            "x1_identical_documents": identical, "x1_differing_documents": differing_docs,
            "x1_reading": x1,
            "constrained_steps_banked": c_steps,
            "documents_with_zero_steps_but_differing_arms": zero_step_differs,
            "documents_with_steps_but_identical_arms": nonzero_step_identical,
            "fenced_plain": fen_p, "fenced_constrained": fen_c,
            "accounting_consistent": accounting_ok,
            "x2_comparison": comparison, "x2_basis": basis,
            "x2_compared_against": other_name,
            "x2_invalid_plain": i_p, "x2_invalid_other_arm": i_other,
            "x2_invalid_difference_other_minus_plain": i_other - i_p,
            "x2_divergent_documents": len(div), "x2_first_divergence_index": div,
            "x2_reading": x2,
            "score_mean_delta_constrained_minus_plain": mean_d,
            "score_wins_losses_ties": [wins, losses, ties],
            "score_sign_test_p": round(p, 4),
        })
        print(f"{family:14s} X1 invalid {i_p:2d} -> {i_c:2d} (identical {identical}/{len(ids)}, "
              f"reg {regressions})  {x1:34s} | X2 {x2:18s} vs {other_name} "
              f"(invalid {i_other}) | accounting {'ok' if accounting_ok else 'INCONSISTENT'}")

        # ---- X3: the defaults, on the family that ships one
        if scope[family]["modifiers"]:
            g = json.loads((OUT_DIR / f"{family}_generate_neutral.json").read_text())
            d_def = sum(g["predictions"][d] != g0["predictions"][d].strip() for d in ids)
            v_reg = v_regressions().get(family)
            detail = []
            for doc_id in ids:
                o_g0, _ = classify(g0["predictions"][doc_id])
                oc, _ = classify(const["predictions"][doc_id])
                if o_g0 is not None and oc is None:      # one of V's regressions
                    o_g, _ = classify(g["predictions"][doc_id])
                    detail.append({"id": doc_id,
                                   "valid_without_the_defaults": o_g is not None})
            recomputed = len(detail)
            explained = sum(not d["valid_without_the_defaults"] for d in detail)
            reproduces = (v_reg is None or recomputed == v_reg)
            # The residual V did not explain is attributed on the same documents:
            # if arms C and P differ there, the constraint owns it; if they agree,
            # the difference is elsewhere in the path.
            residual = [d["id"] for d in detail if d["valid_without_the_defaults"]]
            residual_attribution = {
                doc_id: ("the constraint (arms C and P differ on this document)"
                         if plain["predictions"][doc_id] != const["predictions"][doc_id]
                         else "the path (arms C and P agree; the difference is elsewhere)")
                for doc_id in residual}
            x3 = {
                "family": family,
                "documents_differing_G_vs_G0": d_def,
                "v_published_regressions": v_reg,
                "v_regressions_recomputed_here": recomputed,
                "v_regression_count_reproduces": reproduces,
                "v_regressions_reexamined": detail,
                "regressions_explained_by_the_defaults": explained,
                "residual_documents": residual,
                "residual_attributed_by_arithmetic": residual_attribution,
                "reading": (x3_reading(explained, recomputed, reproduces)
                            if v_reg is not None else
                            "no published regression count for this family"),
                "note_if_v_does_not_reproduce": (
                    None if reproduces else
                    f"V published {v_reg} regressions for this family; recomputing the "
                    f"same quantity from the same two banked cells gives {recomputed}. "
                    "That disagreement is a finding in its own right and is read before "
                    "anything else in X3."),
            }
            print(f"{'':14s} X3 G vs G0 differ on {d_def}/{len(ids)}; {x3['reading']}")

    x1_finding = x1_combine(x1_per_family)
    inconsistent = sorted(r["family"] for r in rows if not r["accounting_consistent"])
    record = {
        "what": "Addendum X: the constraint, the decoding path and the generation "
                "defaults separated on the same fifty documents. Arm C and arm P are "
                "the SAME function with one boolean changed; arm G is generate with "
                "the checkpoint's own config and only its modifiers neutralised; arm "
                "G0 is Addendum V's comparator.",
        "protocol": "docs/research/ADDENDUM_X_PROTOCOL.md (frozen 2026-09-16, before "
                    "the plain arm ran on any family)",
        "date": "2026-09-16",
        "environment": _environment(),
        "determinism_gate": {f: {"n": g["n"], "mismatches": g["mismatches"],
                                 "passed": g["passed"]} for f, g in gates.items()},
        "prompt_identity_between_compared_arms": (
            "verified per document by sha256 of the rendered prompt"
            if not prompt_mismatches else prompt_mismatches),
        "rows": rows,
        "x1_finding": x1_finding,
        "x2_per_family": x2_per_family,
        "x3": x3,
        "decoder_accounting_inconsistent_in": inconsistent,
        "the_predictions_written_before_the_run": [
            "X1: every DOCUMENT with 0 banked constrained steps has byte-identical "
            "arms C and P; Falcon3 differs on exactly the 7 documents where the "
            "validator fired. A zero-step document whose arms differ means the "
            "decoder's own accounting is wrong, and that would be the result.",
            "X2: PATH DIVERGES on Phi-3 (V saw 32 of 50 bodies differ at bfloat16); "
            "PATH REPRODUCES on SmolLM2 and Granite. Qwen and Falcon3 unpredicted.",
        ],
        "how_to_recompute": "PYTHONPATH=src python3 scripts/cord_decoder_isolation.py --read",
    }
    OUT.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{x1_finding}")
    if prompt_mismatches:
        print(f"\nPROMPT MISMATCH between compared arms: {prompt_mismatches[:5]} ...")
    if inconsistent:
        print(f"\nDECODER ACCOUNTING INCONSISTENT IN {', '.join(inconsistent)} -- a "
              "document with 0 constrained steps whose arms differ. This is the result.")
    print(f"\nbanked: {OUT.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-configs", action="store_true")
    ap.add_argument("--force-configs", action="store_true")
    ap.add_argument("--cell", choices=sorted(CELLS))
    ap.add_argument("--constrained", choices=sorted(CELLS))
    ap.add_argument("--generate-neutral", choices=sorted(CELLS))
    ap.add_argument("--determinism", choices=sorted(CELLS))
    ap.add_argument("--read", action="store_true")
    args = ap.parse_args()
    if args.bank_configs or args.force_configs:
        return bank_configs(force=args.force_configs)
    if args.determinism:
        return run_determinism(args.determinism)
    if args.constrained:
        return run_constrained(args.constrained)
    if args.cell:
        return run_plain(args.cell)
    if args.generate_neutral:
        return run_generate_neutral(args.generate_neutral)
    if args.read:
        return read()
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
