#!/usr/bin/env python3
"""jsongreedy -- JSON-constrained greedy decoding for any Hugging Face causal LM.

One file, no dependencies beyond torch and transformers. At each step the
top-k candidate tokens are tried in logit order and the first whose text
keeps the output a valid JSON *prefix* is emitted; if none qualifies the
top-1 token is emitted anyway (the decoder degrades to greedy and counts
the fallback), so it never stalls and never invents text. It is
schema-blind on purpose: it enforces JSON syntax only.

    python3 jsongreedy.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --prompt "Return {\"city\": ..., \"zip\": ...} for: 90210 Beverly Hills"

    from jsongreedy import generate
    text, info = generate(model, tokenizer, [{"role": "user", "content": prompt}])

Measured, before you trust it: on this repository's CORD receipts the
decoder removed every invalid output on both Qwen2.5-3B arms (Ladder II,
rung E7 -- single-quoted keys and extra closers; 11 of 12 removals carry
a constrained step). Re-run on five families' schema-only prompts
(Addendum V, `VERDICT.md`) no family cleared the bar: the remaining
invalid outputs are truncations at the token cap, which a syntax
constraint cannot close; on Falcon3 it fixed six of seven syntax or
prose faults (14 -> 8, one short of the bar) and elsewhere it never
fired. It addresses one fault class. A valid object is not a correct one.

The decoding core below is byte-identical to `src/arcttt/constrained_json.py`,
the module every banked run used; `tests/test_jsongreedy.py` pins that.

MIT licensed. Copy the single file into your repo if that is easier than
depending on it.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Callable, Iterable

_LITERALS = ("true", "false", "null")
_NUM_CHARS = set("0123456789+-.eE")


@dataclasses.dataclass
class _State:
    stack: list  # of "{" or "[" with a phase marker
    in_string: bool = False
    escape: bool = False
    # phase per container: for "{": "key_or_close" | "key" | "colon" |
    # "value" | "comma_or_close"; for "[": "value_or_close" | "value" |
    # "comma_or_close"
    token: str = ""  # current bare literal/number being assembled
    done: bool = False  # root value complete
    started: bool = False


def _phase(stack: list) -> str | None:
    return stack[-1][1] if stack else None


def _set_phase(stack: list, phase: str) -> None:
    stack[-1] = (stack[-1][0], phase)


def _valid_scalar_prefix(tok: str) -> bool:
    if not tok:
        return True
    if any(lit.startswith(tok) for lit in _LITERALS):
        return True
    if all(c in _NUM_CHARS for c in tok):
        # a loose but sufficient number-prefix check
        if tok in ("-", "+"):
            return tok == "-"
        try:
            # fencecheck: ignore -- asking whether a bare token is a number
            # PREFIX while decoding; not scoring model output.
            float(tok.rstrip("eE+-."))
            return tok[0] != "+"
        except ValueError:
            return False
    return False


def _scalar_complete(tok: str) -> bool:
    if tok in _LITERALS:
        return True
    try:
        # fencecheck: ignore -- asking whether a scalar token is complete
        # while decoding; not scoring model output.
        json.loads(tok)
        return isinstance(json.loads(tok), (int, float))
    except (ValueError, TypeError):
        return False


def is_json_prefix(text: str) -> bool:
    """True iff `text` can be extended to a valid JSON document whose
    root is an object or array, with strict JSON syntax."""
    st = _State(stack=[])
    for ch in text:
        if st.done:
            if ch in " \t\r\n":
                continue
            return False
        if st.in_string:
            if st.escape:
                st.escape = False
                continue
            if ch == "\\":
                st.escape = True
                continue
            if ch == '"':
                st.in_string = False
                ph = _phase(st.stack)
                if ph == "key":
                    _set_phase(st.stack, "colon")
                elif ph in ("value", "value_or_close"):
                    _set_phase(st.stack, "comma_or_close")
                elif ph is None:
                    return False  # a bare string root is not allowed here
            continue
        # not in a string
        if st.token:
            if ch in _NUM_CHARS or ch.isalpha():
                st.token += ch
                if not _valid_scalar_prefix(st.token):
                    return False
                continue
            # token ended
            if not _scalar_complete(st.token):
                return False
            st.token = ""
            _set_phase(st.stack, "comma_or_close")
            # fall through to handle ch
        if ch in " \t\r\n":
            continue
        ph = _phase(st.stack)
        if ph is None:
            if st.started:
                return False
            if ch == "{":
                st.stack.append(("{", "key_or_close")); st.started = True
            elif ch == "[":
                st.stack.append(("[", "value_or_close")); st.started = True
            else:
                return False
            continue
        if ph in ("key_or_close", "key"):
            if ch == '"':
                st.in_string = True
                _set_phase(st.stack, "key")
            elif ch == "}" and ph == "key_or_close":
                st.stack.pop()
                if st.stack:
                    _set_phase(st.stack, "comma_or_close")
                else:
                    st.done = True
            else:
                return False
        elif ph == "colon":
            if ch == ":":
                _set_phase(st.stack, "value")
            else:
                return False
        elif ph in ("value", "value_or_close"):
            if ch == '"':
                st.in_string = True
            elif ch == "{":
                st.stack.append(("{", "key_or_close"))
            elif ch == "[":
                st.stack.append(("[", "value_or_close"))
            elif ch == "]" and ph == "value_or_close":
                st.stack.pop()
                if st.stack:
                    _set_phase(st.stack, "comma_or_close")
                else:
                    st.done = True
            elif ch in _NUM_CHARS or ch.isalpha():
                st.token = ch
                if not _valid_scalar_prefix(st.token):
                    return False
            else:
                return False
        elif ph == "comma_or_close":
            kind = st.stack[-1][0]
            if ch == ",":
                _set_phase(st.stack, "key" if kind == "{" else "value")
            elif (ch == "}" and kind == "{") or (ch == "]" and kind == "["):
                st.stack.pop()
                if st.stack:
                    _set_phase(st.stack, "comma_or_close")
                else:
                    st.done = True
            else:
                return False
        else:  # pragma: no cover
            return False
    if st.token and not _valid_scalar_prefix(st.token):
        return False
    return True


def is_complete_json(text: str) -> bool:
    try:
        # fencecheck: ignore -- the decoder's own stop condition ("is the
        # root closed and parseable"); the failure means KEEP DECODING,
        # it never becomes a score.
        obj = json.loads(text)
    except ValueError:
        return False
    return isinstance(obj, (dict, list))


@dataclasses.dataclass
class ConstrainedResult:
    text: str
    fallbacks: int          # steps where no top-k candidate kept validity
    constrained_steps: int  # steps where the top-1 token was rejected
    steps: int
    stopped_on: str         # "eos" | "complete" | "max_new_tokens"


def constrained_greedy_generate(model, tokenizer, input_ids, *,
                                max_new_tokens: int = 512, top_k: int = 16,
                                allow_leading_fence: bool = True,
                                validator: Callable[[str], bool] = is_json_prefix,
                                ) -> ConstrainedResult:
    """Greedy decoding with a JSON-prefix constraint over the decoded text.

    `allow_leading_fence` tolerates a ```json / ``` opener before the
    root (the fence is stripped before validation and, at the end,
    reported as text -- the symmetric fence policy still applies to
    what is banked), so a model that habitually opens with a fence is
    constrained on the JSON inside it rather than fought on the fence.
    Stops at EOS, at max_new_tokens, or as soon as the root closes and
    the text parses (a complete document needs no more tokens).
    """
    import torch

    device = input_ids.device
    eos = tokenizer.eos_token_id
    generated: list[int] = []
    text = ""
    fallbacks = constrained_steps = 0
    stopped_on = "max_new_tokens"
    past = None
    cur = input_ids
    attn = torch.ones_like(input_ids)

    def body(t: str) -> str:
        if not allow_leading_fence:
            return t
        s = t.lstrip()
        if s.startswith("```"):
            nl = s.find("\n")
            return "" if nl == -1 else s[nl + 1:]
        if "```".startswith(s) and len(s) < 3:
            return ""  # a fence still being typed
        return t

    with torch.no_grad():
        for step in range(max_new_tokens):
            out = model(input_ids=cur, attention_mask=attn,
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
            cand = torch.topk(logits, top_k).indices.tolist()
            chosen = None
            for rank, tok_id in enumerate(cand):
                if tok_id == eos:
                    if is_complete_json(body(text)):
                        chosen, stopped_on = tok_id, "eos"
                        break
                    continue  # EOS before the document is complete
                trial = tokenizer.decode(generated + [tok_id],
                                         skip_special_tokens=True)
                inner = body(trial)
                if validator(inner):
                    chosen = tok_id
                    if rank > 0:
                        constrained_steps += 1
                    break
            if chosen is None:
                chosen = cand[0]
                fallbacks += 1
            if chosen == eos:
                break
            generated.append(chosen)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            if is_complete_json(body(text)):
                stopped_on = "complete"
                break
            cur = torch.tensor([[chosen]], device=device)
            attn = torch.cat([attn, torch.ones((1, 1), device=device,
                                               dtype=attn.dtype)], dim=1)
    return ConstrainedResult(text=text.strip(), fallbacks=fallbacks,
                             constrained_steps=constrained_steps,
                             steps=len(generated), stopped_on=stopped_on)


# ---------------------------------------------------------------- convenience
def generate(model, tokenizer, prompt, *, max_new_tokens: int = 512, top_k: int = 16,
             allow_leading_fence: bool = True, add_generation_prompt: bool = True):
    """Decode a JSON object from `prompt` with the constrained greedy decoder.

    `prompt` is either a string (used verbatim) or a list of chat messages
    (rendered through the tokenizer's own chat template). Returns
    (text, info) where info carries the decode accounting: fallbacks,
    constrained_steps, steps, stopped_on.
    """
    import torch
    if isinstance(prompt, str):
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    else:
        rendered = tokenizer.apply_chat_template(prompt, tokenize=False,
                                                 add_generation_prompt=add_generation_prompt)
        ids = tokenizer(rendered, return_tensors="pt").input_ids
    ids = ids.to(next(model.parameters()).device)
    res = constrained_greedy_generate(model, tokenizer, ids, max_new_tokens=max_new_tokens,
                                      top_k=top_k, allow_leading_fence=allow_leading_fence)
    return res.text, {"fallbacks": res.fallbacks, "constrained_steps": res.constrained_steps,
                      "steps": res.steps, "stopped_on": res.stopped_on}


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="jsongreedy", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="HF model id or local path (a causal LM)")
    ap.add_argument("--prompt", required=True, help="the user turn; rendered with the model's chat template")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    ap.add_argument("--json", action="store_true", help="print {text, info} as JSON")
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype))
    model.eval()
    text, info = generate(model, tok, [{"role": "user", "content": a.prompt}],
                          max_new_tokens=a.max_new_tokens, top_k=a.top_k)
    if a.json:
        print(json.dumps({"text": text, "info": info}, ensure_ascii=False, indent=1))
    else:
        print(text)
        print(f"# steps={info['steps']} constrained_steps={info['constrained_steps']} "
              f"fallbacks={info['fallbacks']} stopped_on={info['stopped_on']}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
