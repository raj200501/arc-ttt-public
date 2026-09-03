"""JSON-constrained greedy decoding (engineering ladder II, rung E7).

The E6 decomposition (docs/research/ADAPTATION_ENGINEERING_LADDER_II.md)
found that every invalid output the adapted 3B produced on CORD was a
SYNTAX fault -- single-quoted keys, an extra closing brace, a second
object appended -- and that those faults carried 55% of its loss. This
decoder makes such outputs unreachable: at each step the top-k
candidate tokens are tried in logit order and the first whose text
keeps the output a valid JSON *prefix* is emitted. If none qualifies,
the top-1 token is emitted anyway (the decoder degrades to greedy and
counts the fallback), so it can never stall and never invents text.

It is schema-blind on purpose: it enforces JSON syntax only, so the
prompted and adapted arms receive an identical decoder and the ADAPT
reading credits nothing but the adapter.

The prefix validator is a small incremental checker over the decoded
STRING, not a token grammar. It tracks: string / escape state, the
open-container stack, whether a value is expected, and completion of
the root value (after which only whitespace is allowed). It rejects
single-quote delimiters, bare identifiers, trailing commas at closers,
and anything after the root closes. Numbers and literals are checked
as prefixes of {true,false,null,number}.
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
