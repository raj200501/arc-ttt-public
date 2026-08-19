"""Per-task test-time training and generation.

The solver is model-agnostic behind ``Predictor``; the real implementation
wraps a causal LM with a fresh LoRA adapter per task (the NVARC shape:
adapt on leave-one-out augmented demonstrations, then generate the test
output in each augmentation's frame). Everything here is CPU-runnable with a
tiny model so the pipeline is testable offline; scale comes from swapping the
base model and device, not from changing this code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from arcttt.augment import Augmentation
from arcttt.serialize import (
    ChatTurn,
    grid_to_text,
    task_to_messages,
    text_to_grid,
    ttt_training_examples,
)
from arcttt.tasks import Grid, Task, TaskFormatError



def _template_ids(rendered: Any) -> torch.Tensor:
    """Normalize ``apply_chat_template(..., return_tensors="pt")`` output.

    transformers 4.x returns the id tensor directly; 5.x returns a
    BatchEncoding whose bare attribute access raises an empty
    ``AttributeError`` (the cord-scale kernel incident, 2026-08-11 — same
    pinned-image API-drift class as the v7 cache incident). Probe the shape,
    not the version string.
    """

    ids = rendered.input_ids if hasattr(rendered, "input_ids") else rendered
    return cast(torch.Tensor, ids)


class Predictor(Protocol):
    """Adapts to one task, then predicts and scores in augmented frames."""

    def adapt(self, task: Task, augmentations: Sequence[Augmentation]) -> None: ...

    def predict(self, task: Task, test_index: int, samples: int) -> list[Grid]: ...

    def log_probability(self, task: Task, test_index: int, output: Grid) -> float: ...


@dataclass(frozen=True)
class TTTConfig:
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 5e-5
    epochs: int = 1
    max_new_tokens: int = 992  # 30 rows * (30 digits + newline) + margin
    max_sequence_tokens: int = 8192
    temperature: float = 0.7
    raw_qwen_format: bool = False  # champion-style <|im_start|> framing, no system turn
    gradient_checkpointing: bool = False
    use_dfs: bool = False  # constrained DFS decoding instead of sampled generation
    dfs_probability_cutoff: float = 0.2  # keep completions whose TOTAL probability exceeds this (DFS prunes at cumulative NLL -ln(cutoff))
    dfs_max_candidates: int = 32
    shuffle_examples: bool = False  # permute demonstration order per augmentation
    ttt_batch_size: int = 1  # examples per optimizer step (padded batch)
    dfs_time_budget_seconds: float | None = None  # wall-clock cap per predict()
    dfs_include_greedy: bool = True  # always add the greedy completion; the
    # cumulative-NLL cutoff can exclude even the argmax path, and greedy
    # guarantees one candidate per augmentation frame for the voting pool.
    chunked_loss_tokens: int = 0  # 0 = HF labels-path loss (legacy, exact).
    # N > 0 = compute the SAME shifted fp32 cross-entropy in N-token slices
    # of the sequence with a two-phase backward, so the seq x vocab logits
    # tensor never materializes whole. Mathematically identical (the loss is
    # per-token additive; mean = sum / count either way); exists because a
    # 7.5k-token training sequence's full logits over a 152k vocab OOM a T4
    # (observed: Addendum B k=30 adapted arms, 2026-08-15). Guarded by
    # tests/test_chunked_loss.py, which pins loss AND gradients against the
    # labels path.


def turns_to_raw_qwen(turns: Sequence[ChatTurn], add_generation_prompt: bool) -> str:
    """Champion-format serialization: no system turn, no inter-turn newlines."""

    text = "".join(
        f"<|im_start|>{turn.role}\n{turn.content}<|im_end|>" for turn in turns
    )
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


def turns_to_chat(turns: Sequence[ChatTurn]) -> list[dict[str, str]]:
    return [{"role": turn.role, "content": turn.content} for turn in turns]


class CausalLMPredictor:
    """LoRA-per-task TTT over a HuggingFace causal LM. Device-agnostic."""

    # Search telemetry, CLASS-level on purpose: the kernel builds a fresh
    # predictor per task (and per OOM-ladder level), so instance counters
    # would reset ~240 times and measure nothing. These accumulate across the
    # whole worker process and are dumped once at the end of the run.
    dfs_stop_reasons: dict[str, int] = {}
    dfs_candidates_found: int = 0
    dfs_frames_searched: int = 0

    @classmethod
    def dfs_telemetry(cls) -> dict[str, Any]:
        """Snapshot of why searches stopped and how much they found."""

        frames = cls.dfs_frames_searched
        return {
            "frames_searched": frames,
            "stop_reasons": dict(cls.dfs_stop_reasons),
            "candidates_found_total": cls.dfs_candidates_found,
            "candidates_per_frame_mean": (
                round(cls.dfs_candidates_found / frames, 2) if frames else 0.0
            ),
        }

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: TTTConfig,
        device: torch.device,
    ) -> None:
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.model: Any = model
        self._grid_vocab: Any = None

    def _pad_id(self) -> int:
        """Padding token id with None-based fallback (a legitimate pad id of 0
        must NOT fall through to eos — truthiness would drop it)."""

        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        return 0 if pad is None else int(pad)

    # -- adaptation ---------------------------------------------------------

    def adapt(self, task: Task, augmentations: Sequence[Augmentation]) -> None:
        examples: list[tuple[ChatTurn, ...]] = []
        for augmentation_index, augmentation in enumerate(augmentations):
            transformed = augmentation.apply_task(task)
            shuffle_seed = (
                augmentation_index if self.config.shuffle_examples else None
            )
            examples.extend(ttt_training_examples(transformed, shuffle_seed))
        self.adapt_on_examples(examples)

    def adapt_on_examples(self, examples: Sequence[Sequence[ChatTurn]]) -> None:
        """Inject a fresh LoRA adapter and train it on supervised chat examples.

        Task-agnostic core of ``adapt``: each example is a chat-turn sequence
        whose final assistant turn is the supervised completion (grid or text
        alike — the text-mode TTT path calls this directly)."""

        from arcttt.lora import inject_lora, lora_parameters, remove_lora

        remove_lora(self.base_model)  # drop any adapter from a prior task
        inject_lora(
            self.base_model,
            self.config.lora_rank,
            self.config.lora_alpha,
            use_rslora=True,
        )
        self.model = self.base_model
        if self.config.gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            # Frozen embeddings mean the checkpointed segment has no input
            # requiring grad; this hook makes the embedding output require it.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()
        self.model.train()
        # v8 forensics closed the case on HF's gradient_checkpointing_enable
        # in this image: flags read True everywhere, yet ~430 MB/layer of
        # activations stay resident (11.3 GB at 5.4k tokens) - a no-op at
        # layer level. When the chunked path (and therefore long-sequence
        # training) is in play, checkpoint each decoder layer OURSELVES with
        # torch.utils.checkpoint, preserving HF's forward orchestration.
        # Wrappers are restored in the finally below so eval and generation
        # see the original forwards.
        wrapped_layers: list[tuple[object, object]] = []
        if self.config.chunked_loss_tokens > 0 and self.config.gradient_checkpointing:
            layers = getattr(getattr(self.model, "model", None), "layers", None)
            if layers is not None:
                from torch.utils.checkpoint import checkpoint as _torch_checkpoint

                def _wrap(fwd):
                    def wrapped(*args, **kwargs):
                        return _torch_checkpoint(
                            fwd, *args, use_reentrant=False, **kwargs
                        )

                    return wrapped

                for layer in layers:
                    wrapped_layers.append((layer, layer.forward))
                    layer.forward = _wrap(layer.forward)
                print(
                    f"manually checkpointed {len(wrapped_layers)} decoder layers",
                    flush=True,
                )
        try:
            optimizer = torch.optim.AdamW(
                lora_parameters(self.model), lr=self.config.learning_rate
            )
            encoded = [
                batch
                for turns in examples
                if (batch := self._encode(turns, supervise_final=True)) is not None
            ]
            dropped = len(examples) - len(encoded)
            if dropped:
                print(
                    f"adapt_on_examples: dropped {dropped}/{len(examples)} examples over "
                    f"max_sequence_tokens={self.config.max_sequence_tokens}",
                    flush=True,
                )
            if not encoded:
                print(
                    "adapt_on_examples: ALL examples dropped — zero optimizer steps, "
                    "adapter is a no-op (predictions = base model)",
                    flush=True,
                )
            batch_size = max(1, self.config.ttt_batch_size)
            if batch_size > 1:
                # Sort by length so padded batches stay dense (batch_size 1 keeps
                # the original example order and exact legacy behavior).
                encoded.sort(key=lambda pair: pair[0].shape[1])
            pad_id = self._pad_id()
            for _ in range(self.config.epochs):
                for start in range(0, len(encoded), batch_size):
                    chunk = encoded[start : start + batch_size]
                    if len(chunk) == 1:
                        input_ids, labels = chunk[0]
                        attention_mask = torch.ones_like(input_ids)
                    else:
                        width = max(ids.shape[1] for ids, _ in chunk)
                        input_ids = torch.full(
                            (len(chunk), width), pad_id, dtype=torch.long
                        )
                        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
                        attention_mask = torch.zeros(
                            (len(chunk), width), dtype=torch.long
                        )
                        for row, (ids, label_row) in enumerate(chunk):
                            length = ids.shape[1]
                            input_ids[row, :length] = ids[0].cpu()
                            labels[row, :length] = label_row[0].cpu()
                            attention_mask[row, :length] = 1
                        input_ids = input_ids.to(self.device)
                        labels = labels.to(self.device)
                        attention_mask = attention_mask.to(self.device)
                    optimizer.zero_grad()
                    if self.config.chunked_loss_tokens > 0:
                        self._chunked_loss_backward(input_ids, attention_mask, labels)
                    else:
                        loss = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        ).loss
                        loss.backward()
                    optimizer.step()
        finally:
            for layer, original_forward in wrapped_layers:
                layer.forward = original_forward
            if self.config.gradient_checkpointing and hasattr(
                self.model, "gradient_checkpointing_disable"
            ):
                self.model.gradient_checkpointing_disable()  # cached generation next
            self.model.eval()

    def _chunked_loss_backward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """The HF causal-LM loss and its gradients, without whole-seq logits.

        Reproduces exactly what ``forward(labels=...)`` computes - shift by
        one, cross-entropy in fp32, mean over non-ignored tokens - but applies
        the lm_head to ``chunked_loss_tokens``-sized slices of the hidden
        states, backpropagating each slice's SUM loss scaled by the global
        token count. Per-token CE is additive, so slice sums divided by the
        one global count equal the whole-sequence mean, and the accumulated
        gradients are equal too (autograd sums across backward calls).

        Two-phase backward: slices push gradients into a detached copy of the
        hidden states (freeing each slice's logits before the next), then one
        trunk backward carries the accumulated hidden-state gradient through
        the LoRA parameters. The lm_head is frozen under LoRA, so skipping
        its weight gradient changes nothing.
        """

        trunk = getattr(self.model, "model", None)
        lm_head = getattr(self.model, "lm_head", None)
        if trunk is None or lm_head is None:  # architecture without the split
            loss = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            ).loss
            loss.backward()
            return
        # gradient_checkpointing_enable() is called on the wrapper; whether
        # the flag reaches the trunk that we call DIRECTLY is version-
        # dependent (11 GB of held activations at 5.4k tokens says it did
        # not, 2026-08-15). Assert-and-repair, loudly, once.
        if not getattr(self, "_ckpt_reported", False):
            self._ckpt_reported = True
            engaged = bool(getattr(trunk, "gradient_checkpointing", False))
            print(f"trunk gradient_checkpointing={engaged}", flush=True)
            if self.config.gradient_checkpointing and not engaged:
                trunk.gradient_checkpointing = True
                print("trunk gradient_checkpointing FORCED on", flush=True)
        # torch-level activation offload: the scoring image's transformers
        # ignores BOTH its own gradient_checkpointing_enable and a per-layer
        # torch.utils.checkpoint monkeypatch (verified: saved-activation
        # bytes match the no-checkpoint profile exactly, while the identical
        # mechanisms verifiably work in transformers 5.15 locally). save_on_cpu
        # is beneath the framework - every tensor autograd saves for backward
        # lives in host RAM instead of GPU memory, whatever HF does or does
        # not do. Gradient-identical (pinned in the local probe); costs one
        # PCIe round-trip per training example, noise next to generation.
        if input_ids.is_cuda:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                hidden = trunk(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
        else:
            hidden = trunk(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        detached = hidden.detach().requires_grad_(True)
        shifted_labels = labels[:, 1:]
        total = int((shifted_labels != -100).sum().item())
        if total == 0:
            return
        width = hidden.shape[1]
        step = self.config.chunked_loss_tokens
        for start in range(0, width - 1, step):
            end = min(width - 1, start + step)
            logits = lm_head(detached[:, start:end])
            slice_loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                shifted_labels[:, start:end].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            (slice_loss / total).backward()
        assert detached.grad is not None
        hidden.backward(detached.grad)

    # -- inference ----------------------------------------------------------

    def predict(self, task: Task, test_index: int, samples: int) -> list[Grid]:
        turns = task_to_messages(task, test_index)
        prompt = self._prompt_ids(turns)
        if prompt is None:
            return []
        if self.config.use_dfs:
            return self._predict_dfs(prompt)
        grids: list[Grid] = []
        for text in self._sample_texts(prompt, samples):
            try:
                grids.append(text_to_grid(text))
            except TaskFormatError:
                continue
        return grids

    def _sample_texts(self, prompt: torch.Tensor, samples: int) -> list[str]:
        """Decoded completions for one prompt: greedy first, then samples."""

        texts: list[str] = []
        attention_mask = torch.ones_like(prompt)
        with torch.no_grad():
            for sample in range(samples):
                generated = self.model.generate(
                    input_ids=prompt,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=sample > 0,
                    temperature=self.config.temperature,
                    pad_token_id=self._pad_id(),
                )
                texts.append(
                    self.tokenizer.decode(
                        generated[0][prompt.shape[1] :], skip_special_tokens=True
                    )
                )
        return texts

    def _predict_dfs(self, prompt: torch.Tensor) -> list[Grid]:
        import math
        import time

        from arcttt.decode import build_grid_vocab, constrained_dfs

        if self._grid_vocab is None:
            self._grid_vocab = build_grid_vocab(self.tokenizer)
        budget = self.config.dfs_time_budget_seconds
        results = constrained_dfs(
            self.model,
            prompt,
            self._grid_vocab,
            self.tokenizer,
            max_score=-math.log(self.config.dfs_probability_cutoff),
            max_new_tokens=self.config.max_new_tokens,
            max_candidates=self.config.dfs_max_candidates,
            deadline=time.monotonic() + budget if budget is not None else None,
        )
        seen: set[Grid] = set()
        ordered: list[Grid] = []
        for grid, _score in results:
            if grid not in seen:
                seen.add(grid)
                ordered.append(grid)
        if self.config.dfs_include_greedy:
            attention_mask = torch.ones_like(prompt)
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=prompt,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._pad_id(),
                )
            text = self.tokenizer.decode(
                generated[0][prompt.shape[1] :], skip_special_tokens=True
            )
            try:
                greedy = text_to_grid(text)
            except TaskFormatError:
                greedy = None
            if greedy is not None and greedy not in seen:
                ordered.append(greedy)
        return ordered

    def predict_frames(
        self, tasks: Sequence[Task], test_index: int, samples: int
    ) -> list[list[Grid]]:
        """Predict for several augmentation frames of one task at once.

        With DFS decoding the frames' searches run in lockstep batched
        forwards (``constrained_dfs_multi``) instead of one sequential
        search per frame; the sampling path falls back to per-frame
        ``predict``. Results per frame match the per-frame path.
        """

        if not self.config.use_dfs:
            return [self.predict(task, test_index, samples) for task in tasks]
        import math
        import time

        from arcttt.decode import build_grid_vocab, constrained_dfs_multi

        prompts = [
            self._prompt_ids(task_to_messages(task, test_index)) for task in tasks
        ]
        live = [(i, p) for i, p in enumerate(prompts) if p is not None]
        per_frame: list[list[Grid]] = [[] for _ in tasks]
        if not live:
            return per_frame
        if self._grid_vocab is None:
            self._grid_vocab = build_grid_vocab(self.tokenizer)
        budget = self.config.dfs_time_budget_seconds
        frame_stats: list[tuple[str, int]] = []
        results = constrained_dfs_multi(
            self.model,
            [prompt for _, prompt in live],
            self._grid_vocab,
            self.tokenizer,
            max_score=-math.log(self.config.dfs_probability_cutoff),
            max_new_tokens=self.config.max_new_tokens,
            max_candidates=self.config.dfs_max_candidates,
            deadline=time.monotonic() + budget if budget is not None else None,
            stats_out=frame_stats,
        )
        # Running tally of WHY searches stop, across the whole run. A run
        # dominated by "deadline" is time-limited (buy search time); one
        # dominated by "exhausted" is bound-limited (widen the NLL bound);
        # "candidate_cap" means raise max_candidates. Without this the only
        # signal is the next day's leaderboard score - one bit per day.
        cls = type(self)  # class-level tally: `self.x += 1` would shadow it
        for reason, found in frame_stats:
            cls.dfs_stop_reasons[reason] = cls.dfs_stop_reasons.get(reason, 0) + 1
            cls.dfs_candidates_found += found
            cls.dfs_frames_searched += 1
        greedy: list[Grid | None] = [None] * len(live)
        if self.config.dfs_include_greedy:
            greedy = self._greedy_grids([prompt for _, prompt in live])
        for (index, _), frame_results, greedy_grid in zip(live, results, greedy):
            seen: set[Grid] = set()
            ordered: list[Grid] = []
            for grid, _score in frame_results:
                if grid not in seen:
                    seen.add(grid)
                    ordered.append(grid)
            if greedy_grid is not None and greedy_grid not in seen:
                ordered.append(greedy_grid)
            per_frame[index] = ordered
        return per_frame

    def _greedy_grids(self, prompts: Sequence[torch.Tensor]) -> list[Grid | None]:
        """Greedy completion per prompt via one left-padded batch generate."""

        pad_id = self._pad_id()
        width = max(prompt.shape[1] for prompt in prompts)
        ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(prompts), width), dtype=torch.long)
        for row, prompt in enumerate(prompts):
            length = prompt.shape[1]
            ids[row, width - length :] = prompt[0].cpu()
            mask[row, width - length :] = 1
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=ids.to(self.device),
                attention_mask=mask.to(self.device),
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        grids: list[Grid | None] = []
        for row in range(len(prompts)):
            text = self.tokenizer.decode(generated[row][width:], skip_special_tokens=True)
            try:
                grids.append(text_to_grid(text))
            except TaskFormatError:
                grids.append(None)
        return grids

    def log_probability(self, task: Task, test_index: int, output: Grid) -> float:
        return self.log_probabilities(task, test_index, [output])[0]

    def log_probabilities_pairs(
        self,
        pairs: Sequence[tuple[Task, int, Grid]],
        chunk_rows: int = 12,
    ) -> list[float]:
        """Score heterogeneous (task, test_index, output) pairs, chunked.

        One padded batch forward per chunk instead of one call per
        augmentation frame; chunk_rows bounds activation memory — each chunk
        still materializes FULL-vocab logits ([chunk_rows, width, ~152k]
        float32, the dominant tensor; see the 152k-vocab OOM note at the top
        of this file), so keep chunk_rows small. The 16-token grid vocabulary
        only constrains DFS search in decode.py; it never shrinks scoring
        logits."""

        encoded: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for task, test_index, output in pairs:
            turns = task_to_messages(task, test_index) + (
                ChatTurn("assistant", grid_to_text(output)),
            )
            encoded.append(self._encode(turns, supervise_final=True))
        scores = [float("-inf")] * len(pairs)
        live = [(i, e) for i, e in enumerate(encoded) if e is not None]
        pad_id = self._pad_id()
        for start in range(0, len(live), chunk_rows):
            chunk = live[start : start + chunk_rows]
            width = max(e[0].shape[1] for _, e in chunk)
            ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            labels = torch.full((len(chunk), width), -100, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, (_, (input_ids, label_row)) in enumerate(chunk):
                length = input_ids.shape[1]
                ids[row, width - length :] = input_ids[0].cpu()
                labels[row, width - length :] = label_row[0].cpu()
                mask[row, width - length :] = 1
            with torch.no_grad():
                # use_cache=False: scoring needs logits only; the config
                # default (True) materializes a multi-GB throwaway KV cache
                # per chunk — the dominant OOM at the rescore stage on T4.
                logits = self.model(
                    input_ids=ids.to(self.device),
                    attention_mask=mask.to(self.device),
                    use_cache=False,
                ).logits.float()
            log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
            targets = labels[:, 1:].to(self.device)
            supervised = targets != -100
            gathered = log_probs.gather(
                -1, targets.clamp(min=0).unsqueeze(-1)
            ).squeeze(-1)
            for row, (index, _) in enumerate(chunk):
                positions = supervised[row]
                count = int(positions.sum().item())
                if count:
                    scores[index] = (
                        float(gathered[row][positions].sum().item()) / count
                    )
        return scores

    def log_probabilities(
        self, task: Task, test_index: int, outputs: Sequence[Grid]
    ) -> list[float]:
        """Score many candidate outputs in one left-padded batch forward."""

        sequences = [
            task_to_messages(task, test_index)
            + (ChatTurn("assistant", grid_to_text(output)),)
            for output in outputs
        ]
        return self.score_turn_sequences(sequences)

    def score_turn_sequences(self, sequences: Sequence[Sequence[ChatTurn]]) -> list[float]:
        """Mean supervised-token log-probability per chat sequence, one batch.

        Task-agnostic core of ``log_probabilities``: each sequence must end
        with the assistant turn being scored (grid or text alike)."""

        encoded = [self._encode(turns, supervise_final=True) for turns in sequences]
        live = [(i, e) for i, e in enumerate(encoded) if e is not None]
        scores = [float("-inf")] * len(sequences)
        if not live:
            return scores
        pad_id = self._pad_id()
        width = max(e[0].shape[1] for _, e in live)
        ids = torch.full((len(live), width), pad_id, dtype=torch.long)
        labels = torch.full((len(live), width), -100, dtype=torch.long)
        mask = torch.zeros((len(live), width), dtype=torch.long)
        for row, (_, (input_ids, label_row)) in enumerate(live):
            length = input_ids.shape[1]
            ids[row, width - length :] = input_ids[0].cpu()
            labels[row, width - length :] = label_row[0].cpu()
            mask[row, width - length :] = 1
        with torch.no_grad():
            # use_cache=False for the same reason as the chunked scorer above.
            logits = self.model(
                input_ids=ids.to(self.device),
                attention_mask=mask.to(self.device),
                use_cache=False,
            ).logits.float()
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = labels[:, 1:].to(self.device)
        supervised = targets != -100
        safe_targets = targets.clamp(min=0)
        gathered = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        for row, (index, _) in enumerate(live):
            positions = supervised[row]
            count = int(positions.sum().item())
            if count == 0:
                continue
            scores[index] = float(gathered[row][positions].sum().item()) / count
        return scores

    # -- encoding -----------------------------------------------------------

    def _prompt_ids(self, turns: Sequence[ChatTurn]) -> torch.Tensor | None:
        if self.config.raw_qwen_format:
            ids = self.tokenizer(
                turns_to_raw_qwen(turns, add_generation_prompt=True),
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
        else:
            ids = _template_ids(
                self.tokenizer.apply_chat_template(
                    turns_to_chat(turns),
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            )
        if ids.shape[1] > self.config.max_sequence_tokens:
            return None
        return cast(torch.Tensor, ids.to(self.device))

    def _encode(
        self, turns: Sequence[ChatTurn], supervise_final: bool
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Token ids plus labels masking everything except the final assistant turn."""

        if turns[-1].role != "assistant":
            raise TaskFormatError("supervised encoding needs a final assistant turn")
        if self.config.raw_qwen_format:
            full = self.tokenizer(
                turns_to_raw_qwen(turns, add_generation_prompt=False),
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            prefix = self.tokenizer(
                turns_to_raw_qwen(turns[:-1], add_generation_prompt=True),
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
        else:
            full = _template_ids(
                self.tokenizer.apply_chat_template(
                    turns_to_chat(turns), add_generation_prompt=False, return_tensors="pt"
                )
            )
            prefix = _template_ids(
                self.tokenizer.apply_chat_template(
                    turns_to_chat(turns[:-1]),
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            )
        if full.shape[1] > self.config.max_sequence_tokens:
            return None
        labels = full.clone()
        boundary = min(prefix.shape[1], full.shape[1])
        labels[0, :boundary] = -100
        if not supervise_final:
            labels[:, :] = -100
        return full.to(self.device), labels.to(self.device)
