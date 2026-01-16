"""
Beam search decoder for MBA expressions with grammar constraints and diversity.
"""

from typing import List, Dict, Optional, Tuple, Set
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field

from src.constants import (
    BEAM_WIDTH, BEAM_DIVERSITY_GROUPS, BEAM_DIVERSITY_PENALTY,
    BEAM_TEMPERATURE, MAX_SEQ_LEN, SOS_IDX, EOS_IDX, PAD_IDX
)
from src.models.full_model import MBADeobfuscator
from src.data.tokenizer import MBATokenizer
from src.inference.grammar import GrammarConstraint


@dataclass
class BeamHypothesis:
    """
    Single hypothesis in beam search.
    """
    tokens: List[int] = field(default_factory=list)  # Token sequence
    score: float = 0.0                                # Log probability
    finished: bool = False                            # Hit EOS or max length
    length: int = 0                                   # Sequence length

    @property
    def normalized_score(self) -> float:
        """
        Length-normalized score for ranking.
        Uses Wu et al. (2016) length penalty: score / (length^0.6)
        """
        if self.length == 0:
            return self.score
        return self.score / (self.length ** 0.6)


class BeamSearchDecoder:
    """
    Beam search decoder with grammar constraints and diversity groups.
    """

    def __init__(
        self,
        model: MBADeobfuscator,
        tokenizer: MBATokenizer,
        beam_width: int = BEAM_WIDTH,
        num_groups: int = BEAM_DIVERSITY_GROUPS,
        diversity_penalty: float = BEAM_DIVERSITY_PENALTY,
        temperature: float = BEAM_TEMPERATURE,
        max_length: int = MAX_SEQ_LEN,
        use_grammar: bool = True
    ):
        """
        Initialize beam search decoder.

        Args:
            model: Trained MBADeobfuscator model
            tokenizer: MBATokenizer instance
            beam_width: Number of beams to keep
            num_groups: Number of diversity groups (1 = standard beam search)
            diversity_penalty: Penalty for selecting same token across groups
            temperature: Softmax temperature for sampling diversity
            max_length: Maximum generation length
            use_grammar: Apply grammar constraints
        """
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.num_groups = num_groups
        self.diversity_penalty = diversity_penalty
        self.temperature = temperature
        self.max_length = max_length
        self.use_grammar = use_grammar

        if use_grammar:
            self.grammar = GrammarConstraint(tokenizer)
        else:
            self.grammar = None

        self.device = next(model.parameters()).device

    def decode(
        self,
        memory: torch.Tensor,
        src_tokens: Optional[List[int]] = None
    ) -> List[BeamHypothesis]:
        """
        Beam search decoding.

        Args:
            memory: [1, 1, D_MODEL] encoder context from model.encode()
            src_tokens: Optional source token IDs for copy mechanism

        Returns:
            List of BeamHypothesis sorted by normalized score

        Algorithm:
            1. Initialize with <sos> token
            2. For each step:
                a. Get logits from model.decode()
                b. Apply grammar mask if enabled
                c. Apply copy mechanism if src_tokens provided
                d. Apply diversity penalty across groups
                e. Expand top-k hypotheses per group
                f. Prune finished beams
            3. Return top beams ranked by normalized score
        """
        batch_size = memory.size(0)
        assert batch_size == 1, "Beam search only supports batch_size=1"

        # Initialize beams with <sos>
        beams = [BeamHypothesis(tokens=[SOS_IDX], score=0.0, length=0)]
        finished_beams = []

        for step in range(self.max_length):
            if all(beam.finished for beam in beams):
                break

            # Prepare batch of active beams
            active_beams = [b for b in beams if not b.finished]
            if not active_beams:
                break

            # Stack target sequences [num_beams, seq_len]
            tgt_tokens = [torch.tensor(b.tokens, dtype=torch.long, device=self.device)
                         for b in active_beams]
            max_len = max(len(t) for t in tgt_tokens)
            tgt_batch = torch.full((len(tgt_tokens), max_len), PAD_IDX,
                                  dtype=torch.long, device=self.device)
            for i, tokens in enumerate(tgt_tokens):
                tgt_batch[i, :len(tokens)] = tokens

            # Expand memory for batch [num_beams, 1, D_MODEL]
            memory_expanded = memory.expand(len(active_beams), -1, -1)

            # Get logits from model
            with torch.no_grad():
                decode_output = self.model.decode(tgt_batch, memory_expanded)
                vocab_logits = decode_output['vocab_logits']  # [num_beams, seq_len, vocab_size]
                copy_attn = decode_output.get('copy_attn')  # [num_beams, seq_len, src_len]
                p_gen = decode_output.get('p_gen')  # [num_beams, seq_len, 1]

            # Get logits for next token (last position)
            next_logits = vocab_logits[:, -1, :]  # [num_beams, vocab_size]

            # Apply copy mechanism if available
            if src_tokens is not None and copy_attn is not None and p_gen is not None:
                copy_attn_last = copy_attn[:, -1, :]  # [num_beams, src_len]
                p_gen_last = p_gen[:, -1, :]  # [num_beams, 1]
                next_logits = self._apply_copy_mechanism(
                    next_logits, copy_attn_last, p_gen_last, src_tokens
                )

            # Apply temperature
            next_logits = next_logits / self.temperature

            # Apply grammar constraints
            if self.grammar is not None:
                for i, beam in enumerate(active_beams):
                    valid_tokens = self.grammar.get_valid_tokens(beam.tokens)
                    next_logits[i] = self.grammar.mask_logits(next_logits[i], valid_tokens)

            # Convert to log probabilities
            log_probs = F.log_softmax(next_logits, dim=-1)  # [num_beams, vocab_size]

            # Expand beams with diversity
            new_beams = []

            if self.num_groups > 1:
                # Diverse beam search
                beams_per_group = self.beam_width // self.num_groups
                selected_tokens_global = set()

                for group_idx in range(self.num_groups):
                    group_log_probs = log_probs.clone()

                    # Apply diversity penalty
                    if selected_tokens_global:
                        penalty = torch.zeros_like(group_log_probs[0])
                        for token_id in selected_tokens_global:
                            penalty[token_id] = self.diversity_penalty
                        group_log_probs -= penalty.unsqueeze(0)

                    # Select top-k for this group
                    group_beams = self._expand_beams(
                        active_beams, group_log_probs, beams_per_group
                    )
                    new_beams.extend(group_beams)

                    # Track selected tokens
                    for beam in group_beams:
                        if beam.tokens:
                            selected_tokens_global.add(beam.tokens[-1])
            else:
                # Standard beam search
                new_beams = self._expand_beams(active_beams, log_probs, self.beam_width)

            # Separate finished and active beams
            beams = []
            for beam in new_beams:
                if beam.tokens[-1] == EOS_IDX or beam.length >= self.max_length:
                    beam.finished = True
                    finished_beams.append(beam)
                else:
                    beams.append(beam)

            # Keep only top beams
            beams = sorted(beams, key=lambda b: b.normalized_score, reverse=True)[:self.beam_width]

        # Combine finished and remaining beams
        all_beams = finished_beams + beams

        # Sort by normalized score
        all_beams = sorted(all_beams, key=lambda b: b.normalized_score, reverse=True)

        return all_beams[:self.beam_width]

    def _expand_beams(
        self,
        beams: List[BeamHypothesis],
        log_probs: torch.Tensor,
        k: int
    ) -> List[BeamHypothesis]:
        """
        Expand beams with top-k tokens.

        Args:
            beams: Current hypotheses
            log_probs: [num_beams, vocab_size] log probabilities
            k: Number of expansions to keep

        Returns:
            List of expanded hypotheses
        """
        vocab_size = log_probs.size(-1)
        num_beams = len(beams)

        # Compute scores for all expansions [num_beams * vocab_size]
        beam_scores = torch.tensor([b.score for b in beams], device=log_probs.device)
        beam_scores = beam_scores.unsqueeze(1) + log_probs  # [num_beams, vocab_size]
        beam_scores_flat = beam_scores.view(-1)

        # Get top-k
        top_scores, top_indices = torch.topk(beam_scores_flat, min(k, len(beam_scores_flat)))

        # Decode indices
        new_beams = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            beam_idx = idx // vocab_size
            token_idx = idx % vocab_size

            # Create new hypothesis
            parent_beam = beams[beam_idx]
            new_beam = BeamHypothesis(
                tokens=parent_beam.tokens + [token_idx],
                score=score,
                length=parent_beam.length + 1,
                finished=False
            )
            new_beams.append(new_beam)

        return new_beams

    def _apply_copy_mechanism(
        self,
        vocab_logits: torch.Tensor,
        copy_attn: torch.Tensor,
        p_gen: torch.Tensor,
        src_tokens: List[int]
    ) -> torch.Tensor:
        """
        Combine vocabulary and copy distributions.

        Args:
            vocab_logits: [num_beams, vocab_size] vocabulary logits
            copy_attn: [num_beams, src_len] copy attention weights
            p_gen: [num_beams, 1] generation probability
            src_tokens: Source token IDs for copying

        Returns:
            [num_beams, vocab_size] combined log probabilities

        Formula:
            P(w) = p_gen * softmax(vocab_logits) + (1 - p_gen) * Σ(attn_i | src_i == w)
        """
        num_beams = vocab_logits.size(0)
        vocab_size = vocab_logits.size(1)

        # Vocabulary distribution
        vocab_dist = F.softmax(vocab_logits, dim=-1)  # [num_beams, vocab_size]

        # Copy distribution: scatter copy_attn to vocabulary space
        copy_dist = torch.zeros_like(vocab_dist)
        src_len = len(src_tokens)

        for i in range(num_beams):
            for j in range(min(src_len, copy_attn.size(1))):
                token_id = src_tokens[j]
                if token_id < vocab_size:
                    copy_dist[i, token_id] += copy_attn[i, j]

        # Combine distributions
        p_gen = torch.sigmoid(p_gen)  # Ensure [0, 1]
        combined_dist = p_gen * vocab_dist + (1 - p_gen) * copy_dist

        # Convert back to logits
        combined_logits = torch.log(combined_dist + 1e-20)

        return combined_logits
