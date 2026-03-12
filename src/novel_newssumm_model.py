import torch
import torch.nn as nn
from transformers import LongT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class EntityGraphModule(nn.Module):
    """
    NER-inspired learned entity attention head.

    Applies a token-wise binary classifier over encoder hidden states to
    produce entity salience weights, mimicking the role of named entity
    recognition without requiring an external NLP library.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.entity_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [B, T, H]
        Returns:
            entity_scores: [B, T, 1] — per-token entity salience weights
        """
        entity_logits = self.entity_head(hidden_states)  # [B, T, 1]
        entity_scores = torch.sigmoid(entity_logits)      # token-wise binary
        return entity_scores


class DiscourseHierarchyModule(nn.Module):
    """
    Positional discourse scorer.

    Models the positional importance of discourse segments via a two-layer
    MLP applied to sentence-level representations.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.discourse_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, sentence_reps):
        """
        Args:
            sentence_reps: [B, S, H] — one vector per discourse segment
        Returns:
            discourse_scores: [B, S, 1] — importance score per segment
        """
        return self.discourse_scorer(sentence_reps)  # [B, S, 1]


class MultiSourceFusion(nn.Module):
    """
    Fuses salience-weighted, entity-weighted, and discourse-weighted
    representations by concatenation followed by a learned linear projection
    back to hidden_size.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.fusion = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, salience_rep, entity_rep, discourse_rep):
        """
        Args:
            salience_rep:  [B, T, H]
            entity_rep:    [B, T, H]
            discourse_rep: [B, T, H]
        Returns:
            fused: [B, T, H]
        """
        concatenated = torch.cat(
            [salience_rep, entity_rep, discourse_rep], dim=-1
        )  # [B, T, 3H]
        return self.fusion(concatenated)  # [B, T, H]


class SalienceAwareLongT5(nn.Module):
    """
    Salience-Aware Hierarchical LongT5 for multi-document news summarization.

    Architecture:
        1. Long-Context Encoder (LongT5)
        2. Three parallel modules applied to encoder hidden states:
           - Salience Scoring  (token-level softmax attention)
           - Entity Graph      (NER-inspired learned sigmoid head)
           - Discourse Hierarchy (positional MLP scorer)
        3. Multi-Source Fusion  (concat + linear projection)
        4. Decoder with Constrained Attention
           (encoder hidden states re-weighted by fused scores before decoding)
    """

    def __init__(self, model_name="google/long-t5-tglobal-base"):
        super().__init__()

        # Base LongT5 model
        self.base_model = LongT5ForConditionalGeneration.from_pretrained(
            model_name
        )

        hidden_size = self.base_model.config.d_model

        # Module 1 — Salience Scoring Head
        self.salience_head = nn.Linear(hidden_size, 1)

        # Module 2 — Entity Graph (NER-inspired learned attention)
        self.entity_graph = EntityGraphModule(hidden_size)

        # Module 3 — Discourse Hierarchy scorer
        self.discourse_hierarchy = DiscourseHierarchyModule(hidden_size)

        # Multi-Source Fusion layer
        self.fusion = MultiSourceFusion(hidden_size)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):
        """
        Forward pass with salience-aware multi-source modeling.

        Returns a dict with keys:
            loss            — summarization loss (if labels provided)
            logits          — decoder logits
            salience_scores — [B, T, 1] token-level salience weights
            entity_scores   — [B, T, 1] token-level entity salience weights
            discourse_scores— [B, S, 1] segment-level discourse scores
        """

        # ------------------------------------------------------------------
        # Step 1 — Long-Context Encoder
        # ------------------------------------------------------------------
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden_states = encoder_outputs.last_hidden_state  # [B, T, H]

        # ------------------------------------------------------------------
        # Step 2 — Salience Scoring (softmax over token dimension)
        # ------------------------------------------------------------------
        salience_logits = self.salience_head(hidden_states)          # [B, T, 1]
        salience_scores = torch.softmax(salience_logits, dim=1)      # [B, T, 1]
        salience_rep = hidden_states * salience_scores               # [B, T, H]

        # ------------------------------------------------------------------
        # Step 3 — Entity Graph (token-level sigmoid entity head)
        # ------------------------------------------------------------------
        entity_scores = self.entity_graph(hidden_states)             # [B, T, 1]
        entity_rep = hidden_states * entity_scores                   # [B, T, H]

        # ------------------------------------------------------------------
        # Step 4 — Discourse Hierarchy (segment-level MLP scorer)
        # Treat [SEP]-delimited segments; fall back to token positions as
        # a proxy for discourse segments using the full hidden state sequence.
        # ------------------------------------------------------------------
        discourse_scores = self.discourse_hierarchy(hidden_states)   # [B, T, 1]
        discourse_rep = hidden_states * discourse_scores             # [B, T, H]

        # ------------------------------------------------------------------
        # Step 5 — Multi-Source Fusion
        # ------------------------------------------------------------------
        fused_hidden = self.fusion(
            salience_rep, entity_rep, discourse_rep
        )  # [B, T, H]

        # ------------------------------------------------------------------
        # Step 6 — Decoder with Constrained Cross-Attention
        # Re-weight encoder hidden states using fused multi-source scores
        # before passing to the decoder.
        # ------------------------------------------------------------------
        fused_scores = (salience_scores + entity_scores + discourse_scores) / 3.0
        fused_weights = torch.softmax(fused_scores, dim=1)           # [B, T, 1]
        constrained_hidden_states = fused_hidden * fused_weights     # [B, T, H]

        # Wrap constrained hidden states into a BaseModelOutput so the
        # LongT5 decoder can consume them transparently.
        constrained_encoder_outputs = BaseModelOutput(
            last_hidden_state=constrained_hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )

        outputs = self.base_model(
            encoder_outputs=constrained_encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "salience_scores": salience_scores,
            "entity_scores": entity_scores,
            "discourse_scores": discourse_scores
        }
