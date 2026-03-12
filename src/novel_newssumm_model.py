import torch
import torch.nn as nn
from transformers import LongT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class EntityGraphModule(nn.Module):
    """
    Learned NER-inspired entity scoring head.
    Applies token-wise entity importance scoring via a linear layer.
    Mimics a graph attention network over named entity tokens without
    requiring an external NER tool.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.entity_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # hidden_states: [B, T, H]
        entity_logits = self.entity_head(hidden_states)        # [B, T, 1]
        entity_scores = torch.sigmoid(entity_logits)           # [B, T, 1]
        entity_repr = hidden_states * entity_scores            # [B, T, H]
        return entity_repr, entity_scores


class DiscourseHierarchyModule(nn.Module):
    """
    Positional discourse scorer using a 2-layer MLP.
    Models the discourse-level importance of each token position,
    capturing structural hierarchy in multi-document input.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.discourse_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states):
        # hidden_states: [B, T, H]
        discourse_logits = self.discourse_scorer(hidden_states)    # [B, T, 1]
        discourse_scores = torch.softmax(discourse_logits, dim=1)  # [B, T, 1]
        discourse_repr = hidden_states * discourse_scores           # [B, T, H]
        return discourse_repr, discourse_scores


class MultiSourceFusion(nn.Module):
    """
    Fuses three source representations (salience, entity, discourse)
    via learned linear projection from 3*hidden_size → hidden_size.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.fusion = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, salience_repr, entity_repr, discourse_repr):
        # Each repr: [B, T, H]
        concatenated = torch.cat([salience_repr, entity_repr, discourse_repr], dim=-1)  # [B, T, 3H]
        fused = self.fusion(concatenated)   # [B, T, H]
        return fused


class SalienceAwareLongT5(nn.Module):
    """
    Salience-Aware Hierarchical LongT5 for multi-document news summarization.

    Architecture:
      1. Long-Context Encoder (LongT5) encodes [SEP]-segmented multi-document input
      2. Three parallel modules: Salience Scoring, Entity Graph, Discourse Hierarchy
      3. Multi-Source Fusion: fuses the three weighted representations
      4. Constrained Attention: re-weights encoder hidden states before decoding
      5. Standard LongT5 decoder generates the summary

    Loss: L = L_summarization (cross-entropy)
    Lambda (λ) for auxiliary salience loss = 0.1 (future work when oracle labels available)
    """

    LAMBDA = 0.1  # Weight for auxiliary salience loss (reserved for future oracle supervision)

    def __init__(self, model_name="google/long-t5-tglobal-base"):
        super().__init__()

        # Base LongT5 model
        self.base_model = LongT5ForConditionalGeneration.from_pretrained(model_name)
        hidden_size = self.base_model.config.d_model

        # Module ①: Salience Scoring Head
        self.salience_head = nn.Linear(hidden_size, 1)

        # Module ②: Entity Graph Module (NER-inspired)
        self.entity_module = EntityGraphModule(hidden_size)

        # Module ③: Discourse Hierarchy Module
        self.discourse_module = DiscourseHierarchyModule(hidden_size)

        # Multi-Source Fusion
        self.fusion = MultiSourceFusion(hidden_size)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass with salience-aware hierarchical modeling and constrained attention.

        Args:
            input_ids:       [B, T] — tokenized [SEP]-joined multi-document input
            attention_mask:  [B, T]
            labels:          [B, L] — tokenized reference summary (for training)

        Returns:
            dict with loss, logits, salience_scores, entity_scores, discourse_scores
        """

        # ── Step 1: Encode ──────────────────────────────────────────────
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state  # [B, T, H]

        # ── Step 2: Salience Scoring (Module ①) ─────────────────────────
        salience_logits = self.salience_head(hidden_states)            # [B, T, 1]
        salience_scores = torch.softmax(salience_logits, dim=1)        # [B, T, 1] — softmax across tokens
        salience_repr   = hidden_states * salience_scores              # [B, T, H]

        # ── Step 3: Entity Graph (Module ②) ─────────────────────────────
        entity_repr, entity_scores = self.entity_module(hidden_states)

        # ── Step 4: Discourse Hierarchy (Module ③) ──────────────────────
        discourse_repr, discourse_scores = self.discourse_module(hidden_states)

        # ── Step 5: Multi-Source Fusion ──────────────────────────────────
        fused_hidden = self.fusion(salience_repr, entity_repr, discourse_repr)  # [B, T, H]

        # ── Step 6: Constrained Attention ───────────────────────────────
        # Average the three score distributions and re-weight encoder hidden states
        fused_scores   = (salience_scores + entity_scores + discourse_scores) / 3.0  # [B, T, 1]
        fused_weights  = torch.softmax(fused_scores, dim=1)                          # [B, T, 1]
        constrained_H  = fused_hidden * fused_weights                                # [B, T, H]

        # ── Step 7: Decode with constrained hidden states ───────────────
        constrained_encoder_outputs = BaseModelOutput(
            last_hidden_state=constrained_H,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )

        outputs = self.base_model(
            encoder_outputs=constrained_encoder_outputs,
            labels=labels,
            return_dict=True
        )

        return {
            "loss":             outputs.loss,
            "logits":           outputs.logits,
            "salience_scores":  salience_scores,
            "entity_scores":    entity_scores,
            "discourse_scores": discourse_scores,
        }
