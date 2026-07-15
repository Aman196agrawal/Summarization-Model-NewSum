import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def pool_sentences(hidden_states, input_ids, eos_token_id):
    """Mean-pool encoder token hidden states into per-sentence embeddings.

    Sentence boundaries are the positions of `eos_token_id` in `input_ids`
    (see src.sentence_utils.build_cluster_input, which joins sentences with
    the real eos token so exactly one boundary marks the end of each
    sentence). Ragged per-example sentence counts are zero-padded.

    Returns (sentence_embeds [B, S, H], sentence_mask [B, S] bool).
    """
    batch_size, _, hidden_size = hidden_states.shape

    per_example_embeds = []
    max_sentences = 0
    for b in range(batch_size):
        boundaries = (input_ids[b] == eos_token_id).nonzero(as_tuple=True)[0].tolist()
        sentence_embeds = []
        start = 0
        for end in boundaries:
            span = hidden_states[b, start:end + 1]
            if span.shape[0] > 0:
                sentence_embeds.append(span.mean(dim=0))
            start = end + 1
        if not sentence_embeds:
            sentence_embeds = [hidden_states[b].mean(dim=0)]
        per_example_embeds.append(torch.stack(sentence_embeds))
        max_sentences = max(max_sentences, len(sentence_embeds))

    sentence_embeds = hidden_states.new_zeros(batch_size, max_sentences, hidden_size)
    sentence_mask = torch.zeros(batch_size, max_sentences, dtype=torch.bool, device=hidden_states.device)
    for b, embeds in enumerate(per_example_embeds):
        n = embeds.shape[0]
        sentence_embeds[b, :n] = embeds
        sentence_mask[b, :n] = True

    return sentence_embeds, sentence_mask


class SalienceAwareLongT5(nn.Module):
    """Salience-Aware LongT5, matching the report's Section IV formulation.

    a_ij = W_s h_ij + b_s                         (per-sentence salience score)
    alpha_ij = softmax over ALL sentences in the cluster
    Hdoc = sum_ij alpha_ij * h_ij                 (single global document vector)
    H~_ij = H_ij + Hdoc                           (residual "hierarchical" fusion)
    decoder cross-attends to H~ instead of raw H  ("constrained attention")

    Loss: L = L_sum + lambda * L_sal, where L_sal is BCE(alpha, oracle_labels).
    """

    LAMBDA = 1.0

    def __init__(self, model_name="google/long-t5-tglobal-base"):
        super().__init__()
        base_model = LongT5ForConditionalGeneration.from_pretrained(model_name)
        self._init_from_base_model(base_model)

    @classmethod
    def from_base_model(cls, base_model):
        """Wrap an already-constructed LongT5ForConditionalGeneration.

        Used by tests to avoid downloading the ~990MB pretrained backbone --
        pass a tiny randomly-initialized LongT5ForConditionalGeneration(config)
        instead. Production code should use the normal constructor.
        """
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance._init_from_base_model(base_model)
        return instance

    def _init_from_base_model(self, base_model):
        self.base_model = base_model
        hidden_size = self.base_model.config.d_model
        self.salience_head = nn.Linear(hidden_size, 1)

    def _encode_with_salience(self, input_ids, attention_mask):
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state

        eos_token_id = self.base_model.config.eos_token_id
        sentence_embeds, sentence_mask = pool_sentences(hidden_states, input_ids, eos_token_id)

        salience_logits = self.salience_head(sentence_embeds).squeeze(-1)
        salience_logits = salience_logits.masked_fill(~sentence_mask, float("-inf"))
        alpha = torch.softmax(salience_logits, dim=1)

        h_doc = (alpha.unsqueeze(-1) * sentence_embeds).sum(dim=1)
        constrained_hidden = hidden_states + h_doc.unsqueeze(1)

        return constrained_hidden, alpha, sentence_mask

    def _reconcile_salience_labels(self, salience_labels, target_len):
        current_len = salience_labels.shape[1]
        if current_len < target_len:
            return F.pad(salience_labels, (0, target_len - current_len), value=0.0)
        if current_len > target_len:
            return salience_labels[:, :target_len]
        return salience_labels

    def forward(self, input_ids, attention_mask, labels=None, salience_labels=None):
        constrained_hidden, alpha, sentence_mask = self._encode_with_salience(
            input_ids, attention_mask
        )

        constrained_encoder_outputs = BaseModelOutput(last_hidden_state=constrained_hidden)
        outputs = self.base_model(
            encoder_outputs=constrained_encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        loss = outputs.loss
        salience_loss = None
        if salience_labels is not None:
            reconciled = self._reconcile_salience_labels(salience_labels, alpha.shape[1])
            probs = alpha.clamp(min=1e-7, max=1 - 1e-7)
            # alpha sums to 1 across sentences (softmax); BCE treats each independently -- this is
            # the report's literal formulation (softmax salience + BCE loss), not a claim that
            # multiple sentences can be simultaneously highly salient.
            bce = F.binary_cross_entropy(probs, reconciled, reduction="none")
            bce = bce * sentence_mask.float()
            denom = sentence_mask.float().sum().clamp(min=1.0)
            salience_loss = bce.sum() / denom
            if loss is not None:
                loss = loss + self.LAMBDA * salience_loss

        return {
            "loss": loss,
            "logits": outputs.logits,
            "salience_scores": alpha,
            "sentence_mask": sentence_mask,
            "salience_loss": salience_loss,
        }

    def generate(self, input_ids, attention_mask, **generate_kwargs):
        constrained_hidden, _, _ = self._encode_with_salience(input_ids, attention_mask)
        constrained_encoder_outputs = BaseModelOutput(last_hidden_state=constrained_hidden)
        return self.base_model.generate(
            encoder_outputs=constrained_encoder_outputs,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
