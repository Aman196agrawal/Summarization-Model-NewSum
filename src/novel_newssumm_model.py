import torch
import torch.nn as nn
from transformers import LongT5ForConditionalGeneration


class SalienceAwareLongT5(nn.Module):
    """
    Salience-Aware Hierarchical LongT5 for multi-document news summarization.
    """

    def __init__(self, model_name="google/long-t5-tglobal-base"):
        super().__init__()

        # Base LongT5 model
        self.base_model = LongT5ForConditionalGeneration.from_pretrained(
            model_name
        )

        hidden_size = self.base_model.config.d_model

        # Salience prediction head (NEW)
        self.salience_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):
        """
        Forward pass with salience-aware modeling.
        """

        # Encoder forward
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden_states = encoder_outputs.last_hidden_state  # [B, T, H]

        # Sentence-level representation (mean pooling)
        sentence_rep = hidden_states.mean(dim=1)  # [B, H]

        # Salience scores
        salience_logits = self.salience_head(sentence_rep)
        salience_scores = torch.sigmoid(salience_logits)

        # Decoder forward (standard summarization loss)
        outputs = self.base_model(
            encoder_outputs=encoder_outputs,
            labels=labels,
            return_dict=True
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "salience_scores": salience_scores
        }
