import torch
import torch.nn.functional as F
import pytest
from transformers import LongT5Config, LongT5ForConditionalGeneration

from src.novel_newssumm_model import SalienceAwareLongT5, pool_sentences


def _tiny_base_model():
    config = LongT5Config(
        vocab_size=32,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        local_radius=4,
        global_block_size=4,
        feed_forward_proj="relu",
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return LongT5ForConditionalGeneration(config)


@pytest.fixture
def tiny_model():
    return SalienceAwareLongT5.from_base_model(_tiny_base_model())


def test_pool_sentences_mean_pools_between_eos_boundaries():
    hidden_states = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    input_ids = torch.tensor([[10, 1, 20, 21, 1]])

    embeds, mask = pool_sentences(hidden_states, input_ids, eos_token_id=1)

    assert embeds.shape == (1, 2, 1)
    assert torch.allclose(embeds[0, 0], torch.tensor([1.5]))
    assert torch.allclose(embeds[0, 1], torch.tensor([4.0]))
    assert mask.tolist() == [[True, True]]


def test_forward_returns_finite_loss_and_normalized_salience_scores(tiny_model):
    input_ids = torch.tensor([[5, 6, 1, 7, 8, 9, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[2, 3, 1]])

    output = tiny_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert torch.isfinite(output["loss"])
    assert output["salience_scores"].shape == (1, 2)
    assert torch.allclose(output["salience_scores"].sum(dim=1), torch.tensor([1.0]), atol=1e-5)


def test_forward_adds_salience_loss_when_labels_provided(tiny_model):
    input_ids = torch.tensor([[5, 6, 1, 7, 8, 9, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[2, 3, 1]])
    salience_labels = torch.tensor([[1.0, 0.0]])

    without = tiny_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    with_sal = tiny_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        salience_labels=salience_labels,
    )

    assert with_sal["salience_loss"] is not None
    assert torch.isfinite(with_sal["salience_loss"])
    assert not torch.allclose(with_sal["loss"], without["loss"])


def test_forward_reconciles_salience_labels_length_mismatch(tiny_model):
    input_ids = torch.tensor([[5, 6, 1, 7, 8, 9, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[2, 3, 1]])

    too_long = torch.zeros(1, 50)
    output_long = tiny_model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
        salience_labels=too_long,
    )
    assert torch.isfinite(output_long["salience_loss"])

    too_short = torch.zeros(1, 1)
    output_short = tiny_model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
        salience_labels=too_short,
    )
    assert torch.isfinite(output_short["salience_loss"])


def test_generate_runs_through_the_salience_path_not_the_bare_backbone(tiny_model):
    calls = []
    original = tiny_model._encode_with_salience

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    tiny_model._encode_with_salience = spy

    input_ids = torch.tensor([[5, 6, 1, 7, 8, 9, 1]])
    attention_mask = torch.ones_like(input_ids)

    generated = tiny_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=5)

    assert len(calls) == 1
    assert generated.shape[0] == 1
