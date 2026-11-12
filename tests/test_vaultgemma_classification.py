"""Small, random CPU models: no pretrained weights or tokenizer downloads."""

import copy

import pytest
import torch
from transformers import (
    AutoModelForSequenceClassification,
    VaultGemmaConfig,
    VaultGemmaModel,
    VaultGemmaForCausalLM,
)

from dpdl.models.vaultgemma import VaultGemmaForSequenceClassification


def config():
    return VaultGemmaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        layer_types=["full_attention"],
        num_labels=2,
        pad_token_id=0,
        attention_dropout=0.0,
        attn_implementation="eager",
    )


@pytest.fixture
def registered_models():
    from dpdl.models.hugging_face_models import register_custom_huggingface_models

    register_custom_huggingface_models()


@pytest.fixture
def model(registered_models):
    torch.manual_seed(42)
    return VaultGemmaForSequenceClassification(config())


def test_logits_ignore_padding_on_either_side(model):
    model.eval()
    ids = torch.tensor([[2, 5, 7, 0, 0], [0, 0, 2, 5, 7]])
    out = model(input_ids=ids, attention_mask=ids.ne(0))
    expected = model(input_ids=torch.tensor([[2, 5, 7]])).logits
    assert out.logits.shape == (2, 2)
    torch.testing.assert_close(out.logits, expected.expand(2, -1), atol=1e-6, rtol=1e-5)
    assert getattr(out, "past_key_values", None) is None


def test_mask_free_padding(model):
    ids = torch.tensor([[2, 5, 7, 0], [2, 8, 0, 0]])
    for i, count in enumerate([3, 2]):
        torch.testing.assert_close(
            model(input_ids=ids[i : i + 1, :count]).logits,
            model(input_ids=ids).logits[i : i + 1],
        )


def test_all_padding_is_rejected(model):
    with pytest.raises(ValueError, match="non-padding"):
        model(input_ids=torch.zeros(1, 3, dtype=torch.long))


@pytest.mark.parametrize(
    "argument", ["use_cache", "past_key_values", "labels", "return_dict"]
)
def test_classification_rejects_unsupported_arguments(model, argument):
    with pytest.raises(TypeError, match=argument):
        model(input_ids=torch.tensor([[2, 5, 7]]), **{argument: None})


def test_full_finetuning_includes_embeddings_and_no_lm_head(model):
    ids = torch.tensor([[2, 5, 7], [2, 5, 8]])
    torch.nn.functional.cross_entropy(
        model(input_ids=ids).logits, torch.tensor([0, 1])
    ).backward()
    assert not hasattr(model, "lm_head")
    assert all(p.requires_grad and p.grad is not None for p in model.parameters())
    assert model.get_input_embeddings().weight.grad.abs().sum() > 0


@pytest.mark.parametrize("source_cls", [VaultGemmaModel, VaultGemmaForCausalLM])
def test_pretrained_backbone_loads_without_changing_embeddings(
    tmp_path, source_cls, registered_models
):
    source = source_cls(config())
    source.save_pretrained(tmp_path)
    backbone = source.model if isinstance(source, VaultGemmaForCausalLM) else source
    loaded = AutoModelForSequenceClassification.from_pretrained(tmp_path, num_labels=2)
    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value)
    ids = torch.tensor([[2, 5, 7]])
    torch.testing.assert_close(
        loaded.model(input_ids=ids, use_cache=False).last_hidden_state,
        backbone(input_ids=ids, use_cache=False).last_hidden_state,
    )


def test_classification_save_reload(model, tmp_path):
    model.save_pretrained(tmp_path)
    loaded = AutoModelForSequenceClassification.from_pretrained(tmp_path)
    ids = torch.tensor([[2, 5, 7]])
    torch.testing.assert_close(
        model(input_ids=ids).logits, loaded(input_ids=ids).logits
    )


def test_opacus_grad_samples_match_individual_examples(model):
    from opacus.grad_sample import GradSampleModule

    ids = torch.tensor([[2, 5, 7, 0], [2, 5, 8, 9]])
    labels = torch.tensor([0, 1])
    reference = copy.deepcopy(model)
    expected = {name: [] for name, _ in reference.named_parameters()}
    for i in range(2):
        reference.zero_grad()
        torch.nn.functional.cross_entropy(
            reference(input_ids=ids[i : i + 1]).logits, labels[i : i + 1]
        ).backward()
        for name, param in reference.named_parameters():
            expected[name].append(param.grad.clone())
    wrapped = GradSampleModule(model, batch_first=True, loss_reduction="mean")
    torch.nn.functional.cross_entropy(wrapped(input_ids=ids).logits, labels).backward()
    for name, param in model.named_parameters():
        torch.testing.assert_close(
            param.grad_sample, torch.stack(expected[name]), atol=2e-6, rtol=2e-4
        )


def test_dpdl_loader_and_reload_use_classification_adapter(
    model, tmp_path, monkeypatch
):
    from dpdl.models.hugging_face_models import HuggingfaceLanguageModel

    model.save_pretrained(tmp_path)

    class Tokenizer:
        pad_token = "<pad>"

    monkeypatch.setattr(
        "dpdl.models.hugging_face_models.AutoTokenizer.from_pretrained",
        lambda *a, **k: Tokenizer(),
    )
    wrapper = HuggingfaceLanguageModel(
        str(tmp_path), quantization=False, num_labels=2, task="SequenceClassification"
    )
    assert isinstance(wrapper.model, VaultGemmaForSequenceClassification)
    assert all(p.device.type == "cpu" and p.requires_grad for p in wrapper.parameters())
    batch = {"input_ids": torch.tensor([[2, 5, 7]])}
    torch.testing.assert_close(wrapper(batch), model(**batch).logits)
    wrapper.load_model(str(tmp_path))
    torch.testing.assert_close(wrapper(batch), model(**batch).logits)


def test_existing_bert_loader_remains_classification_on_cpu(tmp_path, monkeypatch):
    from transformers import BertConfig, BertForSequenceClassification
    from dpdl.models.hugging_face_models import HuggingfaceLanguageModel

    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_labels=2,
    )
    BertForSequenceClassification(config).save_pretrained(tmp_path)

    class Tokenizer:
        pad_token = "<pad>"

    monkeypatch.setattr(
        "dpdl.models.hugging_face_models.AutoTokenizer.from_pretrained",
        lambda *a, **k: Tokenizer(),
    )
    wrapper = HuggingfaceLanguageModel(
        str(tmp_path), quantization=False, num_labels=2, task="SequenceClassification"
    )
    assert isinstance(wrapper.model, BertForSequenceClassification)
    assert all(p.device.type == "cpu" for p in wrapper.parameters())
    assert all(not p.requires_grad for p in wrapper.model.bert.embeddings.parameters())
    assert wrapper({"input_ids": torch.tensor([[2, 5, 7]])}).shape == (1, 2)


def test_dp_optimizer_updates_embeddings_and_head(model):
    from opacus.grad_sample import GradSampleModule
    from opacus.optimizers import DPOptimizer

    wrapped = GradSampleModule(model)
    optimizer = DPOptimizer(
        torch.optim.SGD(wrapped.parameters(), lr=0.1),
        noise_multiplier=0.1,
        max_grad_norm=1.0,
        expected_batch_size=2,
        generator=torch.Generator().manual_seed(17),
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    ids = torch.tensor([[2, 5, 7, 0], [2, 5, 8, 9]])
    torch.nn.functional.cross_entropy(
        wrapped(input_ids=ids).logits, torch.tensor([0, 1])
    ).backward()
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    for name in ("model.embed_tokens.weight", "score.weight"):
        assert not torch.equal(before[name], dict(model.named_parameters())[name])


def test_imports_do_not_register_and_explicit_initialization_is_repeatable():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import dpdl.models
assert 'dpdl.models.vaultgemma' not in sys.modules
from transformers import AutoModelForSequenceClassification, VaultGemmaConfig
from opacus.grad_sample import GradSampleModule
models_before = dict(AutoModelForSequenceClassification._model_mapping._extra_content)
samplers_before = dict(GradSampleModule.GRAD_SAMPLERS)
from dpdl.models import vaultgemma
assert AutoModelForSequenceClassification._model_mapping._extra_content == models_before
assert GradSampleModule.GRAD_SAMPLERS == samplers_before
from dpdl.models.hugging_face_models import register_custom_huggingface_models
config = VaultGemmaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
    head_dim=8, layer_types=['full_attention'], num_labels=2)
for _ in range(2):
    register_custom_huggingface_models()
    model = AutoModelForSequenceClassification.from_config(config)
    assert isinstance(model, vaultgemma.VaultGemmaForSequenceClassification)
    assert all(p.requires_grad for p in model.parameters())
    assert GradSampleModule.GRAD_SAMPLERS == samplers_before
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_wrapper_initializes_registration_in_fresh_process(model, tmp_path):
    import subprocess
    import sys

    model.save_pretrained(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from unittest.mock import patch
from types import SimpleNamespace
import torch
from dpdl.models.hugging_face_models import HuggingfaceLanguageModel
assert 'dpdl.models.vaultgemma' not in sys.modules
with patch('dpdl.models.hugging_face_models.AutoTokenizer.from_pretrained',
           return_value=SimpleNamespace(pad_token='<pad>')):
    for _ in range(2):
        wrapper = HuggingfaceLanguageModel(sys.argv[1], quantization=False,
                                          num_labels=2, task='SequenceClassification')
        assert all(p.requires_grad for p in wrapper.parameters())
        ids = {'input_ids': torch.tensor([[2, 5, 7]])}
        expected = wrapper(ids)
        wrapper.load_model(sys.argv[1])
        torch.testing.assert_close(wrapper(ids), expected)
""",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_huggingface_feature_caching_is_explicitly_unsupported():
    from dpdl.models.hugging_face_models import HuggingfaceLanguageModel

    wrapper = HuggingfaceLanguageModel.__new__(HuggingfaceLanguageModel)
    torch.nn.Module.__init__(wrapper)
    with pytest.raises(NotImplementedError, match="[Ff]eature caching"):
        wrapper.forward_features({"input_ids": torch.tensor([[2, 5, 7]])})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_scaled_embedding_matches_hf_dtype_rounding(dtype):
    from transformers.models.vaultgemma import modeling_vaultgemma
    from dpdl.models.vaultgemma import VaultGemmaScaledEmbedding

    hf_class = getattr(modeling_vaultgemma, "VaultGemmaTextScaledWordEmbedding", None)
    if hf_class is None:
        pytest.skip("This Transformers version scales outside the embedding.")
    # Unlike sqrt(16), sqrt(6) is not exactly representable in reduced precision.
    original = hf_class(17, 6, padding_idx=0, embed_scale=6**0.5).to(dtype)
    with torch.no_grad():
        original.weight.copy_(torch.linspace(-1, 1, 102).reshape(17, 6).to(dtype))
    replacement = VaultGemmaScaledEmbedding(original)
    ids = torch.tensor([[2, 5, 2, 0], [4, 7, 8, 1]])
    expected = original(ids)
    actual = replacement(ids)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == dtype and replacement.weight.requires_grad
    assert not list(replacement.buffers())
    upstream = torch.linspace(0.1, 0.9, expected.numel()).reshape_as(expected).to(dtype)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(
        replacement.weight.grad, original.weight.grad, atol=0, rtol=0
    )
