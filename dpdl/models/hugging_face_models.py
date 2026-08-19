import logging
import json
import os
import typing as t
from collections.abc import Mapping

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as safe_load_file
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

log = logging.getLogger(__name__)

SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"


def download_safetensors(model_name):
    """
        Load model directly from HuggingFace
    """
    folder = snapshot_download(model_name, allow_patterns=["*.json", "*model*.safetensors"])

    # Load the index
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)
    assert os.path.isfile(safe_index_file)

    with open(safe_index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    state_dict: t.Dict[str, torch.Tensor] = {}

    for shard_file in shard_files:
        state_dict |= safe_load_file(os.path.join(folder, shard_file), 'cpu')


def download_generic_huggingface_model(
    model_name,
    quantization,
    trust_remote_code=False,
    num_labels: int | None = None,
    peft=False,
    checkpoint_dir=None,
    task: str = None,
):
    """
    download_generic_huggingface_model

    Downloads a generic AutoModelForCausalLM.
    model_name: str
    quantization: dict
    trust_remote_code: bool
    We should:
        - Create a method that is for other tasks
    """

    quantization_config = None

    if quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=quantization,
            llm_int8_skip_modules=[
                "lm_head",  # Language model head
                "classifier",  # Classification head
                "qa_outputs",  # QA task head
                "pooler",  # Pooling layer
                "pre_classifier",  # Pre Classifier layer, common in DistilBert
            ],
        )

    load_kwargs = {
        # "device_map": "auto",
        "dtype": torch.float32,  # torch.bfloat16 doesn't work for inputs that are int64
        "trust_remote_code": trust_remote_code,
    }

    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    # For encoder/sequence classification models (roberta/bert), they are under AutoModelForSequenceClassification.

    is_seq_classification = task == "SequenceClassification"
    if is_seq_classification:
        if num_labels is not None:
            load_kwargs["num_labels"] = num_labels

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_or_not(model_name, checkpoint_dir, peft), device_map="cuda:0", **load_kwargs
        )

        # Freeze embedding layer that doesn't work for Opacus
        model_type = model.config.model_type  # e.g., 'bert', 'roberta', 'distilbert'

        if hasattr(model, model_type):
            base_model = getattr(model, model_type)
            for param in base_model.embeddings.parameters():
                param.requires_grad = False

    else:
        model = AutoModelForCausalLM.from_pretrained( # source for loss=None warning later
            checkpoint_or_not(model_name, checkpoint_dir, peft), **load_kwargs
        )

        if hasattr(model, "transformer"):
            for name, module in model.transformer.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    for param in module.parameters():
                        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))

    if task == "InstructLM" and tokenizer.chat_template is None:
        tokenizer.padding_side = "left"

        special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}

        tokenizer.add_special_tokens(special_tokens)

        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )

        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, quantization_config


def checkpoint_or_not(model_name, checkpoint_dir_latest, peft):

    if checkpoint_dir_latest is not None and not peft:
        return checkpoint_dir_latest

    return model_name


class HuggingfaceLanguageModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        quantization: dict,
        ignore_index: int = -100,
        trust_remote_code: bool = False,
        peft: bool = False,
        checkpoint_dir: str | None = None,
        num_labels: int | None = None,
        task: str | None = None,
    ):

        super().__init__()

        self.ignore_index = ignore_index
        self.peft = peft
        self.task = task
        self.model, self.tokenizer, self.quantization_config = download_generic_huggingface_model(
            model_name=model_name,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
            num_labels=num_labels,
            peft=peft,
            checkpoint_dir=checkpoint_dir,
            task=task,
        )

        self.vocab_size = self.model.config.vocab_size
        self.is_seq_classification = self.task == "SequenceClassification"

    @property
    def config(self):
        return self.model.config

    def prepare_inputs_for_generation(self):
        """Expose the underlying model's method."""
        return self.model.prepare_inputs_for_generation

    def forward(self, x):
        """
        Return logits given tokenized batch or tensor input.
        """
        if isinstance(x, Mapping):
            out = self.model(**x)
        else:
            out = self.model(x)

        if hasattr(out, "logits"):
            return out.logits

        return out

    def forward_features(self, x):
        """
        Extract hidden features prior to classification/LM head.

        - Sequence classification models:
            * Prefer pooler_output, e.g., BERT.
            * Otherwise fallback to CLS token of last_hidden_state.
        - Causal LMs:
            * Return last_hidden_state (sequence length, hidden_size).
        """
        if isinstance(x, Mapping):
            out = self.model(**x, return_dict=True, output_hidden_states=True)
        else:
            out = self.model(x, return_dict=True, output_hidden_states=True)

        if self.is_seq_classification:
            # BERT-style: has pooler_output
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output  # (batch_size, hidden_size)

            # Otherwise: fallback to CLS token of final hidden state
            return out.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Causal LM: return final hidden states (all tokens)
        return out.last_hidden_state  # (batch_size, seq_len, hidden_size)

    def forward_head(self, features):

        head = self.get_classifier()
        if head is None:
            raise RuntimeError("No classifier/LM head found for this model")

        return head(features)

    def generate(self, *args, **kwargs):
        # NB: This requires `prepare_inputs_for_generation` method exists in the model.
        return self.model.generate(**args[0], **kwargs)

    # Encoder models (e.g., RoBERTa/BERT) commonly use classifier or score. Causal LMs use lm_head.
    def get_classifier(self):
        for attr in ["classifier", "score", "lm_head"]:
            if hasattr(self.model, attr):
                head = getattr(self.model, attr)
                if isinstance(head, torch.nn.Module):
                    return head

        return None

    def get_transforms(self):
        return self.tokenizer

    def save_model(self, path):
        # NB: Only saves the underlying model, no PEFT configuration.
        self.model.save_pretrained(path)

    def load_model(
        self,
        fpath: str,
        *,
        is_trainable: bool = True,
    ) -> None:

        if not fpath:
            raise ValueError('Cannot load Huggingface model from disk: fpath is required')

        if not os.path.isdir(fpath):
            raise FileNotFoundError(f'Cannot load Huggingface model from disk: checkpoint {fpath} missing')

        if self.task == "SequenceClassification":
            model_cls = AutoModelForSequenceClassification
        else:
            model_cls = AutoModelForCausalLM

        model_from_disk = model_cls.from_pretrained(
            fpath,
            torch_dtype=torch.float32, # NB: No quantization support!
        )

        try:
            device = next(self.model.parameters()).device
        except StopIteration as exc:
            raise RuntimeError('Cannot resolve device from Huggingface model parameters.') from exc

        model_from_disk = model_from_disk.to(device)

        self.model = model_from_disk
