"""VaultGemma sequence classification.

The pretrained decoder and token embeddings remain trainable. A small linear
head reads the last non-padding token; no vocabulary logits or KV cache are
allocated. The classifier can be saved/reloaded with Hugging Face checkpoints.
"""

import torch
from torch import nn
from transformers import (
    VaultGemmaModel,
    VaultGemmaPreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput


class VaultGemmaScaledEmbedding(nn.Embedding):
    """Wrapper for embedding scale compatibility with Opacus.

    HF's scaled embedding stores a fixed scale as a registered buffer. Opacus
    rejects buffers on trainable layers. Keep HF's scalar scale attribute instead,
    so its generic functorch fallback can compute per-example gradients.
    """

    def __init__(self, original):
        super().__init__(
            original.num_embeddings,
            original.embedding_dim,
            padding_idx=original.padding_idx,
            _weight=original.weight,
        )
        self.scalar_embed_scale = original.scalar_embed_scale

    def forward(self, input_ids):
        embeddings = super().forward(input_ids)

        # Device and dtype can change after init. Round the scale to the
        # embedding dtype, as HF does, without registering a module buffer.
        scale = torch.tensor(
            self.scalar_embed_scale,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )

        return embeddings * scale


class VaultGemmaForSequenceClassification(VaultGemmaPreTrainedModel):
    """Pretrained VaultGemma backbone with a last-token classification head."""

    def __init__(self, config):
        super().__init__(config)

        # Disable KV cache in the HF transformers config, we don't need
        # this for classification, as we don't generate subsequent tokens.
        config.use_cache = False

        # Create the VaultGemma model using HF transformers.
        self.model = VaultGemmaModel(config)
        embedding = self.model.get_input_embeddings()

        # Transformers starting from version 5.4 stores the scalar embedding
        # scaling factor in a buffer. See `VaultGemmaScaledEmbedding` docs for
        # details.
        if hasattr(embedding, "scalar_embed_scale"):
            self.model.set_input_embeddings(VaultGemmaScaledEmbedding(embedding))

        # Add a classification head after the VaultGemmaModel "backbone"
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # Let HF VaultGemmaPreTrainedModel handle the rest of the init
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
    ):
        """Classify the last non-padding token, supporting either padding side."""
        if attention_mask is None:
            attention_mask = (
                input_ids.ne(self.config.pad_token_id)
                if self.config.pad_token_id is not None
                else torch.ones_like(input_ids, dtype=torch.bool)
            )

        valid = attention_mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("Each sequence must contain a non-padding token.")

        # Positions count real tokens so padding side does not change features.
        if position_ids is None:
            position_ids = (valid.long().cumsum(dim=1) - 1).clamp_min(0)

        # Get the "features" out of the backbone.
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

        indices = torch.arange(input_ids.shape[1], device=input_ids.device)

        # We classify based on the last token, this is what the GPT based
        # classification models also do: "GPT2ForSequenceClassification uses
        # the last token in order to do the classification, as other causal
        # models (e.g. GPT-1) do."
        # - https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2ForSequenceClassification
        last = torch.where(valid, indices, -1).amax(dim=1)

        # Use the last non-padding token's hidden representation for classification
        features = hidden[
            torch.arange(input_ids.shape[0], device=input_ids.device), last
        ]

        # Forward pass
        logits = self.score(features)

        return SequenceClassifierOutput(logits=logits)
