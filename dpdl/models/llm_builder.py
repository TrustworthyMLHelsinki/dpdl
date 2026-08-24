import os
import logging

from .hugging_face_models import HuggingfaceLanguageModel

from dpdl.configurationmanager import Configuration

log = logging.getLogger(__name__)



class LLMBuilder:

    @staticmethod
    def matches(configuration):
        return configuration.task in ("CausalLM", "InstructLM", "SequenceClassification", "DiseaseTask")

    @staticmethod
    def get_model(
        configuration: Configuration,
        output_dim: int | None,
        checkpoints_dir_latest: str | None = None,
    ):

        model_instance = HuggingfaceLanguageModel(
            configuration.model_name,
            configuration.load_in_4bit,
            num_labels=output_dim,
            peft=configuration.peft,
            checkpoint_dir=checkpoints_dir_latest,
            task=configuration.task,
        )

        transforms = model_instance.get_transforms()

        # For generative LM tasks (CausalLM / InstructLM / DiseaseTask), the
        # downstream MetricsFactory uses output_dim as vocab_size for a
        # token-level MulticlassAccuracy, so we MUST override any datamodule
        # value (for DiseaseTask that value is the disease-class count, not the
        # vocab). SequenceClassification keeps the datamodule-provided count
        # because its classifier head is sized by the label count.
        force_vocab_size = configuration.task in ("CausalLM", "InstructLM", "DiseaseTask")

        if output_dim is None or force_vocab_size:
            try:
                output_dim = int(model_instance.model.num_classes) \
                    if configuration.task == "SequenceClassification" \
                    else int(model_instance.config.vocab_size)
            except AttributeError:
                raise ValueError('Output dimension not given and unable to infer it.')

        return model_instance, transforms, output_dim