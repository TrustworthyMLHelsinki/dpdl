import logging
import torch
import os
import sys
import importlib

from typing import Any, Dict, Optional

from .model_base import ModelBase

from .llm_builder import LLMBuilder
from .timm_builder import TimmBuilder
from .custom_builder import CustomBuilder

from dpdl.configurationmanager import Configuration, Hyperparameters
from dpdl.peft import PeftFactory

log = logging.getLogger(__name__)

def add_noise_to_weights(model, noise_level):
    for name, param in model.named_parameters():
        if 'weight' in name:
            noise = torch.randn(param.size()) * noise_level
            param.data.add_(noise)

def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint by modification time"""
    if checkpoint_dir is None:
        return None

    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        d
        for d in os.listdir(checkpoint_dir)
        if d.startswith('checkpoint_step_') and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]

    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, latest)

class ModelFactory:

    @staticmethod
    def get_model(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        output_dim: int,
        loss_fn: torch.nn,
        metrics: Optional[Dict[str, Any]] = None
    ):

        """
        Create a model instance based on the configuration, with support for PEFT and zeroing head weights.

        Parameters:
        - configuration: Configuration object containing model specs.
        - hyperparams: Optional hyperparameters, not directly used here.
        - output_dim: The number of classes for a classification problem

        Returns:
        - A tuple of (ModelBase instance, Data Transforms).
        """

        """
        TO DO: create LLM_base class, similar to ModelBase, but for LLMs?
        or just use ModelBase directly?
        """
        # NOTE: need to check if eg model loads checkpoints weights automatically!

        checkpoints_dir_latest = get_latest_checkpoint(
            #configuration.checkpoints_dir
            os.path.join(configuration.log_dir, 'checkpoints')
        )

        if LLMBuilder.matches(configuration):
            model_instance, transforms, output_dim = LLMBuilder.get_model(configuration, output_dim, checkpoints_dir_latest)
        elif CustomBuilder.matches(configuration):
            model_instance, transforms, output_dim = CustomBuilder.get_model(configuration, output_dim)
        elif TimmBuilder.matches(configuration):
            model_instance, transforms, output_dim = TimmBuilder.get_model(configuration, output_dim)
        else:
            raise ValueError(f"No model buildable for name '{configuration.model_name}' and task '{configuration.task}'!")

        # Wrap the instantiated model with ModelBase
        model = ModelBase(
            model_instance=model_instance,
            output_dim=output_dim,
            use_feature_cache=configuration.cache_features,
            criterion=loss_fn,
            metrics=metrics
        )

        # Add noise to (pretrained) weights?
        if configuration.weight_perturbation_level > 0:
            add_noise_to_weights(model, configuration.weight_perturbation_level)

        # zero the head weights?
        if configuration.zero_head:
            model.zero_head_weights()

        # should we do Parameter Efficient Fine-Tuning (PEFT)?
        if configuration.peft:
            model = PeftFactory.get_peft_model(model, configuration, checkpoints_dir_latest)

        return model, transforms, output_dim
