import logging
from typing import Any, Dict, Optional

import torch
import torchmetrics
from torchmetrics.text import Perplexity
import yaml


log = logging.getLogger(__name__)

_METRIC_MODULES = (
    torchmetrics,
    getattr(torchmetrics, 'classification', None),
    getattr(torchmetrics, 'text', None),
    getattr(torchmetrics, 'regression', None),
    getattr(torchmetrics, 'image', None),
    getattr(torchmetrics, 'audio', None),
)


def _build_custom_metrics(metric_config, key, default_kwargs, sync_on_compute):
    if not metric_config:
        return None
    return CustomMetricsFactory.build_metric_collection(metric_config[key], default_kwargs, sync_on_compute)


class ClassificationMetrics(torchmetrics.MetricCollection):
    def __init__(
        self,
        output_dim: int,
        sync: bool,
        with_confusion_matrix: bool,
        custom_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    ) -> None:
        # NB: If `sync_on_compute` is enabled, this breaks
        # distributed training. If this needs to be enabled,
        # then we also need to actually run the validation on
        # all the GPUs.
        metrics = {
            'MulticlassAccuracy': torchmetrics.classification.MulticlassAccuracy(
                num_classes=output_dim,
                average='macro',
                sync_on_compute=sync,
            ),
            'MulticlassAccuracyWithMicro': torchmetrics.classification.MulticlassAccuracy(
                num_classes=output_dim,
                average='micro',
                sync_on_compute=sync,
            ),
            'MulticlassAccuracyPerClass': torchmetrics.classification.MulticlassAccuracy(
                num_classes=output_dim,
                average='none',
                sync_on_compute=sync,
            ),
        }

        if with_confusion_matrix:
            metrics['ConfusionMatrix'] = torchmetrics.ConfusionMatrix(
                task='multiclass' if output_dim > 2 else 'binary',
                num_classes=output_dim,
                sync_on_compute=sync,
            )

        if custom_metrics:
            metrics.update(custom_metrics)

        super().__init__(metrics)

    def update(self, preds, target) -> None:
        # ConfusionMatrix expects class labels, not logits, unlike the
        # other classification metrics, which apply argmax internally.
        if 'ConfusionMatrix' not in self.keys():
            return super().update(preds, target)

        preds_labels = preds.argmax(dim=-1) if preds.ndim > target.ndim else preds

        for name, metric in self.items():
            if name == 'ConfusionMatrix':
                metric.update(preds_labels, target)
            else:
                metric.update(preds, target)

        return None


class LanguageModelMetrics(torchmetrics.MetricCollection):
    def __init__(
        self,
        vocab_size: int,
        ignore_index: int,
        sync: bool,
        custom_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    ) -> None:
        metrics = {
            'MulticlassAccuracy': torchmetrics.classification.MulticlassAccuracy(
                num_classes=vocab_size,
                average='micro',
                ignore_index=ignore_index,
                sync_on_compute=sync,
            ),
            'Perplexity': Perplexity(
                ignore_index=ignore_index,
                sync_on_compute=sync,
            ),
        }

        if custom_metrics:
            metrics.update(custom_metrics)

        super().__init__(metrics)

    def update(self, preds, target) -> None:
        # Perplexity expects 3D logits and 2D labels, unlike the other
        # metrics, which expect flattened inputs.
        if not hasattr(preds, 'ndim') or preds.ndim != 3:
            return super().update(preds, target)

        shift_logits = preds[:, :-1, :].contiguous()                      # (batch, seq_len-1, vocab)
        shift_labels = target[:, 1:].contiguous()                         # (batch, seq_len-1)
        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))  # (batch*(seq_len-1), vocab)
        shift_labels_flat = shift_labels.view(-1)                         # (batch*(seq_len-1))

        for name, metric in self.items():
            if name == 'Perplexity':
                metric.update(shift_logits, shift_labels)
            else:
                metric.update(shift_logits_flat, shift_labels_flat)

        return None


class CustomMetricsFactory:
    @staticmethod
    def _resolve_metric_class(metric_name: str):
        for module in _METRIC_MODULES:
            if module and hasattr(module, metric_name):
                return getattr(module, metric_name)
        raise ValueError(f'Metric class "{metric_name}" not found in torchmetrics.')

    @staticmethod
    def _build_metric(metric_spec: Dict[str, Any], default_kwargs: Dict[str, Any], sync_on_compute: bool):
        metric_name = metric_spec['name']
        metric_cls = CustomMetricsFactory._resolve_metric_class(metric_name)

        user_params = dict(metric_spec.get('params') or {})
        
        params = default_kwargs.copy()
        params.update(user_params)

        if 'sync_on_compute' not in params:
            params['sync_on_compute'] = sync_on_compute

        return metric_cls(**params)

    @staticmethod
    def _normalize_metric_entries(metric_entries, section_name: str, metric_config: str) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []

        for idx, entry in enumerate(metric_entries):
            if isinstance(entry, str):
                entry = {'name': entry, 'params': {}}
            elif not isinstance(entry, dict):
                raise ValueError(
                    f'Metric config file "{metric_config}" section "{section_name}" entry #{idx + 1} must be a mapping or string.'
                )

            metric_name = entry.get('name')
            if not metric_name:
                raise ValueError(
                    f'Metric config file "{metric_config}" section "{section_name}" entry #{idx + 1} is missing a metric name.'
                )

            params = entry.get('params') or {}
            if not isinstance(params, dict):
                raise ValueError(
                    f'Metric config file "{metric_config}" section "{section_name}" entry #{idx + 1} has invalid params.'
                )

            normalized.append({'name': metric_name, 'params': params})

        return normalized

    @staticmethod
    def read_metric_config(metric_config: Optional[str]) -> Dict[str, list[Dict[str, Any]]]:
        if not metric_config:
            return {'train_metrics': [], 'valid_metrics': [], 'test_metrics': []}

        with open(metric_config, 'rb') as fh:
            raw_config = yaml.safe_load(fh) or {}

        if not isinstance(raw_config, dict):
            raise ValueError(f'Metric config file "{metric_config}" must contain a mapping at the top level.')

        normalized_config = {
            'train_metrics': [],
            'valid_metrics': [],
            'test_metrics': [],
        }

        for key in ('train_metrics', 'valid_metrics', 'test_metrics'):
            entries = raw_config.get(key, None)
            if entries is None:
                continue
            if not isinstance(entries, list):
                raise ValueError(
                    f'Metric config file "{metric_config}" section "{key}" must contain a list of metrics.'
                )

            normalized_config[key].extend(CustomMetricsFactory._normalize_metric_entries(entries, key, metric_config))

        return normalized_config

    @staticmethod
    def build_metric_collection(metric_specs, default_kwargs: Dict[str, Any], sync_on_compute: bool) -> Dict[str, torchmetrics.Metric]:
        return {
            spec['name']: CustomMetricsFactory._build_metric(spec, default_kwargs, sync_on_compute)
            for spec in metric_specs
        }



class MetricsFactory:

    @staticmethod
    def get_metrics(
        configuration,
        output_dim: Optional[int] = None,
    ) -> Dict[str, torchmetrics.MetricCollection]:
        task = configuration.task
        metric_config = getattr(configuration, 'metric_config', None)
        metric_config = CustomMetricsFactory.read_metric_config(metric_config) if metric_config else None

        # we explicitly validate on all ranks
        train_sync, eval_sync = True, True

        if task in ('ImageClassification', 'SequenceClassification'):
            if torch.distributed.get_rank() == 0:
                log.info(f'Task is "{configuration.task}", initializing classification metrics.')

            if not output_dim or output_dim < 1:
                raise ValueError('output_dim required for classification tasks')

            classification_defaults = {
                'num_classes': output_dim,
                'task': 'multiclass' if output_dim > 2 else 'binary',
            }

            train = ClassificationMetrics(
                output_dim=output_dim,
                sync=train_sync,
                with_confusion_matrix=False,
                custom_metrics=_build_custom_metrics(metric_config, 'train_metrics', classification_defaults, train_sync),
            )
            valid = ClassificationMetrics(
                output_dim=output_dim,
                sync=eval_sync,
                with_confusion_matrix=False,
                custom_metrics=_build_custom_metrics(metric_config, 'valid_metrics', classification_defaults, eval_sync),
            )
            test = ClassificationMetrics(
                output_dim=output_dim,
                sync=eval_sync,
                with_confusion_matrix=True,
                custom_metrics=_build_custom_metrics(metric_config, 'test_metrics', classification_defaults, eval_sync),
            )

        elif task in ('CausalLM', 'InstructLM'):
            if torch.distributed.get_rank() == 0:
                log.info(f'Task is "{configuration.task}", initializing language model metrics.')

            vocab_size = int(output_dim)
            ignore_index = -100

            language_defaults = {
                'num_classes': vocab_size,
                'ignore_index': ignore_index,
                'task': 'multiclass',
            }

            train = LanguageModelMetrics(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=train_sync,
                custom_metrics=_build_custom_metrics(metric_config, 'train_metrics', language_defaults, train_sync),
            )
            valid = LanguageModelMetrics(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=eval_sync,
                custom_metrics=_build_custom_metrics(metric_config, 'valid_metrics', language_defaults, eval_sync),
            )
            test = LanguageModelMetrics(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=eval_sync,
                custom_metrics=_build_custom_metrics(metric_config, 'test_metrics', language_defaults, eval_sync),
            )

        else:
            raise ValueError(f'No metrics defined for task: {task}')

        metrics = {'train_metrics': train, 'valid_metrics': valid, 'test_metrics': test}
        return metrics
