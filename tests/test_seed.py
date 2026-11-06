import pytest

torch = pytest.importorskip('torch')

from dpdl.configurationmanager import ConfigurationManager


def _params(seed):
    return {
        'command': 'train',
        'epochs': 1,
        'seed': seed,
        'noise_multiplier': None,
        'max_grad_norm': 1.0,
        'target_epsilon': None,
    }


def test_resolve_seed_keeps_given_seed():
    config_manager = ConfigurationManager(_params(seed=0))
    config_manager.resolve_seed()

    assert config_manager.configuration.seed == 0


def test_resolve_seed_draws_and_records_seed():
    config_manager = ConfigurationManager(_params(seed=None))
    config_manager.resolve_seed()
    seed = config_manager.configuration.seed

    assert isinstance(seed, int)

    # clones (as in train-predict) must reuse the recorded seed, not draw a new one
    clone = config_manager.clone_with_overrides(command='predict')
    assert clone.configuration.seed == seed
