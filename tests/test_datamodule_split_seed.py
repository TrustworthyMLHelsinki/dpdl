import datasets

from dpdl.datamodules import DataModule


def _validation_ids(training_seed, split_seed):
    train = datasets.Dataset.from_dict(
        {
            'id': list(range(1000)),
            'label': [index % 10 for index in range(1000)],
        }
    ).class_encode_column('label')
    test = datasets.Dataset.from_dict(
        {
            'id': list(range(1000, 1200)),
            'label': [index % 10 for index in range(200)],
        }
    ).class_encode_column('label')

    datamodule = DataModule.__new__(DataModule)
    datamodule._dataset_splits = datasets.DatasetDict(train=train, test=test)
    datamodule._label_field = 'label'
    datamodule.test_size = 0.1
    datamodule.val_size = 0.1
    datamodule.seed = training_seed
    datamodule.split_seed = split_seed
    datamodule.evaluation_mode = False
    datamodule._create_dataset_splits()
    return datamodule.val_dataset['id']


def test_split_seed_keeps_validation_rows_fixed_across_training_seeds():
    assert _validation_ids(7101, 7000) == _validation_ids(7102, 7000)
    assert _validation_ids(7101, 7000) != _validation_ids(7101, 7001)


def test_disease_holdout_stratifies_repeat_classes_and_balances_singletons():
    holdout = datasets.Dataset.from_dict({
        'id': list(range(22)),
        'disease': ['common_a'] * 10 + ['common_b'] * 10 + ['rare_a', 'rare_b'],
    }).class_encode_column('disease')

    module = DataModule.__new__(DataModule)
    module._label_field = 'disease'
    module.val_size = module.test_size = 0.1
    module.split_seed = 2303
    validation, test = module._split_disease_holdout(holdout)

    assert len(validation) == len(test) == 11
    assert set(validation['id']).isdisjoint(test['id'])
    assert set(validation['id']) | set(test['id']) == set(holdout['id'])
    for split in (validation, test):
        labels = [holdout.features['disease'].int2str(i) for i in split['disease']]
        assert labels.count('common_a') == labels.count('common_b') == 5
        assert sum(label.startswith('rare_') for label in labels) == 1

    module.seed = 7101
    validation_again, test_again = module._split_disease_holdout(holdout)
    assert validation['id'] == validation_again['id']
    assert test['id'] == test_again['id']


def test_train_only_disease_split_is_independent_of_training_seed():
    train = datasets.Dataset.from_dict({
        'id': list(range(204)),
        'disease': ['common_a'] * 100 + ['common_b'] * 100 + ['rare'] * 4,
    }).class_encode_column('disease')

    def split(task, training_seed):
        module = DataModule.__new__(DataModule)
        module._dataset_splits = datasets.DatasetDict(train=train)
        module._label_field = 'disease'
        module.task = task
        module.test_size = module.val_size = 0.1
        module.split_seed = 2303
        module.seed = training_seed
        module.evaluation_mode = False
        module._create_dataset_splits()
        return module

    first = split('DiseaseTask', 2807)
    second = split('DiseaseTask', 7101)
    assert first.val_dataset['id'] == second.val_dataset['id']
    assert first.test_dataset['id'] == second.test_dataset['id']
    ids = [set(part['id']) for part in (
        first.train_dataset, first.val_dataset, first.test_dataset
    )]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == set(train['id'])

    # The existing non-disease path remains independent of the training seed.
    ordinary = split('ImageClassification', 2807)
    ordinary_second = split('ImageClassification', 7101)
    assert ordinary.train_dataset['id'] == ordinary_second.train_dataset['id']
    assert ordinary.val_dataset['id'] == ordinary_second.val_dataset['id']
