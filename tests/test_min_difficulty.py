import random
from multiprocessing import Value

import pytest

pytest.importorskip("torch")

from curriculum import TaskCurriculumState
from data import AlgorithmicDataset


def test_min_difficulty_clamps_curriculum_and_weights():
    curriculum = TaskCurriculumState(
        manager=None,
        tasks=["addition"],
        init_difficulty=0.1,
        min_difficulty=0.4,
        ema_decay=0.5,
        step_size=0.2,
        min_task_evals=1,
    )

    assert curriculum.get_task_state("addition")["difficulty"] == pytest.approx(0.4)

    metrics = curriculum.update_metrics("addition", accuracy=0.9, loss=None, step=1)
    assert metrics["ema_acc"] == pytest.approx(0.9)

    increased = curriculum.step_curriculum("addition", step=1, cooldown=0)
    assert increased == pytest.approx(0.6)

    curriculum.update_metrics("addition", accuracy=0.0, loss=None, step=2)
    decreased = curriculum.step_curriculum("addition", step=2, cooldown=0)
    assert decreased == pytest.approx(0.4)

    expected_weight = max(0.05, 1.0 - curriculum.get_task_state("addition")["ema_acc"])
    assert curriculum.get_sampling_weight("addition") == pytest.approx(expected_weight)
    jittered = curriculum.get_difficulty("addition", jitter_prob=1.0)
    assert 0.0 <= jittered <= 0.4


@pytest.mark.parametrize("schedule", ["fixed", "linear", "phased", "smooth", "warmup_ramp"])
def test_min_difficulty_applies_to_dataset_schedules(schedule):
    dataset = AlgorithmicDataset(
        tokenizer=None,
        num_examples=1,
        max_seq_len=8,
        difficulty_schedule=schedule,
        easy_mix_frac=0.2,
        min_difficulty=0.4,
    )
    rng = random.Random(0)
    sample = dataset._sample_difficulty(rng, progress=0.0)
    assert 0.4 <= sample <= 1.0


def test_min_difficulty_applies_to_shared_value():
    difficulty_value = Value("d", 0.1)
    dataset = AlgorithmicDataset(
        tokenizer=None,
        num_examples=1,
        max_seq_len=8,
        difficulty_value=difficulty_value,
        min_difficulty=0.4,
    )
    rng = random.Random(0)
    sample = dataset._sample_difficulty(rng, progress=0.5)
    assert sample == pytest.approx(0.4)
