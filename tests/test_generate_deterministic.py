import torch

from model import create_model


def test_generate_is_deterministic_with_greedy_settings():
    torch.manual_seed(0)

    model = create_model(size="nano", variant="baseline")
    model.eval()

    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))

    torch.manual_seed(42)
    first = model.generate(
        input_ids,
        max_new_tokens=5,
        do_sample=False,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    )

    torch.manual_seed(123)
    second = model.generate(
        input_ids,
        max_new_tokens=5,
        do_sample=False,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    )

    assert torch.equal(first, second)
