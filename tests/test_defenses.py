"""PROTOCOL v1.1 A9.2/A9.3: kl and ldifs regularise on the training batch.
Skipped where torch/transformers/peft are not installed (the rest of tests/ is pure python)."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from emr.defenses import assistant_token_mask, build_defense, every_kth_layer  # noqa: E402


def _tiny():
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=10,
                                   num_attention_heads=4, num_key_value_heads=2)
    torch.manual_seed(0)
    return cfg


def _batch():
    ids = torch.randint(0, 64, (2, 12))
    att = torch.ones_like(ids); att[1, 9:] = 0
    lab = ids.clone(); lab[:, :6] = -100; lab[1, 9:] = -100
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


def test_every_kth_layer():
    assert every_kth_layer(28) == [4, 9, 14, 19, 24]
    assert every_kth_layer(10) == [4, 9]


def test_assistant_mask_is_shifted():
    b = _batch()
    assert assistant_token_mask(b).shape == (2, 11)
    assert int(assistant_token_mask(b).sum()) == 6 + 3


def test_kl_zero_at_init_and_uses_training_batch():
    cfg = _tiny()
    m = peft.get_peft_model(transformers.Qwen2ForCausalLM(cfg),
                            peft.LoraConfig(r=4, target_modules="all-linear", task_type="CAUSAL_LM"))
    d = build_defense({"defense": "kl"}, m, None)          # no safe_iter needed any more
    assert float(d.extra_loss(0, _batch(), m)) == pytest.approx(0.0, abs=1e-6)


def test_ldifs_gradient_flows():
    cfg = _tiny()
    m = peft.get_peft_model(transformers.Qwen2ForCausalLM(cfg),
                            peft.LoraConfig(r=4, target_modules="all-linear", task_type="CAUSAL_LM",
                                            init_lora_weights=False))
    d = build_defense({"defense": "ldifs"}, m, None)
    assert d.layers == [4, 9]
    loss = d.extra_loss(0, _batch(), m)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
