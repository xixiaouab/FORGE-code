import copy

import numpy as np
import pytest
import torch

from forge.checkpoint import load_checkpoint, save_checkpoint
from forge.objectives import (
    DatasetCostNormalizer,
    StructuredStandardizer,
    boltzmann_targets,
    categorical_kl,
    clipped_policy_loss,
    group_advantages,
    pareto_utility,
)
from forge.policy import FactorizedRouter, FlatRouter, RouterConfig, parameter_count
from forge.training import RefineConfig, WarmStartConfig, refine, sample_stratified_actions, warm_start


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_config(n_thinking=2):
    return RouterConfig(input_dim=4, n_thinking=n_thinking, hidden_dim=12, embedding_dim=3, dropout=0, variant="test")


def test_policy_probability_factorization_and_uniform_thinking():
    model = FactorizedRouter(tiny_config()).eval()
    features = torch.randn(7, 4)
    p = model.joint_log_probs(features).exp()
    assert p.shape == (7, 3, 2)
    torch.testing.assert_close(p.sum((1, 2)), torch.ones(7))
    torch.testing.assert_close(p[:, :, 0], p[:, :, 1])
    torch.testing.assert_close(p.sum(-1), model.support_log_probs(features).exp())


def test_parameter_counts_match_declared_architecture():
    assert parameter_count(FactorizedRouter()) == 269397
    assert parameter_count(FactorizedRouter(RouterConfig(n_thinking=4))) == 269943
    assert parameter_count(FlatRouter()) == 269574


def test_boltzmann_and_group_advantage_equations():
    utilities = torch.tensor([[1.0, 0.0, -1.0]])
    target = boltzmann_targets(utilities)
    torch.testing.assert_close(target, utilities.exp() / utilities.exp().sum(1, keepdim=True))
    rewards = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    advantages = group_advantages(rewards)
    expected = torch.tensor([-1.0, 0.0, 1.0]) / (np.sqrt(2 / 3) + 1e-4)
    torch.testing.assert_close(advantages[0], expected)
    torch.testing.assert_close(advantages[1], torch.zeros(3))
    torch.testing.assert_close(group_advantages(rewards, method="rloo")[0], torch.tensor([-1.5, 0.0, 1.5]))
    torch.testing.assert_close(group_advantages(rewards, method="dr_grpo")[0], torch.tensor([-1.0, 0.0, 1.0]))


def test_clipped_policy_loss_correctly_stops_clipped_gradients():
    ratios = torch.tensor([[1.5, 0.5, 1.5, 0.5]])
    log_probs = ratios.log().requires_grad_()
    advantages = torch.tensor([[1.0, -1.0, -1.0, 1.0]])
    loss = clipped_policy_loss(log_probs, torch.zeros_like(log_probs), advantages)
    torch.testing.assert_close(loss, torch.tensor(0.15))
    loss.backward()
    torch.testing.assert_close(log_probs.grad, torch.tensor([[0.0, 0.0, 0.375, -0.125]]))


def test_stratified_sampler_coverage_and_reproducibility():
    probabilities = torch.tensor([[[0.7, 0.1], [0.04, 0.06], [0.07, 0.03]]]).repeat(32, 1, 1)
    a = sample_stratified_actions(probabilities.log(), 8, torch.Generator().manual_seed(7))
    b = sample_stratified_actions(probabilities.log(), 8, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    assert a.shape == (32, 8)
    assert all(set(row.tolist()) == {0, 1, 2} for row in a // 2)


def test_stratified_sampling_survives_tiny_support_probabilities():
    logits = torch.tensor([[[0.0, -1.0], [-1000.0, -1001.0], [-2000.0, -2001.0]]])
    actions = sample_stratified_actions(logits, 8, torch.Generator().manual_seed(2))
    assert set((actions // 2).flatten().tolist()) == {0, 1, 2}


def test_training_only_cost_normalization_and_reward():
    cin = np.array([[10, 20, 100], [5, 25, 50]])
    cout = np.array([[10, 20, 50], [5, 10, 25]])
    normalizer = DatasetCostNormalizer().fit(cin, cout, ["a", "a"], split="train")
    reward = pareto_utility(np.ones((2, 3)), cin, cout, ["a", "a"], normalizer)
    np.testing.assert_allclose(reward, 1 - 0.1 * cin / 100 - 0.2 * cout / 50)
    nc, _ = normalizer.transform(np.array([[200]]), np.array([[25]]), ["a"])
    np.testing.assert_allclose(nc, [[2]])
    with pytest.raises(ValueError, match="training"):
        normalizer.fit(cin, cout, ["a", "a"], split="test")
    with pytest.raises(ValueError, match="No training"):
        normalizer.transform(cin, cout, ["b", "b"])
    with pytest.raises(ValueError, match="percentages"):
        pareto_utility(np.ones((2, 3)) * 90, cin, cout, ["a", "a"], normalizer)


def test_standardizer_preserves_embeddings_and_train_statistics():
    features = np.arange(4 * 772, dtype=np.float32).reshape(4, 772)
    standardizer = StructuredStandardizer().fit(features, split="train")
    transformed = standardizer.transform(features)
    np.testing.assert_array_equal(transformed[:, :768], features[:, :768])
    np.testing.assert_allclose(transformed[:, 768:].mean(0), 0, atol=1e-7)
    np.testing.assert_allclose(transformed[:, 768:].std(0), 1, atol=1e-7)
    with pytest.raises(ValueError, match="train"):
        StructuredStandardizer().fit(features, split="dev")
    restored = StructuredStandardizer.from_dict(standardizer.to_dict())
    np.testing.assert_array_equal(restored.transform(features), transformed)


def test_warm_start_improves_support_kl_and_keeps_uniform_thinking():
    torch.manual_seed(5)
    model = FactorizedRouter(tiny_config())
    features = np.ones((32, 4), dtype=np.float32)
    utilities = np.tile([0.0, 0.0, 1.0], (32, 1))
    target = boltzmann_targets(torch.tensor(utilities, dtype=torch.float32))
    with torch.no_grad():
        initial = torch.nn.functional.kl_div(model.support_log_probs(torch.from_numpy(features)), target, reduction="batchmean")
    model, history = warm_start(features, utilities, features, utilities,
                                WarmStartConfig(max_epochs=12, patience=8, learning_rate=0.02), model=model)
    p = model.joint_log_probs(torch.from_numpy(features)).exp()
    final = torch.nn.functional.kl_div(model.support_log_probs(torch.from_numpy(features)), target, reduction="batchmean")
    assert final < initial
    assert history[0]["warm_actions"] == 3
    assert history[0]["selection_split"] == "dev"
    torch.testing.assert_close(p[:, :, 0], p[:, :, 1])


def test_warm_start_full_alphabet_learns_thinking():
    features = np.ones((32, 4), dtype=np.float32)
    utilities = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 2.0], (32, 1))
    model, history = warm_start(features, utilities, config=WarmStartConfig(max_epochs=4, learning_rate=0.02), router_config=tiny_config())
    p = model.joint_log_probs(torch.from_numpy(features)).exp()
    assert history[0]["warm_actions"] == 6
    assert (p[:, 2, 1] > p[:, 2, 0]).all()


def test_refinement_callback_accounting_determinism_and_learning():
    torch.manual_seed(9)
    initial = FactorizedRouter(tiny_config())
    features = np.ones((16, 4), dtype=np.float32)
    calls = []

    def reward(indices, actions):
        calls.append((indices, actions))
        return (actions == 5).astype(np.float32)

    config = RefineConfig(steps=8, batch_size=8, group_size=8, inner_steps=4, learning_rate=0.02, adaptive_beta=False)
    before = initial.joint_log_probs(torch.from_numpy(features)).exp()[:, 2, 1].mean()
    reference = copy.deepcopy(initial)
    model, history = refine(copy.deepcopy(initial), features, reward, config, reference_model=reference)
    duplicate, history2 = refine(copy.deepcopy(initial), features, reward, config, reference_model=reference)
    after = model.joint_log_probs(torch.from_numpy(features)).exp()[:, 2, 1].mean()
    assert after > before
    assert history == history2
    assert len(calls) == 16
    assert all(a.shape == (8, 8) and q.shape == (8,) for q, a in calls)
    assert history[-1]["sampled_actions"] == 512
    assert history[-1]["optimizer_steps"] == 32
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, duplicate.state_dict()[key])
    for key, value in initial.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key])


def test_zero_variance_refinement_and_adaptive_beta():
    model = FactorizedRouter(tiny_config())
    features = np.ones((4, 4), dtype=np.float32)
    _, history = refine(model, features, lambda q, a: np.ones(a.shape),
                        RefineConfig(steps=1, batch_size=4, beta_interval=1, entropy_alpha=0, weight_decay=0))
    assert history[0]["zero_variance_groups"] == 4
    assert history[0]["next_beta"] == pytest.approx(0.05 / 1.5)
    assert abs(history[0]["kl_to_stage1"]) < 1e-7


@pytest.mark.parametrize("algorithm", ["dpo", "dr_grpo", "rloo"])
def test_optional_ablation_objectives_execute(algorithm):
    model = FactorizedRouter(tiny_config())
    _, history = refine(model, np.ones((8, 4), dtype=np.float32), lambda q, a: (a == 5).astype(np.float32),
                        RefineConfig(steps=2, batch_size=4, algorithm=algorithm))
    assert np.isfinite(history[-1]["loss"])


@pytest.mark.parametrize("factory", [FactorizedRouter, FlatRouter])
def test_checkpoint_roundtrip_and_feature_contract(tmp_path, factory):
    model = factory(tiny_config()).eval()
    names = ["f0", "f1", "f2", "f3"]
    features = torch.randn(4, 4)
    scaler = StructuredStandardizer(embedding_dim=2).fit(features.numpy(), split="train")
    path = tmp_path / "policy.pt"
    save_checkpoint(path, model, feature_names=names, standardizer=scaler, metadata={"run": "test"})
    loaded, state = load_checkpoint(path, expected_variant="test", expected_feature_names=names, expected_n_thinking=2)
    torch.testing.assert_close(loaded(features), model(features))
    assert state["metadata"]["run"] == "test"
    with pytest.raises(ValueError, match="variant"):
        load_checkpoint(path, expected_variant="lite")
    with pytest.raises(ValueError, match="order"):
        load_checkpoint(path, expected_feature_names=list(reversed(names)))
    with pytest.raises(ValueError, match="alphabet"):
        load_checkpoint(path, expected_n_thinking=4)
    state["action_layout"] = "thinking-major"
    torch.save(state, path)
    with pytest.raises(ValueError, match="layout"):
        load_checkpoint(path)


def test_joint_kl_uses_current_to_reference_direction():
    p = torch.tensor([[0.1, 0.9]])
    q = torch.tensor([[0.5, 0.5]])
    expected = (p * (p.log() - q.log())).sum()
    torch.testing.assert_close(categorical_kl(p.log(), q.log()), expected)
