from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .objectives import (
    boltzmann_targets,
    categorical_entropy,
    categorical_kl,
    clipped_policy_loss,
    dpo_loss,
    forward_kl,
    group_advantages,
)
from .policy import FactorizedRouter, Router, RouterConfig


@dataclass
class WarmStartConfig:
    temperature: float = 1.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 7
    seed: int = 42
    device: str = "cpu"
    hard_labels: bool = False


@dataclass
class RefineConfig:
    steps: int = 5000
    batch_size: int = 32
    group_size: int = 8
    inner_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    clip_range: float = 0.2
    advantage_epsilon: float = 1e-4
    beta_initial: float = 0.05
    entropy_alpha: float = 0.01
    adaptive_beta: bool = True
    beta_interval: int = 100
    beta_factor: float = 1.5
    kl_low: float = 0.005
    kl_high: float = 0.05
    seed: int = 42
    device: str = "cpu"
    algorithm: str = "grpo"
    dpo_beta: float = 0.1


def _tensor(value, device: str) -> Tensor:
    result = torch.as_tensor(value, dtype=torch.float32, device=device)
    if result.ndim != 2 or result.shape[0] == 0 or not torch.isfinite(result).all():
        raise ValueError("Expected a nonempty finite matrix")
    return result


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _warm_metrics(model: Router, features: Tensor, utilities: Tensor, temperature: float, support_only: bool) -> dict:
    model.eval()
    log_probs = model.support_log_probs(features) if support_only else model.joint_log_probs(features).flatten(1)
    targets = boltzmann_targets(utilities, temperature)
    return {
        "kl": float(forward_kl(log_probs, targets)),
        "accuracy": float((log_probs.argmax(-1) == utilities.argmax(-1)).float().mean()),
    }


def warm_start(
    features,
    utilities,
    dev_features=None,
    dev_utilities=None,
    config: WarmStartConfig | None = None,
    router_config: RouterConfig | None = None,
    model: Router | None = None,
) -> tuple[Router, list[dict]]:
    c = config or WarmStartConfig()
    if c.batch_size < 1 or c.max_epochs < 1 or c.patience < 1 or c.learning_rate <= 0 or c.weight_decay < 0:
        raise ValueError("Invalid warm-start optimizer or loop configuration")
    _seed(c.seed)
    x, u = _tensor(features, c.device), _tensor(utilities, c.device)
    if len(x) != len(u):
        raise ValueError("One utility row is required per query")
    if model is None:
        rc = router_config or RouterConfig(input_dim=x.shape[1])
        model = FactorizedRouter(rc)
    elif router_config is not None and model.config != router_config:
        raise ValueError("Supplied model and router_config disagree")
    model = model.to(c.device)
    if x.shape[1] != model.config.input_dim or u.shape[1] not in (3, model.config.n_actions):
        raise ValueError("Features or utility action count do not match router configuration")
    support_only = u.shape[1] == 3
    if support_only:
        model.reset_thinking_uniform()
    if (dev_features is None) != (dev_utilities is None):
        raise ValueError("Provide both dev_features and dev_utilities, or neither")
    xd = x if dev_features is None else _tensor(dev_features, c.device)
    ud = u if dev_utilities is None else _tensor(dev_utilities, c.device)
    if len(xd) != len(ud) or xd.shape[1] != x.shape[1] or ud.shape[1] != u.shape[1]:
        raise ValueError("Development matrices must match the training feature/action layout")
    target = boltzmann_targets(u, c.temperature)
    if c.hard_labels:
        target = torch.nn.functional.one_hot(u.argmax(-1), u.shape[1]).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
    generator = torch.Generator().manual_seed(c.seed)
    best_accuracy, stale = -float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    for epoch in range(1, c.max_epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator).to(x.device)
        total_loss = 0.0
        for idx in order.split(c.batch_size):
            optimizer.zero_grad(set_to_none=True)
            log_probs = model.support_log_probs(x[idx]) if support_only else model.joint_log_probs(x[idx]).flatten(1)
            loss = forward_kl(log_probs, target[idx])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)
        metrics = _warm_metrics(model, xd, ud, c.temperature, support_only)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(x),
            "dev_kl": metrics["kl"],
            "dev_accuracy": metrics["accuracy"],
            "selection_split": "dev" if dev_features is not None else "train",
            "warm_actions": u.shape[1],
        })
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, stale = metrics["accuracy"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= c.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    model.training_config = {"stage": 1, **asdict(c)}
    return model, history


@torch.no_grad()
def sample_stratified_actions(log_probs: Tensor, group_size: int, generator: torch.Generator | None = None) -> Tensor:
    """Reserve one conditional draw per support; draw remaining actions jointly; shuffle."""
    if log_probs.ndim != 3 or log_probs.shape[1] != 3 or group_size < 1:
        raise ValueError("Expected joint log probabilities (batch, 3, thinking) and group_size>=1")
    logits = log_probs.cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("Invalid action probabilities")
    probabilities = logits.flatten(1).softmax(-1).reshape_as(logits)
    batch, _, n_thinking = probabilities.shape
    if group_size < 3:
        return torch.multinomial(probabilities.flatten(1), group_size, replacement=True, generator=generator).to(log_probs.device)
    conditional = logits.softmax(-1)
    forced = torch.stack([
        torch.multinomial(conditional[:, mu], 1, generator=generator).squeeze(1) + mu * n_thinking
        for mu in range(3)
    ], dim=1)
    if group_size > 3:
        free = torch.multinomial(probabilities.flatten(1), group_size - 3, replacement=True, generator=generator)
        forced = torch.cat((forced, free), dim=1)
    permutations = torch.rand(batch, group_size, generator=generator).argsort(1)
    return forced.gather(1, permutations).to(log_probs.device)


def refine(
    model: Router,
    features,
    reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    config: RefineConfig | None = None,
    reference_model: Router | None = None,
    progress: Callable[[dict], None] | None = None,
) -> tuple[Router, list[dict]]:
    c = config or RefineConfig()
    if min(c.steps, c.batch_size, c.inner_steps, c.beta_interval) < 1 or c.group_size < 2:
        raise ValueError("Invalid refinement loop configuration")
    if c.algorithm not in ("grpo", "dr_grpo", "rloo", "dpo"):
        raise ValueError("algorithm must be grpo, dr_grpo, rloo, or dpo")
    if c.learning_rate <= 0 or c.weight_decay < 0 or c.beta_initial < 0 or c.entropy_alpha < 0:
        raise ValueError("Optimizer and regularization coefficients must be nonnegative")
    if not 0 <= c.kl_low < c.kl_high or c.beta_factor <= 1:
        raise ValueError("Invalid adaptive KL schedule")
    _seed(c.seed)
    model.to(c.device).eval()
    x = _tensor(features, c.device)
    if x.shape[1] != model.config.input_dim:
        raise ValueError("Features do not match router input dimension")
    reference = copy.deepcopy(reference_model if reference_model is not None else model).to(c.device).eval()
    if reference.config != model.config:
        raise ValueError("The reference policy configuration must match the trained policy")
    reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
    rng = np.random.default_rng(c.seed)
    generator = torch.Generator().manual_seed(c.seed)
    beta = c.beta_initial
    history: list[dict] = []
    for step in range(1, c.steps + 1):
        query_indices = rng.integers(len(x), size=c.batch_size)
        xb = x[torch.as_tensor(query_indices, device=x.device)]
        with torch.no_grad():
            old_joint = model.joint_log_probs(xb)
            actions = sample_stratified_actions(old_joint, c.group_size, generator)
            old_selected = old_joint.flatten(1).gather(1, actions)
            ref_log_probs = reference.joint_log_probs(xb).flatten(1)
        raw_rewards = reward_fn(query_indices.copy(), actions.cpu().numpy().copy())
        rewards = torch.as_tensor(raw_rewards, dtype=torch.float32, device=c.device)
        if rewards.shape != actions.shape or not torch.isfinite(rewards).all():
            raise ValueError("reward_fn must return finite rewards with shape (batch_size, group_size)")
        advantages = group_advantages(rewards, c.advantage_epsilon, "grpo" if c.algorithm == "dpo" else c.algorithm)
        losses = []
        for _ in range(c.inner_steps):
            optimizer.zero_grad(set_to_none=True)
            log_probs = model.joint_log_probs(xb).flatten(1)
            selected = log_probs.gather(1, actions)
            if c.algorithm == "dpo":
                hi, lo = rewards.argmax(1), rewards.argmin(1)
                chosen, rejected = actions.gather(1, hi[:, None]).squeeze(1), actions.gather(1, lo[:, None]).squeeze(1)
                valid = rewards.max(1).values > rewards.min(1).values
                policy_loss = dpo_loss(log_probs[valid], ref_log_probs[valid], chosen[valid], rejected[valid], c.dpo_beta) if valid.any() else log_probs.sum() * 0
            elif c.algorithm == "rloo":
                policy_loss = -(selected * advantages.detach()).mean()
            else:
                policy_loss = clipped_policy_loss(selected, old_selected, advantages, c.clip_range)
            kl = categorical_kl(log_probs, ref_log_probs)
            entropy = categorical_entropy(log_probs)
            loss = policy_loss + beta * kl - c.entropy_alpha * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite policy loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            current = model.joint_log_probs(xb).flatten(1)
            observed_kl = float(categorical_kl(current, ref_log_probs))
            entropy_value = float(categorical_entropy(current))
        beta_used = beta
        if c.adaptive_beta and step % c.beta_interval == 0:
            if observed_kl > c.kl_high:
                beta *= c.beta_factor
            elif observed_kl < c.kl_low:
                beta /= c.beta_factor
        row = {
            "step": step,
            "optimizer_steps": step * c.inner_steps,
            "sampled_actions": step * c.batch_size * c.group_size,
            "mean_reward": float(rewards.mean()),
            "loss": float(np.mean(losses)),
            "kl_to_stage1": observed_kl,
            "entropy": entropy_value,
            "beta": beta_used,
            "next_beta": beta,
            "zero_variance_groups": int((rewards.std(-1, correction=0) == 0).sum()),
            "algorithm": c.algorithm,
        }
        history.append(row)
        if progress is not None:
            progress(row)
    model.eval()
    model.training_config = {"stage": 2, **asdict(c)}
    return model, history
