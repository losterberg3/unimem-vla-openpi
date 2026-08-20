import dataclasses
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def inner(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        return optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), self.inner(lr, weight_decay_mask))


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def inner(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        return self.inner(lr, weight_decay_mask)


def _param_leaf_label(path, _: at.PyTree) -> str:
    """Labels trainable leaves for multi_transform: phase_head MLP vs backbone."""
    path_str = "/".join(str(k) for k in path)
    return "phase_head" if "phase_head" in path_str else "backbone"


def _label_trainable_params(params: at.PyTree) -> at.PyTree:
    return jax.tree_util.tree_map_with_path(_param_leaf_label, params)


def create_optimizer(
    optimizer: OptimizerConfig,
    lr_schedule: LRScheduleConfig,
    weight_decay_mask: at.PyTree | None = None,
    *,
    phase_head_lr_multiplier: float = 1.0,
) -> optax.GradientTransformation:
    """Builds the optimizer. When ``phase_head_lr_multiplier`` != 1, uses a higher LR for ``phase_head`` params."""
    base_lr = lr_schedule.create()
    if phase_head_lr_multiplier == 1.0:
        return optimizer.create(base_lr, weight_decay_mask=weight_decay_mask)

    if not isinstance(optimizer, AdamW):
        raise ValueError(
            "phase_head_lr_multiplier != 1.0 is only supported with AdamW; "
            f"got {type(optimizer).__name__}"
        )

    if callable(base_lr):

        def phase_lr(step):
            return base_lr(step) * phase_head_lr_multiplier

    else:
        phase_lr = base_lr * phase_head_lr_multiplier

    clip = optax.clip_by_global_norm(optimizer.clip_gradient_norm)
    inner = optax.multi_transform(
        {
            "backbone": optimizer.inner(base_lr, weight_decay_mask=weight_decay_mask),
            "phase_head": optimizer.inner(phase_lr, weight_decay_mask=weight_decay_mask),
        },
        _label_trainable_params,
    )
    return optax.chain(clip, inner)
