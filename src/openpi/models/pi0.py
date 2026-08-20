import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models.event_head import EventHead
import optax

logger = logging.getLogger("openpi")

# Event head (internally still named "phase_head" — see README for why): 0..8 = semantic
# events; 8 = dataset label -1 ("unsupervised / ignore" bucket), now trained.
EVENT_LOGITS_NUM_CLASSES = 12
EVENT_LABEL_IGNORE_TARGET_CLASS = 11


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    
    # Compute cumsum on the 1-D mask_ar before broadcasting to batch dim.
    # Broadcasting first produces a (B, N) constant that XLA folds into a single
    # global-shaped buffer and distributes via CUDA peer access; computing cumsum
    # on (N,) keeps it as a small replicated constant that each device copies
    # independently, avoiding peer-access failures on multi-GPU setups.
    if mask_ar.ndim == 1:
        cumsum_1d = jnp.cumsum(mask_ar, axis=0)  # (N,)
        cumsum = jnp.broadcast_to(cumsum_1d[None, :], input_mask.shape)
    else:
        mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
        cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    """Flow-matching VLA: a Gemma LLM prefix/suffix over SigLIP image tokens, tokenized
    language, and (for Pi0, not Pi0.5) a continuous state token, denoising a chunk of
    actions via `sample_actions`/`sample_actions_event`.

    Two optional additions on top of the base architecture, both gated by
    `Pi0Config` flags and independent of each other except where noted:
      - `event_tracking`: adds `self.phase_head` (an `EventHead` MLP; see
        models/event_head.py and the README's "note on terminology" for why the
        attribute keeps its old name) that classifies a semantic event from the
        pooled prefix, trained via the auxiliary loss in `compute_loss_event`.
      - `video_encoder`: SigLIP ingests `num_frames` per camera instead of 1, via
        temporal attention (models/siglip.py). Only the current frame's tokens are
        forwarded to Gemma (see `embed_prefix`) — history is not accumulated at the
        Gemma stage, so `video_encoder` alone (without `event_tracking`) trains a
        purely temporal-context policy with no event classification.
      - `video_encoder=True` combined with `event_tracking=True` is only exercised by
        the "keyframe" data configs (training/config.py's *EventKeyframeDataConfig) —
        see `Pi0Config.video_encoder`'s docstring for why video and events aren't
        symmetric.
    """

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.event_tracking = config.event_tracking
        self.video_encoder = config.video_encoder
        self.num_frames = config.num_frames
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=config.scan if config.scan is not None else not config.video_encoder,
                dtype_mm=config.dtype,
                num_frames=config.num_frames if config.video_encoder else 1,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        if config.event_tracking:
            # Attribute name stays "phase_head": it's the checkpoint's actual parameter-tree
            # path (nnx.split/merge key), and weight_loaders.py's missing_regex references it
            # by this exact string. Renaming it would need param-tree remapping at restore
            # time, unlike the class name / file name / method name below, which are just
            # Python identifiers with no effect on checkpoint structure.
            self.phase_head = EventHead(
                in_features=paligemma_config.width, num_events=EVENT_LOGITS_NUM_CLASSES, rngs=rngs
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True
        self.action_samples = config.action_samples

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            if obs.pre_encoded_images is not None and name in obs.pre_encoded_images:
                # Hidden-state-cached path: tokens already computed by VideoEncoderCached + head projection.
                image_tokens = obs.pre_encoded_images[name]  # (B, L, D) in LLM width
                image_mask = jnp.ones(image_tokens.shape[:1], dtype=jnp.bool_)
            else:
                raw_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
                if self.video_encoder:
                    # raw_tokens: (B*T, L, D) → select current (last) frame → (B, L, D)
                    # Temporal attention enriches the current frame's tokens with context
                    # from past frames; we forward only the current frame's output to Gemma.
                    BT, L, D = raw_tokens.shape
                    B = BT // self.num_frames
                    reshaped = jnp.reshape(raw_tokens, (B, self.num_frames, L, D))
                    image_tokens = reshaped[:, -1, :, :]  # (B, L, D)
                    image_mask = obs.image_masks[name][:, -1]  # (B,) — current frame slot
                else:
                    image_tokens = raw_tokens
                    image_mask = obs.image_masks[name]

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    image_mask,
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return self._action_mse(v_t, u_t)

    def _action_mse(self, v_t, u_t) -> at.Float[at.Array, "*b ah"]:
        """Per-timestep action loss: mean squared error over the action dims."""
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def compute_loss_event(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, labels: _model.Labels, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """Like `compute_loss`, plus the auxiliary event-classification loss (only
        called when `event_tracking=True`; see training/data_loader.py for where
        `labels` comes from). Returns `(total_loss, {"action_loss", "event_loss"})`;
        `total_loss = action_loss + 0.1 * event_loss` — the 0.1 weight and the
        `EVENT_LABEL_IGNORE_TARGET_CLASS`/downweighting below are hardcoded here (not
        config knobs) so the objective can't drift between experiment configs.
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # prefix_out is shape b,s,d

        prefix_positions = positions[:, :prefix_tokens.shape[1]]
        last_valid_idx = jnp.argmax(prefix_positions, axis=1)

        last_tokens = jnp.take_along_axis(
            prefix_out, 
            last_valid_idx[:, None, None], 
            axis=1
        ).squeeze(1)

        event_logits = self.phase_head(last_tokens, train=train)

        # Per-example CE: dataset label -1 maps to class EVENT_LABEL_IGNORE_TARGET_CLASS so those
        # steps are supervised (predict "ignore bucket") instead of masking loss out.
        labels_arr = jnp.asarray(labels, dtype=jnp.int32)
        labels_flat = jnp.reshape(labels_arr, (event_logits.shape[0],))
        target = jnp.where(labels_flat >= 0, labels_flat, EVENT_LABEL_IGNORE_TARGET_CLASS)
        event_ce = optax.softmax_cross_entropy_with_integer_labels(event_logits, target)
        
        weights = jnp.where(labels_flat >= 0, 1.0, 0.02) # we downweight the unknown class since it is so frequent in the dataset
        weighted_event_ce = event_ce * weights
        
        event_mean = jnp.mean(weighted_event_ce)

        action_loss = self._action_mse(v_t, u_t)
        event_loss = 0.1 * event_mean

        return action_loss + event_loss, {"action_loss": jnp.mean(action_loss), "event_loss": event_loss}

    def _sample_actions_core(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, _model.Labels | None]:
        """Flow-matching sampling loop shared by `sample_actions` and `sample_actions_event`.

        `sample_actions` (required by BaseModel; used by pi0_fast/pytorch callers and
        tests) returns just the actions. `sample_actions_event` (used by Policy when
        event_tracking is enabled) additionally returns the current-step event
        logits, so both wrap this one implementation instead of duplicating it.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if self.event_tracking:
            prefix_positions = positions[:, :prefix_tokens.shape[1]]
            last_valid_idx = jnp.argmax(prefix_positions, axis=1)

            current_token = jnp.take_along_axis(
                prefix_out,
                last_valid_idx[:, None, None],
                axis=1
            ).squeeze(1)

            current_logits = jax.nn.softmax(self.phase_head(current_token, train=False), axis=-1)
        else:
            current_logits = None

        if self.action_samples > 1:
            # Tile the KV cache so the suffix tokens can attend to it N times
            kv_cache = jax.tree_util.tree_map(
                # Gemma KV cache layout is [layers, batch, time, heads, dim], so repeat batch axis.
                lambda x: jnp.repeat(x, self.action_samples, axis=1),
                kv_cache
            )
            # Tile masks and observations for the suffix loop
            observation = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, self.action_samples, axis=0),
                observation
            )
            prefix_mask = jnp.repeat(prefix_mask, self.action_samples, axis=0)

        batch_size = observation.state.shape[0]

        if noise is None:
            rng_samples = jax.random.split(rng, batch_size)
            noise = jax.vmap(
                lambda k: jax.random.normal(k, (self.action_horizon, self.action_dim))
            )(rng_samples)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))

        return x_0, current_logits

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        actions, _ = self._sample_actions_core(rng, observation, num_steps=num_steps, noise=noise)
        return actions

    @override
    def sample_actions_event(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, _model.Labels | None]:
        return self._sample_actions_core(rng, observation, num_steps=num_steps, noise=noise)

    @override
    def sample_text(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation, 
        *,
        history_len: int = 20,
    ) -> _model.Text:  
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        
        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        (prefix_out, _), cache = self.PaliGemma.llm(
            [prefix_tokens, None], 
            mask=prefix_attn_mask, 
            positions=positions
        )

        prefix_len = jnp.sum(prefix_mask, axis=-1).astype(int)
        prompt_length = jnp.sum(observation.tokenized_prompt_mask, axis=-1)[0].astype(int)

        all_logits = self.PaliGemma.llm(prefix_out[0], method="decode_to_logits")
        first_token_logits = all_logits[-200+prompt_length-1:-200+prompt_length, :] # will change this

        last_token = jnp.argmax(first_token_logits, axis=-1)

        token_history = jnp.zeros((batch_size, history_len), dtype=jnp.int32)
        token_history = token_history.at[:, 0].set(last_token)
    
        step = 1 # start at step 1 since we already ran one forward pass

        def text_step(carry):
            step, last_token, cache, history = carry

            pos_input = jnp.array([[prefix_len[0] + step - 1]], dtype=jnp.int32)

            token_input = jnp.reshape(last_token, (1,1))
            embedded_token = self.PaliGemma.llm(token_input, method="embed")
            extended_mask = prefix_mask  # (1, prefix_len) with True for valid, False for padding
        
            # Add mask for all generated tokens (all True since they're all valid)
            # Shape: (batch, step)
            generated_mask = jnp.ones((batch_size, step), dtype=jnp.bool_)
            
            # Concatenate to get full mask: prefix + generated tokens
            # Shape: (batch, cache_len + step)
            full_mask = jnp.concatenate([extended_mask, generated_mask], axis=1)
            
            # Reshape for attention: (batch, 1, cache_len + step)
            # The new token (query) can attend to positions where mask is True
            mask = full_mask[:, None, :]

            (out, _), new_cache = self.PaliGemma.llm(
                [embedded_token, None], 
                mask=mask,
                positions=pos_input,
                kv_cache=cache
            )
            logits = self.PaliGemma.llm(out[0], method="decode_to_logits")
            next_token = jnp.argmax(logits[0], axis=-1)
            
            # Update history
            new_history = history.at[:, step].set(next_token) 

            tokenizer = PaligemmaTokenizer()
            tokens_list = next_token.tolist()
            decoded_text = tokenizer._tokenizer.decode(tokens_list)
            
            # we print tokens one-by-one since it takes so long
            print(f"{decoded_text}", end="", flush=True)

            return (step + 1, next_token, new_cache, new_history)

        # can't use a jax loop with dynamic mask arrays
        while step < history_len:
            step, last_token, cache, token_history = text_step(
                (step, last_token, cache, token_history)
            )

        return token_history
