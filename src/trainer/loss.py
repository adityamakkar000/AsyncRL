import jax
import jax.numpy as jnp


def cross_entropy_loss(): ...


def grpo_loss(
    logits: jnp.ndarray,
    tokens: jnp.ndarray,
    token_mask: jnp.ndarray,
    rewards: jnp.ndarray,
    *,
    ref_log_probs: jnp.ndarray | None = None,
    beta: float = 0.0,
    group_size: int | None = None,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    token_mask = token_mask.astype(logits.dtype)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_log_probs = jnp.take_along_axis(log_probs, tokens[..., None], axis=-1)[..., 0]
    token_counts = jnp.maximum(token_mask.sum(axis=-1), 1.0)
    sequence_log_probs = jnp.sum(token_log_probs * token_mask, axis=-1) / token_counts

    if group_size is None:
        rewards_mean = jnp.mean(rewards)
        rewards_std = jnp.std(rewards)
        advantages = (rewards - rewards_mean) / (rewards_std + epsilon)
    else:
        if group_size <= 0:
            raise ValueError("group_size must be positive.")
        batch_size = rewards.shape[0]
        if batch_size % group_size != 0:
            raise ValueError("Batch size must be divisible by group_size.")
        grouped_rewards = rewards.reshape((-1, group_size))
        grouped_mean = jnp.mean(grouped_rewards, axis=1, keepdims=True)
        grouped_std = jnp.std(grouped_rewards, axis=1, keepdims=True)
        advantages = (grouped_rewards - grouped_mean) / (grouped_std + epsilon)
        advantages = advantages.reshape((batch_size,))

    advantages = jax.lax.stop_gradient(advantages)
    loss = -jnp.mean(advantages * sequence_log_probs)

    if ref_log_probs is not None:
        ref_log_probs = jnp.asarray(ref_log_probs)
        token_kl = (token_log_probs - ref_log_probs) * token_mask
        sequence_kl = jnp.sum(token_kl, axis=-1) / token_counts
        loss += beta * jnp.mean(sequence_kl)

    return loss
