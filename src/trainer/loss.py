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
    """Compute the Group Relative Policy Optimization loss.

    Args:
        logits: Model logits with shape [batch, seq_len, vocab].
        tokens: Token ids with shape [batch, seq_len].
        token_mask: Mask for valid tokens with shape [batch, seq_len].
        rewards: Scalar rewards per sequence with shape [batch].
        ref_log_probs: Optional reference log-probabilities with shape [batch, seq_len].
        beta: Weight for the KL penalty when ref_log_probs is provided.
        group_size: Optional group size for relative advantage normalization.
        epsilon: Small value to avoid division by zero.

    Returns:
        Scalar GRPO loss as a JAX array.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if tokens.shape != logits.shape[:-1]:
        raise ValueError(f"tokens shape {tokens.shape} must match logits shape {logits.shape[:-1]}")
    if token_mask.shape != tokens.shape:
        raise ValueError(f"token_mask shape {token_mask.shape} must match tokens shape {tokens.shape}")
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1D, got shape {rewards.shape}")
    if rewards.shape[0] != tokens.shape[0]:
        raise ValueError(f"rewards batch size {rewards.shape[0]} must match tokens batch size {tokens.shape[0]}")
    if ref_log_probs is None and beta > 0:
        raise ValueError("ref_log_probs must be provided when beta > 0")
    if ref_log_probs is not None and beta < 0:
        raise ValueError(f"beta must be non-negative when ref_log_probs is provided, got {beta}")

    token_mask = token_mask.astype(logits.dtype)
    rewards = rewards.astype(logits.dtype)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_log_probs = jnp.take_along_axis(log_probs, tokens[..., None], axis=-1)[..., 0]
    token_counts = jnp.maximum(token_mask.sum(axis=-1), 1.0)
    sequence_log_probs = jnp.sum(token_log_probs * token_mask, axis=-1) / token_counts

    if group_size is None:
        rewards_mean = jnp.mean(rewards)
        rewards_std = jnp.std(rewards)
        advantages = (rewards - rewards_mean) / jnp.maximum(rewards_std, epsilon)
    else:
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        batch_size = rewards.shape[0]
        if batch_size % group_size != 0:
            raise ValueError(f"Batch size ({batch_size}) must be divisible by group_size ({group_size})")
        grouped_rewards = rewards.reshape((-1, group_size))
        grouped_mean = jnp.mean(grouped_rewards, axis=1, keepdims=True)
        grouped_std = jnp.std(grouped_rewards, axis=1, keepdims=True)
        advantages = (grouped_rewards - grouped_mean) / jnp.maximum(grouped_std, epsilon)
        advantages = advantages.reshape((batch_size,))

    advantages = jax.lax.stop_gradient(advantages)
    loss = -jnp.mean(advantages * sequence_log_probs)

    if ref_log_probs is not None:
        if ref_log_probs.shape != token_log_probs.shape:
            raise ValueError(
                f"ref_log_probs shape {ref_log_probs.shape} must match tokens shape {tokens.shape}"
            )
        ref_log_probs = jnp.asarray(ref_log_probs, dtype=logits.dtype)
        ref_log_probs = jax.lax.stop_gradient(ref_log_probs)
        token_kl = (token_log_probs - ref_log_probs) * token_mask
        sequence_kl = jnp.sum(token_kl, axis=-1) / token_counts
        loss += beta * jnp.mean(sequence_kl)

    return loss
