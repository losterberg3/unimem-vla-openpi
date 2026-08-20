import jax
import jax.nn as jnn
from flax import nnx


class EventHead(nnx.Module):
    """MLP classifier on a pooled prefix representation for event labels."""

    def __init__(
        self,
        *,
        in_features: int,
        hidden_dim: int = 512,
        num_events: int = 7,
        dropout_rate: float = 0.1,
        rngs: nnx.Rngs,
    ) -> None:
        self.fc1 = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.fc_out = nnx.Linear(hidden_dim, num_events, rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = self.fc1(x)
        x = self.norm(x)
        x = jnn.gelu(x)
        x = self.fc2(x)
        x = jnn.gelu(x)
        x = self.dropout(x, deterministic=not train)
        return self.fc_out(x)
