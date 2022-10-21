import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.random as jrandom
from jax.nn import softmax
from jax.scipy.special import logsumexp
import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], device_type='GPU')
import time

def MLP_Mean_Field_Init(sizes, key):
    """
    Initializes layers with the mean field regime,
    i.e O(1/sqrt(m)). Uses Kaiming He constant
    in the uniform distribution
    """
    no_layers = len(sizes) - 1
    scale = lambda x1, x2 : jnp.sqrt(6 / (x1 + x2))
    keys = jrandom.split(key, no_layers)
    params = [jrandom.uniform(key, shape=(n_in, n_out),
                              dtype=jnp.float32,
                              minval=-scale(n_in, n_out),
                              maxval=scale(n_in, n_out)) \
                              for n_in, n_out, key in zip(sizes[:-1], sizes[1:], keys)]
    return params


def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)


def relu(x):
  return jnp.maximum(0.01 * x, x)


def forward(params, x):
    """
    takes a single point x
    and does a forward pass
    """
    activations = x
    for layer in params[:-1]:
        activations = jnp.dot(activations, layer)
        activations = relu(activations)
    return jnp.dot(activations, params[-1])


# modify for batch of n points 
batched_forward = vmap(forward, in_axes=(None, 0))

# L2 loss
def L2(params, xbatch, y):
    logits = batched_forward(params, xbatch)
    return jnp.mean((logits - y) ** 2)

# Cross Entropy loss
def CE(params, xbatch, y):
    logits = batched_forward(params, xbatch)
    scores = logits - logsumexp(logits)
    return -jnp.mean(jnp.sum(scores * y, axis=1))

@jit
def L2update(params, x, y, lr):
    grads = grad(L2)(params, x, y)
    return [v - lr * dv for (v, dv) in zip(params, grads)]

@jit
def CEupdate(params, x, y, lr):
    grads = grad(CE)(params, x, y)
    return [v - lr * dv for (v, dv) in zip(params, grads)]



@jit
def eval_L2(params, X, Y):
    """
    evaluates the L2 loss,
    returns loss and predictions
    """
    logits = batched_forward(params, X)
    l2 = jnp.mean((Y - logits) ** 2)
    return l2, logits

@jit
def eval_L2_CE(params, X, Yr, Yc, Ych, midpoints):
    """
    evaluates ce entropy loss to target Ye
    then using midpoings calculates Exp value
    and returns L2 loss to regression target
    Yr = real label
    Yc = classification discrete label
    Ych = one hotted version of Yc
    """
    midpoints = midpoints.reshape(-1, 1)
    # forward pass
    logits = batched_forward(params, X)
    # ce loss
    scores = logits - logsumexp(logits)
    ce = -jnp.mean(jnp.sum(scores * Ych, axis=1))
    # convert to regression
    probs = softmax(logits, axis=-1)
    acc = jnp.mean(jnp.argmax(probs, axis=1) == Yc.flatten())

    preds = probs @ midpoints
    l2 = jnp.mean((Yr - preds) ** 2)
    return l2, ce, acc, preds

if __name__ == "__main__":
    from toydata import mst_tf_dataset
    from tqdm import tqdm
    from utils import get_bins



