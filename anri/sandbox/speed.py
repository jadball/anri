import jax
import jax.numpy as jnp


@jax.jit
def batch_invert(matrices):
    """
    Inverts a batch of matrices using JAX, optimized for GPU execution.

    Args:
        matrices (jnp.ndarray): A batch of square matrices to be inverted.
                                Shape: (batch_size, n, n)

    Returns:
        jnp.ndarray: The batch of inverted matrices.
                     Shape: (batch_size, n, n)
    """
    return jax.vmap(jnp.linalg.inv)(matrices)


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Create a batch of random 3x3 matrices
    batch_size = 10000
    matrices = np.random.rand(batch_size, 3, 3)

    # Convert to JAX array
    jax_matrices = jnp.array(matrices)

    # Invert the batch of matrices on GPU
    inverted_matrices = batch_invert(jax_matrices)

    print("Original Matrices:")
    print(matrices)
    print("\nInverted Matrices:")
    print(inverted_matrices)

    # assert that jax is using a GPU:
    assert str(jax.devices()[0].platform) == "gpu", str(jax.devices()[0].platform)
