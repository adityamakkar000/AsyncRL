import jax

jax.config.update("jax_default_matmul_precision", "highest")
jax.numpy.set_printoptions(precision=9)
