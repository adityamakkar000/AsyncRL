import jax

mesh = jax.make_mesh(axis_shapes=(1,), axis_names=("x",), axis_types=(jax.sharding.AxisType.Explicit,))

breakpoint()
