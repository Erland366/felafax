import jax
from . import tree_util

jax.sharding.NamedSharding = jax.sharding.MeshPspecSharding
jax.tree_util.tree_map_with_path = tree_util.tree_map_with_path