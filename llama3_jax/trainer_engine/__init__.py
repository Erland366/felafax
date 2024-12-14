import jax
from . import tree_util, monitoring

jax.sharding.NamedSharding = jax.sharding.MeshPspecSharding
jax.tree_util.tree_map_with_path = tree_util.tree_map_with_path
jax.tree_util.SequenceKey = tree_util.SequenceKey
jax.tree_util.DictKey = tree_util.DictKey
jax.tree_util.GetAttrKey = tree_util.GetAttrKey
jax.tree_util.FlattenedIndexKey = tree_util.FlattenedIndexKey
jax.sharding.Mesh = jax.experimental.maps.Mesh
jax.sharding.PartitionSpec = jax.experimental.PartitionSpec
jax.monitoring = monitoring