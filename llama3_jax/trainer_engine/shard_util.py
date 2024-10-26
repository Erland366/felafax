from __future__ import annotations

import functools
from typing import Mapping, Sequence

import numpy as np
from jax._src import mesh as mesh_lib
from jax._src import sharding, sharding_specs, tree_util
from jax._src.lib import xla_client as xc
from jax._src.util import safe_zip, use_cpp_class, use_cpp_method
from jax.experimental import ParsedPartitionSpec, PartitionSpec

Shape = tuple[int, ...]
MeshAxisName = str
Device = xc.Device


@util.cache(max_size=4096, trace_context_in_key=False)
def pmap_sharding_devices_indices_map(
    self, global_shape: Shape
) -> Mapping[Device, Index]:
    self.shard_shape(global_shape)  # raises a good error message
    indices = sharding_specs.spec_to_indices(global_shape, self.sharding_spec)
    return dict(safe_zip(self.devices.flat, indices))  # type: ignore[arg-type]


@use_cpp_class(xc.PmapSharding)
class PmapSharding(sharding.Sharding):
    """Describes a sharding used by :func:`jax.pmap`."""

    devices: np.ndarray
    sharding_spec: sharding_specs.ShardingSpec
    _internal_device_list: xc.DeviceList

    @use_cpp_method()
    def __init__(
        self,
        devices: Sequence[Device] | np.ndarray,  # type: ignore
        sharding_spec: sharding_specs.ShardingSpec,
    ):
        self.devices = np.asarray(devices)
        # The sharding spec should be pmap's sharding spec.
        self.sharding_spec = sharding_spec

    def __reduce__(self):
        return (
            type(self),
            (self.devices, self.sharding_spec),
            {"memory_kind": self.memory_kind},
        )

    def __eq__(self, other):
        if not isinstance(other, PmapSharding):
            return False
        if self is other:
            return True
        return (
            self.sharding_spec == other.sharding_spec
            and self.devices.shape == other.devices.shape
            and self._internal_device_list == other._internal_device_list
        )

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash((self._internal_device_list, self.sharding_spec))
        return self._hash

    def __str__(self):
        device_ids = [d.id for d in self.devices.flat]
        return (
            f"PmapSharding(sharding_spec={self.sharding_spec}, "
            f"{device_ids=}, "
            f"device_platform={self.devices.flat[0].platform.upper()}, "
            f"device_shape={self.devices.shape})"
        )

    def __repr__(self):
        return (
            f"PmapSharding(sharding_spec={self.sharding_spec}, "
            f"devices={self.devices})"
        )

    def is_equivalent_to(
        self: PmapSharding,
        other: PmapSharding,  # type: ignore
        ndim: int,
    ) -> bool:
        return self == other

    # TODO(yashkatariya): Expose `sharded_dim_size` in the API if required.
    @classmethod
    def default(
        cls,
        shape: Shape,
        sharded_dim: int = 0,
        devices: Sequence[xc.Device] | None = None,
    ) -> PmapSharding:
        """Creates a :class:`PmapSharding` which matches the default placement
        used by :func:`jax.pmap`.

        Args:
          shape: The shape of the input array.
          sharded_dim: Dimension the input array is sharded on. Defaults to 0.
          devices: Optional sequence of devices to use. If omitted, the implicit
          device order used by pmap is used, which is the order of
            :func:`jax.local_devices`.
        """
        # The dtype doesn't matter here. Its only used for creating the
        # sharding_spec.
        sharding_spec = sharding_specs.create_pmap_sharding_spec(
            tuple(shape), sharded_dim
        )

        num_ways_sharded = None
        for s in sharding_spec.sharding:
            if isinstance(s, sharding_specs.Unstacked):
                assert num_ways_sharded is None
                num_ways_sharded = s.size
            elif isinstance(s, sharding_specs.Chunked):
                assert num_ways_sharded is None
                if len(s.chunks) == 1:
                    num_ways_sharded = s.chunks[0]
                else:
                    raise NotImplementedError(
                        "Multiple chunks in Chunked dimension not supported."
                    )

        if num_ways_sharded is None:
            raise NotImplementedError(
                "`None` to sharded_dim is not supported. Please file a jax "
                "issue if you need this feature."
            )

        if devices is None:
            pmap_devices: np.ndarray = np.array(
                xla_bridge.local_devices()[:num_ways_sharded]
            )
        else:
            pmap_devices = np.array(devices)
        return cls(pmap_devices, sharding_spec)

    @property
    def num_devices(self) -> int:
        return len(self.device_set)

    @functools.cached_property
    def device_set(self) -> set[Device]:
        return set(self.devices.flat)

    def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
        return pmap_sharding_devices_indices_map(self, global_shape)

    @functools.cached_property
    def _device_assignment(self) -> XLADeviceAssignment:
        return tuple(self.devices.flat)

    @property
    def memory_kind(self) -> str | None:
        try:
            return self._internal_device_list.default_memory_kind
        except:
            return None

    def with_memory_kind(self, kind: str):
        raise NotImplementedError("pmap does not support memories.")

    def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
        raise NotImplementedError("pmap doesn't use OpSharding.")

    def _to_sdy_sharding(self, num_dimensions: int) -> sharding.SdyArraySharding:
        raise NotImplementedError("pmap doesn't use SdyArraySharding.")

    @functools.cached_property
    def is_fully_replicated(self) -> bool:
        for s in self.sharding_spec.sharding:
            if isinstance(s, (sharding_specs.Unstacked, sharding_specs.Chunked)):
                return False
        return True

    @functools.cached_property
    def is_fully_addressable(self) -> bool:
        return self._internal_device_list.is_fully_addressable

    def shard_shape(self, global_shape: Shape) -> Shape:
        sharded_dim = None
        sharded_dim_size = None
        for i, s in enumerate(self.sharding_spec.sharding):
            if isinstance(s, sharding_specs.Unstacked):
                sharded_dim = i
                sharded_dim_size = s.size
                sharded_shape = util.tuple_delete(global_shape, sharded_dim)
                break
            elif isinstance(s, sharding_specs.Chunked):
                sharded_dim = i
                assert len(s.chunks) == 1, s.chunks
                sharded_dim_size = s.chunks[0]
                sharded_shape = util.tuple_update(global_shape, sharded_dim, 1)
                break
        if sharded_dim is None:
            return global_shape
        if global_shape[sharded_dim] != sharded_dim_size:
            raise ValueError(
                f"The sharded dimension must be equal to the number of "
                f"devices passed to PmapSharding. Got sharded dimension {sharded_dim} "
                f"with value {global_shape[sharded_dim]} in shape {global_shape} and "
                f"the number of devices={len(self._device_assignment)}"
            )
        return sharded_shape


class AUTO:
    def __init__(self, mesh: mesh_lib.Mesh) -> None:
        self.mesh = mesh


class UnspecifiedValue:
    def __repr__(self):
        return "UnspecifiedValue"


def is_auto(x):
    return isinstance(x, AUTO)


def is_unspecified(x):
    return isinstance(x, UnspecifiedValue)


def is_unspecified_or_auto(x):
    return is_auto(x) or is_unspecified(x)


def prepare_axis_resources(axis_resources, arg_name, allow_unconstrained_dims=False):
    # PyTrees don't treat None values as leaves, so we use an is_leaf function.
    entries, treedef = tree_util.tree_flatten(
        axis_resources, is_leaf=lambda x: x is None
    )
    what = f"{arg_name} leaf specifications"

    new_entries = []
    for entry in entries:
        if is_unspecified_or_auto(entry) or entry is None:
            new_entries.append(entry)
        elif isinstance(entry, sharding.Sharding):
            if isinstance(entry, PmapSharding):
                raise ValueError(
                    f"One of {what} got sharding {entry} which is not " "allowed."
                )
            new_entries.append(entry)
        else:
            new_entries.append(
                ParsedPartitionSpec.from_user_input(
                    entry, what, allow_unconstrained_dims=allow_unconstrained_dims
                )
            )

    _check_unique_resources(new_entries, arg_name)
    return tree_util.tree_unflatten(treedef, new_entries)


def preprocess(mesh, spec, parsed_pspec, _manual_axes=frozenset()):
    if parsed_pspec is None:
        parsed_pspec = prepare_axis_resources(
            PartitionSpec() if spec is None else spec,
            "NamedSharding spec",
            allow_unconstrained_dims=True,
        )
    _check_mesh_resource_axis(mesh, parsed_pspec, _manual_axes)
    return parsed_pspec


# Copied from JAX main -> Accessed at October 5th
@use_cpp_class(xc.NamedSharding)
class NamedSharding(sharding.Sharding):
    r"""A :class:`NamedSharding` expresses sharding using named axes.

    A :class:`NamedSharding` is a pair of a :class:`Mesh` of devices and
    :class:`PartitionSpec` which describes how to shard an array across that
    mesh.

    A :class:`Mesh` is a multidimensional NumPy array of JAX devices,
    where each axis of the mesh has a name, e.g. ``'x'`` or ``'y'``.

    A :class:`PartitionSpec` is a tuple, whose elements can be a ``None``,
    a mesh axis, or a tuple of mesh axes. Each element describes how an input
    dimension is partitioned across zero or more mesh dimensions. For example,
    ``PartitionSpec('x', 'y')`` says that the first dimension of data
    is sharded across ``x`` axis of the mesh, and the second dimension is sharded
    across ``y`` axis of the mesh.

    The Distributed arrays and automatic parallelization
    (https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#namedsharding-gives-a-way-to-express-shardings-with-names)
    tutorial has more details and diagrams that explain how
    :class:`Mesh` and :class:`PartitionSpec` are used.

    Args:
      mesh: A :class:`jax.sharding.Mesh` object.
      spec: A :class:`jax.sharding.PartitionSpec` object.

    Examples:

      >>> from jax.sharding import Mesh
      >>> from jax.sharding import PartitionSpec as P
      >>> mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
      >>> spec = P('x', 'y')
      >>> named_sharding = jax.sharding.NamedSharding(mesh, spec)
    """

    mesh: mesh_lib.Mesh | mesh_lib.AbstractMesh
    spec: PS
    _memory_kind: str | None
    _parsed_pspec: ParsedPartitionSpec
    _manual_axes: frozenset[MeshAxisName]

    @use_cpp_method()
    def __init__(
        self,
        mesh: mesh_lib.Mesh | mesh_lib.AbstractMesh,
        spec: PS,
        *,
        memory_kind: str | None = None,
        _parsed_pspec=None,
        _manual_axes=frozenset(),
    ):
        self.mesh = mesh
        self.spec = spec
        self._memory_kind = memory_kind
        self._manual_axes = _manual_axes
        self._parsed_pspec = preprocess(self.mesh, self.spec, _parsed_pspec)

    def __repr__(self):
        mesh_repr = ", ".join(f"'{k}': {v}" for k, v in self.mesh.shape.items())
        mem = "" if self.memory_kind is None else f", memory_kind={self.memory_kind}"
        return f"NamedSharding(mesh=Mesh({mesh_repr}), spec={self.spec}{mem})"

    def __reduce__(self):
        return (
            type(self),
            (self.mesh, self.spec),
            {"memory_kind": self.memory_kind, "_manual_axes": self._manual_axes},
        )

    @property
    def memory_kind(self) -> str | None:
        return self._memory_kind

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash(
                (self.mesh, self.memory_kind, self._parsed_pspec, self._manual_axes)
            )
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, NamedSharding):
            return False
        if self is other:
            return True
        if (
            self._parsed_pspec != other._parsed_pspec
            or self.memory_kind != other.memory_kind
            or self._manual_axes != other._manual_axes
        ):
            return False
        return self.mesh is other.mesh or self.mesh == other.mesh

    def check_compatible_aval(self, aval_shape: Shape) -> None:
        assert self._parsed_pspec is not None
        if len(aval_shape) < len(self._parsed_pspec):
            extra_msg = (
                " For scalars the PartitionSpec should be P()"
                if len(aval_shape) == 0
                else ""
            )
            raise ValueError(
                f"Sharding {self} is only valid for values of rank at least "
                f"{len(self._parsed_pspec)}, but was applied to a value of rank "
                f"{len(aval_shape)}.{extra_msg}"
            )

    @classmethod
    def _from_parsed_pspec(
        cls, mesh, parsed_pspec, *, memory_kind=None, _manual_axes=frozenset()
    ):
        return cls(
            mesh,
            parsed_pspec.get_partition_spec(),
            memory_kind=memory_kind,
            _parsed_pspec=parsed_pspec,
            _manual_axes=_manual_axes,
        )

    @property
    def num_devices(self) -> int:
        return self.mesh.size

    @property
    def device_set(self) -> set[Device]:
        if isinstance(self.mesh, mesh_lib.AbstractMesh):
            raise ValueError(
                "device_set is not implemented for `jax.sharding.AbstractMesh`."
            )
        return self.mesh._flat_devices_set

    @property
    def _device_assignment(self) -> XLADeviceAssignment:
        if isinstance(self.mesh, mesh_lib.AbstractMesh):
            raise ValueError(
                "_device_assignment is not implemented for"
                " `jax.sharding.AbstractMesh`."
            )
        return self.mesh._flat_devices_tuple

    @property
    def is_fully_addressable(self) -> bool:
        if isinstance(self.mesh, mesh_lib.AbstractMesh):
            raise ValueError(
                "is_fully_addressable is not implemented for "
                "`jax.sharding.AbstractMesh`."
            )
        # Speed up `is_fully_addressable` since there is a high chance that the
        # mesh across multiple NamedSharding objects will be the same.
        return not self.mesh.is_multi_process

    @property
    def addressable_devices(self) -> set[Device]:
        if isinstance(self.mesh, mesh_lib.AbstractMesh):
            raise ValueError(
                "addressable_devices is not implemented for "
                "`jax.sharding.AbstractMesh`."
            )
        # Override addressable devices because there is a high chance that the mesh
        # across multiple NamedSharding objects will be the same.
        return self.mesh._local_devices_set

    @functools.cached_property
    def is_fully_replicated(self) -> bool:
        if self.mesh.size == 1:
            return True
        array_mapping = cast(ParsedPartitionSpec, get_array_mapping(self._parsed_pspec))
        mesh_shape = self.mesh.shape
        num_partitions = 1
        for name in array_mapping:
            num_partitions *= mesh_shape[name]
        return num_partitions == 1

    def with_memory_kind(self, kind: str) -> NamedSharding:
        return NamedSharding(self.mesh, self.spec, memory_kind=kind)

    def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
        return named_sharding_to_xla_hlo_sharding(self, num_dimensions)

    def _to_sdy_sharding(self, num_dimensions: int) -> sharding.SdyArraySharding:
        dim_shardings = [
            sharding.SdyDimSharding(axes=[], is_closed=True)
            for _ in range(num_dimensions)
        ]
        for i, dim_spec in enumerate(self._parsed_pspec):
            if dim_spec is None:
                dim_shardings[i].is_closed = False
            elif not dim_spec:
                # Already empty and closed sharding.
                pass
            else:
                dim_shardings[i].axes = dim_spec
        return sharding.SdyArraySharding("mesh", dim_shardings)
