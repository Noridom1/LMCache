# SPDX-License-Identifier: Apache-2.0
"""Per-MP-server memory capacity, declared by ``config`` cache events.

Cache events report bytes held, never bytes holdable, so servers declare
capacity separately and this registry stores it. It is a
:class:`~lmcache.v1.mp_coordinator.ingest.event_broadcaster.CacheEventConsumer`
like the key directory and the usage manager, so declarations arrive through
the same gate -- inheriting its incarnation fencing, dedup, and ordering
instead of carrying a second mechanism alongside.

One declaration is one ``config`` batch per compartment, all sharing a
``capacity_revision``. A batch whose revision is newer than what is stored
starts a fresh set; batches at the same revision extend it. That is what
retires a compartment: a declaration that omits it simply never adds it.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.api import CacheEventBatch, CacheEventType

logger = init_logger(__name__)

# "No cap declared". Real caps are positive, so an unlimited adapter
# reporting 0 would otherwise read as permanently full.
UNDECLARED_CAPACITY = 0


@dataclass(frozen=True)
class ModuleCapacity:
    """One compartment's declared capacity: the L1 pool or one L2 adapter.

    Keyed on the same ``(tier, backend)`` axis cache events use.

    Attributes:
        tier: ``L1`` or ``L2``; never ``ALL``.
        backend: Backend within the tier (``"dram"``, ``"fs"``, ...).
            Non-empty.
        capacity_bytes: Declared bytes, or :data:`UNDECLARED_CAPACITY`.
        shared: Set when instances mount one storage domain (an S3 bucket,
            a CXL pool). Count once for the fleet, never per mount.
    """

    tier: Tier
    backend: str
    capacity_bytes: int
    shared: bool = False

    def __post_init__(self) -> None:
        """Validate the declaration.

        Raises:
            ValueError: If ``backend`` is empty, ``capacity_bytes`` is
                negative, or ``tier`` is not concrete.
        """
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if self.capacity_bytes < 0:
            raise ValueError(f"capacity_bytes must be >= 0 (got {self.capacity_bytes})")
        if self.tier not in (Tier.L1, Tier.L2):
            raise ValueError(
                f"capacity must target a concrete tier (got {self.tier.value!r})"
            )


class ServerConfigRegistry:
    """Thread-safe store of each MP server's declared capacities.

    A :class:`CacheEventConsumer`: :meth:`consume` accumulates ``config``
    batches into one declaration per ``(incarnation, capacity_revision)``,
    so a compartment the newest declaration omits stops being reported.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._lock = threading.Lock()
        self._by_instance: dict[str, dict[tuple[Tier, str], ModuleCapacity]] = {}
        self._stamps: dict[str, tuple[int, int]] = {}

    def consume(self, batch: CacheEventBatch) -> None:
        """Apply one ``config`` batch; ignore every other event type.

        The gate has already dropped stale incarnations and duplicates, so
        the only ordering left to do is grouping batches into declarations:
        a newer ``(incarnation, capacity_revision)`` starts a fresh set, an
        equal one extends it, an older one is a straggler and is dropped.

        Args:
            batch: A gate-admitted batch. Non-``config`` batches no-op.
        """
        if batch.event_type != CacheEventType.CONFIG:
            return
        module = ModuleCapacity(
            tier=batch.tier,
            backend=batch.backend,
            capacity_bytes=batch.capacity_bytes,
            shared=batch.shared,
        )
        stamp = (batch.incarnation, batch.capacity_revision)
        with self._lock:
            stored = self._stamps.get(batch.instance_id, (-1, -1))
            if stamp < stored:
                return
            if stamp > stored:
                # A new declaration: drop the previous set so compartments
                # it no longer lists stop being reported.
                self._by_instance[batch.instance_id] = {}
                self._stamps[batch.instance_id] = stamp
            self._by_instance[batch.instance_id][(module.tier, module.backend)] = module

    def fence_instance(self, instance_id: str) -> None:
        """No-op: capacity is configuration, not reported L1 state.

        A restarting process re-declares under a higher incarnation, which
        :meth:`consume` supersedes on its own. A departing one has its
        declaration dropped by :meth:`forget`.

        Args:
            instance_id: The instance whose reported L1 state is void.
        """

    def get(self, instance_id: str) -> tuple[ModuleCapacity, ...]:
        """Return ``instance_id``'s declared compartments.

        Args:
            instance_id: The server to look up.

        Returns:
            Its declarations; empty when unknown or nothing was declared.
        """
        with self._lock:
            return tuple(self._by_instance.get(instance_id, {}).values())

    def get_all(self) -> dict[str, tuple[ModuleCapacity, ...]]:
        """Return a snapshot of every server's declarations.

        Returns:
            A copy mapping ``instance_id`` to its compartments.
        """
        with self._lock:
            return {
                instance_id: tuple(modules.values())
                for instance_id, modules in self._by_instance.items()
            }

    def forget(self, instance_id: str) -> None:
        """Drop ``instance_id``'s declaration. Idempotent.

        Args:
            instance_id: The departed server.
        """
        with self._lock:
            self._by_instance.pop(instance_id, None)
            self._stamps.pop(instance_id, None)
