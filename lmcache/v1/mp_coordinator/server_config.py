# SPDX-License-Identifier: Apache-2.0
"""Per-MP-server memory capacity, as declared by each server.

Cache events report bytes held, never bytes holdable, so servers declare
capacity separately and this registry stores it. Storage only -- no event
handling, no pressure math, no liveness.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
import threading

# First Party
from lmcache.v1.distributed.api import Tier

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

    Both writers replace a declaration wholesale, so a dropped L2 adapter's
    capacity cannot linger. :meth:`declare` wins unconditionally
    (registration); :meth:`update` is revision-guarded (capacity events).
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._lock = threading.Lock()
        self._by_instance: dict[str, tuple[ModuleCapacity, ...]] = {}
        self._revisions: dict[str, int] = {}

    def declare(
        self, instance_id: str, modules: Sequence[ModuleCapacity], revision: int = 0
    ) -> None:
        """Record ``instance_id``'s capacities from a registration.

        Overwrites any stored revision: a registration means a (re)started
        process, whose counter may have gone backwards.

        Args:
            instance_id: The declaring server's id. Non-empty.
            modules: Its compartments. Empty means "declared nothing", which
                reads back as unknown capacity rather than zero.
            revision: The revision ``modules`` was taken at.

        Raises:
            ValueError: If ``instance_id`` is empty, or two entries share a
                ``(tier, backend)``.
        """
        self._validate(instance_id, modules)
        with self._lock:
            self._by_instance[instance_id] = tuple(modules)
            self._revisions[instance_id] = revision

    def update(
        self, instance_id: str, modules: Sequence[ModuleCapacity], revision: int
    ) -> bool:
        """Apply a capacity report unless it is older than what is stored.

        Args:
            instance_id: The declaring server's id. Non-empty.
            modules: Its current compartments, replacing any prior set.
            revision: The revision ``modules`` was taken at.

        Returns:
            ``True`` when applied, ``False`` when already superseded.

        Raises:
            ValueError: If ``instance_id`` is empty, or two entries share a
                ``(tier, backend)``.
        """
        self._validate(instance_id, modules)
        with self._lock:
            if revision <= self._revisions.get(instance_id, -1):
                return False
            self._by_instance[instance_id] = tuple(modules)
            self._revisions[instance_id] = revision
            return True

    @staticmethod
    def _validate(instance_id: str, modules: Sequence[ModuleCapacity]) -> None:
        """Reject an empty id, or two entries for one compartment.

        Raises:
            ValueError: On either.
        """
        if not instance_id:
            raise ValueError("instance_id must be non-empty")
        seen: set[tuple[Tier, str]] = set()
        for module in modules:
            identity = (module.tier, module.backend)
            if identity in seen:
                raise ValueError(
                    f"duplicate capacity declaration for "
                    f"{module.tier.value}/{module.backend}"
                )
            seen.add(identity)

    def get(self, instance_id: str) -> tuple[ModuleCapacity, ...]:
        """Return ``instance_id``'s declared compartments.

        Args:
            instance_id: The server to look up.

        Returns:
            Its declarations; empty when unknown or nothing was declared.
        """
        with self._lock:
            return self._by_instance.get(instance_id, ())

    def get_all(self) -> dict[str, tuple[ModuleCapacity, ...]]:
        """Return a snapshot of every server's declarations.

        Returns:
            A copy mapping ``instance_id`` to its compartments.
        """
        with self._lock:
            return dict(self._by_instance)

    def forget(self, instance_id: str) -> None:
        """Drop ``instance_id``'s declaration. Idempotent.

        Args:
            instance_id: The departed server.
        """
        with self._lock:
            self._by_instance.pop(instance_id, None)
            self._revisions.pop(instance_id, None)
