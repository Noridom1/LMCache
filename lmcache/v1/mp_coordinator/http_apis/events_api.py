# SPDX-License-Identifier: Apache-2.0
"""Cache-event ingest endpoint on the coordinator (fleet-level).

The ``/events`` surface, thin over the :class:`EventGate`. Top-level
rather than under ``/directory`` because the stream feeds every consumer
of it, not one of them. See
``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Third Party
from fastapi import APIRouter, Request

# First Party
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.ingest.event_gate import IngestResult
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest, CacheEventsResponse
from lmcache.v1.mp_coordinator.server_config import ModuleCapacity

router = APIRouter()


@router.post("/events")
async def report_cache_events(
    body: CacheEventsRequest, request: Request
) -> CacheEventsResponse:
    """Offer a list of cache-event batches to the ingest gate.

    Batches are offered in list order; per instance they must be sent in
    emission order. Duplicates and stale incarnations are dropped and
    counted, not errors.

    Any capacity reports are applied first: they are whole declarations
    guarded by a revision, so a superseded one is ignored rather than
    regressing the topology.

    Args:
        body: The event batches and capacity reports to ingest.

    Returns:
        Counts of applied and dropped batches.
    """
    ctx = get_context(request)
    for report in body.capacity_reports:
        ctx.server_config.update(
            report.instance_id,
            [
                ModuleCapacity(
                    tier=m.tier,
                    backend=m.backend,
                    capacity_bytes=m.capacity_bytes,
                    shared=m.shared,
                )
                for m in report.modules
            ],
            report.incarnation,
            report.revision,
        )
    event_gate = ctx.event_gate
    response = CacheEventsResponse()
    for batch in body.batches:
        result = event_gate.ingest(batch)
        if result == IngestResult.ADMITTED:
            response.applied += 1
        elif result == IngestResult.DUPLICATE:
            response.duplicates += 1
        else:
            response.stale += 1
    return response
