"""Site-only interpretation of original NZ-NCM qualified AS records.

Not a raw Site event, Kiosk delivery receipt or NCM crisis report.
Returns only Backend-recorded Site entry into a named crisis-area AS state,
plus the crisis context recorded with that exact crisis ID. No radius replay.
"""
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

CONTRACT = "signlog-admin-site-as-provenance.v1"
MAX_SITE_AS = 1000

class AsProvenanceUnavailable(RuntimeError):
    pass

class AsProvenanceOverflow(RuntimeError):
    pass


def read_site_as_provenance(conn: Connection, site_id: str) -> dict:
    required = {"ncm_crisis_site_signal_journal", "crises"}
    if not required.issubset(set(inspect(conn).get_table_names())):
        raise AsProvenanceUnavailable("Required original source tables unavailable")
    rows = conn.execute(text("""
        SELECT j.signal_id, j.site_id, j.crisis_id,
               j.recorded_at, j.source_basis, j.contract_version,
               c.name AS crisis_name, c.hazard_type, c.radius_km
        FROM ncm_crisis_site_signal_journal j
        INNER JOIN crises c ON c.crisis_id = j.crisis_id
        WHERE j.site_id = :site_id
          AND j.signal_state = 'as'
          AND j.source_basis = 'ncm_backend_as_qualification'
        ORDER BY j.recorded_at DESC, j.signal_id DESC
        LIMIT :limit
    """), {"site_id": site_id, "limit": MAX_SITE_AS + 1}).mappings().all()
    if len(rows) > MAX_SITE_AS:
        raise AsProvenanceOverflow("Do not silently truncate historical AS evidence")
    entries = []
    for r in rows:
        if not all(isinstance(r[k],str) and r[k].strip() for k in
                   ("signal_id", "site_id", "crisis_id", "recorded_at", "source_basis")):
            raise AsProvenanceUnavailable("Malformed original AS source row")
        entries.append({
            "signal_id": r["signal_id"],
            "site_id": r["site_id"],
            "crisis_id": r["crisis_id"],
            "crisis_name": r["crisis_name"],
            "hazard_type": r["hazard_type"],
            "declared_radius_km": r["radius_km"],
            "as_recorded_at": r["recorded_at"],
            "source_basis": r["source_basis"],
            "source_contract": r["contract_version"],
            "receipt_proven": False,
            "human_acknowledgement_proven": False,
        })
    return {
        "contract": CONTRACT,
        "site_id": site_id,
        "capture_status": "source_recorded",
        "crisis_metadata_basis": "current_recorded_crisis_row_at_read; NOT frozen_metadata_at_as_trigger",
        "complete": True,
        "count": len(entries),
        "entries": entries,
        "meaning": "Backend-qualified crisis-area AS, not Kiosk delivery or human receipt",
    }
