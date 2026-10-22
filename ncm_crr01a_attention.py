"""SignLog NZ-NCM CRR-01A: recorded crisis activation-time Site observations.

Evidence discipline:
- Snapshot of BACKEND-STORED Site location and status at the time of a new crisis record.
- Candidate geography is NOT a confirmed historical membership/exposure, response obligation,
  receipt by the Kiosk, SOS/FHC attribution, or rescue outcome.
- No raw event, SOS/FHC, AIA, bootstrap, or live Red/Orange/Green state is modified.
- No per-Site allowlist: enumerates registered operational Sites from the database.
- Intentionally does not populate N1A interval tables; CRR-01B may materialise only after
  the N1B evidence and policy requirements are proven.

Requires SQLAlchemy 2.x; supports existing SignLog SQLite/PostgreSQL DB style.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from math import asin, cos, radians, sin, sqrt, isfinite
from typing import Optional
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import text
from sqlalchemy.engine import Connection

CONTRACT = "ncm-crr01a.activation-context.v1"
SOURCE_BASIS = "stored_site_coordinates_observed_during_backend_crisis_creation"
PROJECTOR_AUTHORITY = "context_only_no_crisis_response_attribution"


@dataclass(frozen=True)
class CaptureSummary:
    crisis_id: str
    revision: int
    inventory_scanned: int
    inside_candidates: int
    outside_observed: int
    location_unknown: int
    observation_clock: str
    capture_status: str


def _utc_clock(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Observed clock is required")
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Observed clock must contain a timezone")
    return dt.astimezone(timezone.utc).isoformat()


def _coord(value: object, lower: float, upper: float) -> Optional[float]:
    # A boolean is not a valid geographic coordinate even though it is an int in Python.
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(number) or not lower <= number <= upper:
        return None
    return number


def _distance_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    dlat = radians(lat_b - lat_a)
    dlon = radians(lon_b - lon_a)
    a = sin(dlat / 2) ** 2 + cos(radians(lat_a)) * cos(radians(lat_b)) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(min(1.0, sqrt(max(0.0, a))))


def ensure_crr01a_schema(conn: Connection) -> None:
    """Additive schema only. Does not alter N1A membership/response/other tables."""
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ncm_crisis_site_capture_runs (
            crisis_id TEXT NOT NULL,
            scope_revision INTEGER NOT NULL CHECK (scope_revision > 0),
            observation_clock TEXT NOT NULL,
            inventory_scanned INTEGER NOT NULL,
            inside_candidates INTEGER NOT NULL,
            outside_observed INTEGER NOT NULL,
            location_unknown INTEGER NOT NULL,
            source_basis TEXT NOT NULL,
            capture_status TEXT NOT NULL,
            contract_version TEXT NOT NULL,
            PRIMARY KEY (crisis_id, scope_revision),
            FOREIGN KEY (crisis_id) REFERENCES crises(crisis_id)
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ncm_crisis_site_activation_observations (
            observation_id TEXT PRIMARY KEY,
            crisis_id TEXT NOT NULL,
            site_id TEXT NOT NULL,
            scope_revision INTEGER NOT NULL CHECK (scope_revision > 0),
            observation_clock TEXT NOT NULL,
            site_latitude REAL NOT NULL,
            site_longitude REAL NOT NULL,
            distance_km REAL NOT NULL,
            active_count_snapshot INTEGER,
            source_basis TEXT NOT NULL,
            policy_response_required INTEGER,
            kiosk_issued_at TEXT,
            kiosk_received_at TEXT,
            event_specific_association TEXT NOT NULL,
            evidence_status TEXT NOT NULL CHECK (evidence_status IN ('context_only')),
            contract_version TEXT NOT NULL,
            UNIQUE (crisis_id, site_id, scope_revision),
            FOREIGN KEY (crisis_id) REFERENCES crises(crisis_id),
            FOREIGN KEY (site_id) REFERENCES sites(site_id)
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_ncm_crr01a_observations_crisis_site
        ON ncm_crisis_site_activation_observations (crisis_id, scope_revision, site_id)
    """))


def capture_crisis_activation_observations(
    conn: Connection,
    *,
    crisis_id: str,
    revision: int,
    center_latitude: object,
    center_longitude: object,
    radius_km: object,
    observed_at: str,
    max_sites: int = 50000,
) -> CaptureSummary:
    """Record a *candidate* geography observation for a new crisis, transactionally.

    The caller invokes this within the existing new-crisis DB transaction AFTER the
    recorded N1A creation revision. Repeated invocation for the same crisis/revision
    returns the existing capture; no duplicate rows. The whole enumeration must succeed
    before a capture run is marked complete. Caller should use a DB SAVEPOINT if the
    safety-critical crisis creation must proceed when this supporting capture fails.
    """
    if not isinstance(crisis_id, str) or not crisis_id.strip():
        raise ValueError("Original crisis ID is required")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ValueError("Positive original crisis revision is required")
    if not isinstance(max_sites, int) or not 1 <= max_sites <= 200000:
        raise ValueError("Invalid bounded Site enumeration")
    clock = _utc_clock(observed_at)
    lat = _coord(center_latitude, -90, 90)
    lon = _coord(center_longitude, -180, 180)
    try:
        radius = float(radius_km) if radius_km is not None else None
    except (TypeError, ValueError, OverflowError):
        radius = None
    if radius is not None and (not isfinite(radius) or radius <= 0):
        radius = None

    present = conn.execute(text("""
        SELECT crisis_id, scope_revision, observation_clock, inventory_scanned,
               inside_candidates, outside_observed, location_unknown, capture_status
        FROM ncm_crisis_site_capture_runs
        WHERE crisis_id=:crisis_id AND scope_revision=:revision
    """), {"crisis_id": crisis_id, "revision": revision}).mappings().first()
    if present:
        return CaptureSummary(
            crisis_id=present["crisis_id"], revision=present["scope_revision"],
            observation_clock=present["observation_clock"], inventory_scanned=present["inventory_scanned"],
            inside_candidates=present["inside_candidates"], outside_observed=present["outside_observed"],
            location_unknown=present["location_unknown"], capture_status=present["capture_status"],
        )

    crisis = conn.execute(text("""
        SELECT event_revision FROM crises WHERE crisis_id=:crisis_id
    """), {"crisis_id": crisis_id}).mappings().first()
    if crisis is None or crisis["event_revision"] != revision:
        raise ValueError("Recorded crisis revision does not match captured revision")

    # One SQL statement gives a single database statement snapshot on both
    # supported engines. A multi-page READ COMMITTED sweep would NOT have a
    # provably fixed source view while Sites are changing concurrently.
    rows = conn.execute(text("""
        SELECT s.site_id, s.latitude, s.longitude, ss.active_count
        FROM sites s LEFT JOIN site_status ss ON ss.site_id=s.site_id
        WHERE s.operational_state='active'
        ORDER BY s.site_id ASC LIMIT :row_limit
    """), {"row_limit": max_sites + 1}).mappings().all()
    if len(rows) > max_sites:
        raise RuntimeError("Site capture exceeds bounded enumeration; discard incomplete transaction")

    scanned = len(rows)
    inside = outside = unknown = 0
    for row in rows:
        site_lat = _coord(row["latitude"], -90, 90)
        site_lon = _coord(row["longitude"], -180, 180)
        if lat is None or lon is None or radius is None or site_lat is None or site_lon is None:
            unknown += 1
            continue
        distance = _distance_km(lat, lon, site_lat, site_lon)
        if distance > radius:
            outside += 1
            continue
        inside += 1
        identifier = str(uuid5(NAMESPACE_URL, f"signlog/crr01a/{crisis_id}/{revision}/{row['site_id']}"))
        count = row["active_count"]
        conn.execute(text("""
            INSERT INTO ncm_crisis_site_activation_observations (
                observation_id, crisis_id, site_id, scope_revision, observation_clock,
                site_latitude, site_longitude, distance_km, active_count_snapshot,
                source_basis, policy_response_required, kiosk_issued_at,
                kiosk_received_at, event_specific_association, evidence_status,
                contract_version
            ) VALUES (
                :observation_id, :crisis_id, :site_id, :revision, :clock,
                :latitude, :longitude, :distance, :active_count, :basis,
                NULL, NULL, NULL, :association, 'context_only', :contract
            )
        """), {
            "observation_id": identifier, "crisis_id": crisis_id, "site_id": row["site_id"],
            "revision": revision, "clock": clock, "latitude": site_lat,
            "longitude": site_lon, "distance": distance,
            "active_count": int(count) if count is not None else None,
            "basis": SOURCE_BASIS, "association": PROJECTOR_AUTHORITY, "contract": CONTRACT,
        })

    conn.execute(text("""
        INSERT INTO ncm_crisis_site_capture_runs (
            crisis_id, scope_revision, observation_clock, inventory_scanned,
            inside_candidates, outside_observed, location_unknown,
            source_basis, capture_status, contract_version
        ) VALUES (:crisis_id, :revision, :clock, :scanned, :inside,
                  :outside, :unknown, :basis, 'complete_inventory_snapshot', :contract)
    """), {"crisis_id": crisis_id, "revision": revision, "clock": clock,
           "scanned": scanned, "inside": inside, "outside": outside,
           "unknown": unknown, "basis": SOURCE_BASIS, "contract": CONTRACT})
    return CaptureSummary(
        crisis_id=crisis_id, revision=revision, inventory_scanned=scanned,
        inside_candidates=inside, outside_observed=outside,
        location_unknown=unknown, observation_clock=clock,
        capture_status="complete_inventory_snapshot",
    )


def read_activation_observations(conn: Connection, crisis_id: str, *, max_rows: int = 5000) -> dict:
    '''Bounded, READ-ONLY original activation observations for one crisis.

    Missing optional schema or an unrecorded capture is not an empty, complete
    response history. No current Site state is reinterpreted or backfilled.
    '''
    from sqlalchemy import inspect
    if not isinstance(crisis_id, str) or not crisis_id.strip():
        raise ValueError('Original crisis ID required')
    if not isinstance(max_rows, int) or not 1 <= max_rows <= 5000:
        raise ValueError('Invalid observation bound')
    tables = set(inspect(conn).get_table_names())
    required = {'ncm_crisis_site_capture_runs', 'ncm_crisis_site_activation_observations'}
    if not required.issubset(tables):
        return {'capture_status': 'schema_not_installed', 'capture_runs': [], 'observations': [],
                'complete': False}
    runs = conn.execute(text('''SELECT crisis_id, scope_revision, observation_clock,
        inventory_scanned, inside_candidates, outside_observed, location_unknown,
        source_basis, capture_status, contract_version
        FROM ncm_crisis_site_capture_runs WHERE crisis_id=:id
        ORDER BY scope_revision ASC LIMIT :limit'''),
        {'id': crisis_id, 'limit': max_rows + 1}).mappings().all()
    observations = conn.execute(text('''SELECT observation_id, crisis_id, site_id,
        scope_revision, observation_clock, source_basis, event_specific_association,
        evidence_status, contract_version
        FROM ncm_crisis_site_activation_observations WHERE crisis_id=:id
        ORDER BY observation_clock ASC, observation_id ASC LIMIT :limit'''),
        {'id': crisis_id, 'limit': max_rows + 1}).mappings().all()
    if len(runs) > max_rows or len(observations) > max_rows:
        raise RuntimeError('Crisis observation evidence exceeds bounded read')
    if not runs:
        return {'capture_status': 'not_recorded', 'capture_runs': [],
                'observations': [], 'complete': False}
    complete = all(row['capture_status'] == 'complete_inventory_snapshot' for row in runs)
    return {'capture_status': 'complete_inventory_snapshot' if complete else 'partial_recorded_capture',
            'capture_runs': [dict(row) for row in runs],
            'observations': [dict(row) for row in observations], 'complete': bool(complete)}
