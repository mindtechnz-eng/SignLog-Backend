"""NZ-NCM CRR-01A C2-A — protected, versioned crisis/Site context issuance.

This component records Backend issuance of *candidate* attention to a registered,
currently bound active Site. It does NOT prove Kiosk delivery, display, human
acknowledgement, response obligation or SOS/FHC attribution. It never changes
original signal ingestion or the public Kiosk bootstrap.

Issuance is available only to the protected server-to-server route in main.py.
Kiosk authentication and receipt are separate C2-B work, not inferred here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import asin, cos, isfinite, radians, sin, sqrt
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import text
from sqlalchemy.engine import Connection

CONTRACT = "ncm-crr01a.context-issuance.v1"
BASIS = "backend_stored_site_geography_at_context_issue"
STATUS = "issued_candidate_only_no_renderer_receipt"
ISSUE_BUCKET_SECONDS = 300
MAX_ACTIVE_CRISES = 32


@dataclass(frozen=True)
class ContextIssueResult:
    kiosk_id: str
    site_id: str
    issued_at: str
    contexts: list[dict[str, Any]]
    examined_crises: int
    in_area_candidates: int
    out_of_area: int
    unknown_geometry: int


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("An explicit UTC issue clock is required")
    moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("Issue clock must be timezone aware")
    return moment.astimezone(timezone.utc)


def _coordinate(value: Any, low: float, high: float) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        coordinate = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if not isfinite(coordinate) or not low <= coordinate <= high:
        return None
    return coordinate


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat_delta = radians(lat2 - lat1)
    lon_delta = radians(lon2 - lon1)
    h = (sin(lat_delta / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(lon_delta / 2) ** 2)
    return 6371.0 * 2 * asin(min(1.0, sqrt(max(0.0, h))))


def ensure_crr01a_context_schema(conn: Connection) -> None:
    """Cross-engine additive tables. Existing N1A and CRR-01A Cut 1 unchanged."""
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ncm_crisis_attention_contexts (
            context_id TEXT PRIMARY KEY,
            crisis_id TEXT NOT NULL,
            site_id TEXT NOT NULL,
            kiosk_id TEXT NOT NULL,
            scope_revision INTEGER NOT NULL CHECK (scope_revision > 0),
            issue_bucket INTEGER NOT NULL,
            input_fingerprint TEXT NOT NULL,
            issued_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            site_latitude REAL NOT NULL,
            site_longitude REAL NOT NULL,
            distance_km REAL NOT NULL,
            geometry_snapshot_json TEXT NOT NULL,
            response_required INTEGER CHECK (response_required IN (0,1) OR response_required IS NULL),
            source_basis TEXT NOT NULL,
            evidence_status TEXT NOT NULL CHECK (
                evidence_status='issued_candidate_only_no_renderer_receipt'),
            issuer TEXT NOT NULL,
            contract_version TEXT NOT NULL,
            UNIQUE (crisis_id, site_id, kiosk_id, scope_revision, issue_bucket, input_fingerprint),
            FOREIGN KEY(crisis_id) REFERENCES crises(crisis_id),
            FOREIGN KEY(site_id) REFERENCES sites(site_id)
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_ncm_crr01a_context_crisis_site_time
        ON ncm_crisis_attention_contexts(crisis_id, site_id, issued_at)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_ncm_crr01a_context_kiosk_time
        ON ncm_crisis_attention_contexts(kiosk_id, issued_at)
    """))


def issue_candidate_contexts(
    conn: Connection, *, kiosk_id: str, issued_at: str,
    max_active_crises: int = MAX_ACTIVE_CRISES,
) -> ContextIssueResult:
    """Issue all currently intersecting named-crisis candidates for one bound Kiosk.

    Protected caller only. Preserves overlaps and original IDs; reads current
    Backend Site location, *not* historical exposure. Exact, stable identities
    are deduplicated by a bounded issue slot and source fingerprint, including
    Site/geometry changes. A slot is valid for 5–10 minutes depending on entry
    time. No browser exposure, receipt, or SOS event association occurs here.
    """
    kiosk = kiosk_id.strip() if isinstance(kiosk_id, str) else ""
    if not kiosk or len(kiosk) > 200:
        raise ValueError("Valid bounded Kiosk identity required")
    if not isinstance(max_active_crises, int) or isinstance(max_active_crises, bool) or not 1 <= max_active_crises <= 256:
        raise ValueError("Invalid active-crisis bound")
    now = _parse_utc(issued_at)
    now_iso = now.isoformat()
    epoch_slot = int(now.timestamp()) // ISSUE_BUCKET_SECONDS
    expires_at = datetime.fromtimestamp((epoch_slot + 2) * ISSUE_BUCKET_SECONDS, tz=timezone.utc).isoformat()

    site = conn.execute(text("""
        SELECT s.site_id, s.kiosk_id, s.latitude, s.longitude,
               s.operational_state, s.updated_at
        FROM sites s WHERE s.kiosk_id=:kiosk_id AND s.operational_state='active'
    """), {"kiosk_id": kiosk}).mappings().first()
    if site is None:
        raise LookupError("Kiosk is not bound to an active registered Site")

    crises = conn.execute(text("""
        SELECT crisis_id, event_revision, center_latitude, center_longitude,
               radius_km, response_required, created_at
        FROM crises WHERE status='active'
        ORDER BY crisis_id ASC LIMIT :bound
    """), {"bound": max_active_crises + 1}).mappings().all()
    if len(crises) > max_active_crises:
        raise RuntimeError("Active crises exceed bounded context issuance; no partial issue")

    site_lat = _coordinate(site["latitude"], -90, 90)
    site_lon = _coordinate(site["longitude"], -180, 180)
    candidates: list[dict[str, Any]] = []
    in_area = outside = unknown = 0

    for crisis in crises:
        c_lat = _coordinate(crisis["center_latitude"], -90, 90)
        c_lon = _coordinate(crisis["center_longitude"], -180, 180)
        radius = _coordinate(crisis["radius_km"], 0, 100_000)
        revision = crisis["event_revision"]
        if (site_lat is None or site_lon is None or c_lat is None or c_lon is None
            or radius is None or radius <= 0 or not isinstance(revision, int)
            or isinstance(revision, bool) or revision < 1):
            unknown += 1
            continue
        distance = _distance_km(site_lat, site_lon, c_lat, c_lon)
        if distance > radius:
            outside += 1
            continue
        in_area += 1

        original_geometry = {"center_latitude": c_lat, "center_longitude": c_lon,
                             "radius_km": radius, "crisis_recorded_at": crisis["created_at"]}
        policy = crisis["response_required"]
        if policy not in (None, 0, 1, False, True):
            policy = None
        # Never transform a nullable policy into a report-grade obligation.
        policy = None if policy is None else int(policy)
        identity = {
            "crisis_id": crisis["crisis_id"], "site_id": site["site_id"],
            "kiosk_id": kiosk, "revision": revision,
            "site_latitude": site_lat, "site_longitude": site_lon,
            "site_updated_at": site["updated_at"],
            "geometry": original_geometry, "response_required": policy,
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        context_id = str(uuid5(NAMESPACE_URL, f"signlog/crr01a/c2a/{fingerprint}/{epoch_slot}"))
        conn.execute(text("""
            INSERT INTO ncm_crisis_attention_contexts (
                context_id, crisis_id, site_id, kiosk_id, scope_revision,
                issue_bucket, input_fingerprint, issued_at, expires_at,
                site_latitude, site_longitude, distance_km, geometry_snapshot_json,
                response_required, source_basis, evidence_status, issuer, contract_version
            ) VALUES (
                :context_id, :crisis_id, :site_id, :kiosk_id, :scope_revision,
                :issue_bucket, :fingerprint, :issued_at, :expires_at,
                :site_latitude, :site_longitude, :distance_km, :geometry_json,
                :response_required, :basis, :status, 'protected_backend_server', :contract
            ) ON CONFLICT(context_id) DO NOTHING
        """), {
            "context_id": context_id, "crisis_id": crisis["crisis_id"], "site_id": site["site_id"],
            "kiosk_id": kiosk, "scope_revision": revision, "issue_bucket": epoch_slot,
            "fingerprint": fingerprint, "issued_at": now_iso, "expires_at": expires_at,
            "site_latitude": site_lat, "site_longitude": site_lon,
            "distance_km": distance, "geometry_json": json.dumps(original_geometry, sort_keys=True),
            "response_required": policy, "basis": BASIS, "status": STATUS, "contract": CONTRACT,
        })
        stored = conn.execute(text("""
            SELECT context_id, crisis_id, site_id, kiosk_id, scope_revision,
                   issued_at, expires_at, response_required,
                   source_basis, evidence_status, contract_version
            FROM ncm_crisis_attention_contexts WHERE context_id=:context_id
        """), {"context_id": context_id}).mappings().one()
        candidates.append(dict(stored))

    return ContextIssueResult(
        kiosk_id=kiosk, site_id=str(site["site_id"]), issued_at=now_iso,
        contexts=candidates, examined_crises=len(crises), in_area_candidates=in_area,
        out_of_area=outside, unknown_geometry=unknown,
    )
