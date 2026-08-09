from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from math import radians, sin, cos, sqrt, atan2
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection, Engine, Result

# =========================================================
# SignLog Backend v1.8
# Shared Authority Layer for:
# - Kiosk
# - Admin
# - NZ-NCM
# Dual DB Mode:
# - Postgres via DATABASE_URL
# - SQLite fallback for local dev
# =========================================================

load_dotenv()

APP_NAME = "signlog-backend"
DB_PATH = os.getenv("SIGNLOG_DB_PATH", "signlog.db")
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    ENGINE: Engine = create_engine(DATABASE_URL, future=True)
    DB_MODE = "postgres"
else:
    ENGINE = create_engine(f"sqlite:///{DB_PATH}", future=True)
    DB_MODE = "sqlite"

app = FastAPI(title="SignLog Backend", version="1.8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Helpers
# =========================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(ts: Optional[str]) -> datetime:
    if not ts:
        return datetime.now(timezone.utc)

    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))

        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    except Exception:
        return datetime.now(timezone.utc)


def parse_query_iso(
    value: Optional[str],
    field_name: str,
) -> Optional[str]:
    """
    Parse an optional event-retrieval timestamp into the
    same canonical UTC ISO representation used by stored
    SignLog events.

    Unlike parse_iso(), invalid retrieval filters must not
    silently become the current time. A malformed cursor or
    date boundary would otherwise create an incomplete or
    misleading history result.
    """
    if value is None:
        return None

    cleaned = value.strip()

    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {field_name}. "
                "Provide a valid ISO timestamp."
            ),
        )

    try:
        parsed = datetime.fromisoformat(
            cleaned.replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed.astimezone(
            timezone.utc
        ).isoformat()

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {field_name}. "
                "Provide a valid ISO timestamp."
            ),
        ) from error


def normalise_sign_action(
    value: str,
) -> Literal["signin", "signout"]:
    v = (value or "").strip().lower()

    if v in {
        "in",
        "signin",
        "sign-in",
        "sign_in",
    }:
        return "signin"

    if v in {
        "out",
        "signout",
        "sign-out",
        "sign_out",
    }:
        return "signout"

    raise HTTPException(
        status_code=400,
        detail=(
            "Invalid sign action. "
            "Use in/out or signin/signout."
        ),
    )


def normalise_event_source(
    value: Optional[str],
) -> str:
    cleaned = (
        value or ""
    ).strip()

    return cleaned or "kiosk"


def json_dumps(data: Any) -> str:
    return json.dumps(
        data or {},
        ensure_ascii=False,
    )


def json_loads(
    value: Optional[str],
) -> Dict[str, Any]:
    if not value:
        return {}

    try:
        raw = json.loads(value)

        return (
            raw
            if isinstance(raw, dict)
            else {}
        )

    except Exception:
        return {}


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    r = 6371.0

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    return (
        2
        * r
        * atan2(
            sqrt(a),
            sqrt(1 - a),
        )
    )


@contextmanager
def get_db():
    with ENGINE.begin() as conn:
        yield conn


def fetch_one(
    conn: Connection,
    sql: str,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    result: Result = conn.execute(
        text(sql),
        params or {},
    )

    row = result.mappings().first()

    return (
        dict(row)
        if row
        else None
    )


def fetch_all(
    conn: Connection,
    sql: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    result: Result = conn.execute(
        text(sql),
        params or {},
    )

    return [
        dict(row)
        for row in result.mappings().all()
    ]


def execute_write(
    conn: Connection,
    sql: str,
    params: Optional[Dict[str, Any]] = None,
) -> Result:
    return conn.execute(
        text(sql),
        params or {},
    )


def get_table_column_names(
    conn: Connection,
    table_name: str,
) -> set[str]:
    inspector = inspect(conn)

    return {
        str(column["name"])
        for column in inspector.get_columns(
            table_name
        )
    }


def add_column_if_missing(
    conn: Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    existing_columns = get_table_column_names(
        conn,
        table_name,
    )

    if column_name in existing_columns:
        return

    execute_write(
        conn,
        f"""
        ALTER TABLE {table_name}
        ADD COLUMN {column_name} {column_definition}
        """,
    )


def ensure_sites_lifecycle_schema(
    conn: Connection,
) -> None:
    lifecycle_columns = (
        (
            "operational_state",
            "TEXT NOT NULL DEFAULT 'active'",
        ),
        (
            "archived_at",
            "TEXT",
        ),
        (
            "unlinked_at",
            "TEXT",
        ),
        (
            "archive_note",
            "TEXT",
        ),
        (
            "last_kiosk_id",
            "TEXT",
        ),
    )

    for (
        column_name,
        column_definition,
    ) in lifecycle_columns:
        add_column_if_missing(
            conn,
            "sites",
            column_name,
            column_definition,
        )

    # Defensive backfill for any legacy or imported rows.
    execute_write(
        conn,
        """
        UPDATE sites
        SET operational_state = 'active'
        WHERE operational_state IS NULL
           OR TRIM(operational_state) = ''
        """,
    )


def ensure_sites_display_policy_schema(
    conn: Connection,
) -> None:
    """
    Phase 3 operational headcount display policy.

    Stored on the site because Admin/backend owns the
    deployment policy. Both permissions default OFF so
    existing deployments do not expose occupancy merely
    because the software is upgraded.
    """
    display_policy_columns = (
        (
            "allow_kiosk_headcount_display",
            "INTEGER NOT NULL DEFAULT 0",
        ),
        (
            "allow_physical_headcount_display",
            "INTEGER NOT NULL DEFAULT 0",
        ),
    )

    for (
        column_name,
        column_definition,
    ) in display_policy_columns:
        add_column_if_missing(
            conn,
            "sites",
            column_name,
            column_definition,
        )

    execute_write(
        conn,
        """
        UPDATE sites
        SET allow_kiosk_headcount_display = 0
        WHERE allow_kiosk_headcount_display IS NULL
        """,
    )

    execute_write(
        conn,
        """
        UPDATE sites
        SET allow_physical_headcount_display = 0
        WHERE allow_physical_headcount_display IS NULL
        """,
    )


# =========================================================
# Database Init
# =========================================================

def init_db() -> None:
    with get_db() as conn:
        if DB_MODE == "sqlite":
            conn.execute(
                text(
                    "PRAGMA journal_mode=WAL;"
                )
            )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS sites (
                    site_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    address TEXT,
                    city TEXT,
                    company_name TEXT,
                    kiosk_id TEXT UNIQUE,
                    latitude REAL,
                    longitude REAL,
                    site_policy_mode TEXT DEFAULT 'standard',
                    allow_kiosk_headcount_display INTEGER NOT NULL DEFAULT 0,
                    allow_physical_headcount_display INTEGER NOT NULL DEFAULT 0,
                    operational_state TEXT NOT NULL DEFAULT 'active',
                    archived_at TEXT,
                    unlinked_at TEXT,
                    archive_note TEXT,
                    last_kiosk_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        )

        ensure_sites_lifecycle_schema(conn)
        ensure_sites_display_policy_schema(conn)

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_sites_operational_state
                ON sites(operational_state)
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS site_status (
                    site_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'idle',
                    active_count INTEGER NOT NULL DEFAULT 0,
                    headcount_status TEXT NOT NULL DEFAULT 'pending',
                    sos_active INTEGER NOT NULL DEFAULT 0,
                    full_headcount_confirmed INTEGER NOT NULL DEFAULT 0,
                    last_event_at TEXT,
                    last_sos_at TEXT,
                    last_full_headcount_at TEXT,
                    FOREIGN KEY(site_id)
                        REFERENCES sites(site_id)
                )
                """
            )
        )

        if DB_MODE == "postgres":
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id SERIAL PRIMARY KEY,
                        site_id TEXT NOT NULL,
                        device_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        payload_json TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(site_id)
                            REFERENCES sites(site_id)
                    )
                    """
                )
            )

        else:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        site_id TEXT NOT NULL,
                        device_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        payload_json TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(site_id)
                            REFERENCES sites(site_id)
                    )
                    """
                )
            )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_events_site_id
                ON events(site_id)
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_events_occurred_at
                ON events(occurred_at)
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_events_event_type
                ON events(event_type)
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_events_site_time_id
                ON events(
                    site_id,
                    occurred_at,
                    id
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS crises (
                    crisis_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    crisis_class TEXT NOT NULL,
                    hazard_type TEXT,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    center_latitude REAL,
                    center_longitude REAL,
                    radius_km REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    cleared_at TEXT
                )
                """
            )
        )


@app.on_event("startup")
def on_startup():
    init_db()


# =========================================================
# Pydantic Models
# =========================================================

class CreateSiteRequest(BaseModel):
    name: str = Field(min_length=1)

    address: Optional[str] = None
    city: Optional[str] = None
    company_name: Optional[str] = None
    kiosk_id: Optional[str] = None
    site_id: Optional[str] = None

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    site_policy_mode: Literal[
        "lenient",
        "standard",
        "strict",
    ] = "standard"

    allow_kiosk_headcount_display: bool = False
    allow_physical_headcount_display: bool = False


class UpdateSiteRequest(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    company_name: Optional[str] = None
    kiosk_id: Optional[str] = None

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    site_policy_mode: Optional[
        Literal[
            "lenient",
            "standard",
            "strict",
        ]
    ] = None

    allow_kiosk_headcount_display: Optional[bool] = None
    allow_physical_headcount_display: Optional[bool] = None


class BindKioskRequest(BaseModel):
    kiosk_id: str = Field(min_length=1)
    site_id: str = Field(min_length=1)


class ArchiveAndReleaseSiteRequest(BaseModel):
    archive_note: Optional[str] = Field(
        default=None,
        max_length=2000,
    )


class SignEventRequest(BaseModel):
    site_id: Optional[str] = None

    device_id: str = Field(min_length=1)

    worker_id: Optional[str] = None
    name: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None

    type: str
    timestamp: Optional[str] = None
    source: Optional[str] = None


class HazardEventRequest(BaseModel):
    site_id: Optional[str] = None

    device_id: str = Field(min_length=1)

    worker_id: Optional[str] = None
    name: Optional[str] = None

    description: str = Field(min_length=1)

    severity: Literal[
        "low",
        "medium",
        "high",
    ] = "medium"

    timestamp: Optional[str] = None
    source: Optional[str] = None


class IncidentEventRequest(BaseModel):
    site_id: Optional[str] = None

    device_id: str = Field(min_length=1)

    worker_id: Optional[str] = None
    name: Optional[str] = None

    description: str = Field(min_length=1)

    injury: bool = False

    timestamp: Optional[str] = None
    source: Optional[str] = None


class SOSEventRequest(BaseModel):
    site_id: Optional[str] = None

    device_id: str = Field(min_length=1)

    timestamp: Optional[str] = None
    note: Optional[str] = None
    source: Optional[str] = None


class FullHeadcountEventRequest(BaseModel):
    site_id: Optional[str] = None

    device_id: str = Field(min_length=1)

    timestamp: Optional[str] = None
    note: Optional[str] = None
    source: Optional[str] = None


class CreateCrisisRequest(BaseModel):
    crisis_id: Optional[str] = None

    source: Literal[
        "manual",
        "official_feed",
        "inferred_cluster",
    ] = "manual"

    crisis_class: Literal[
        "local_crisis",
        "regional_crisis",
        "soe",
    ]

    hazard_type: Optional[str] = None

    name: str = Field(min_length=1)

    center_latitude: float
    center_longitude: float

    radius_km: float = Field(gt=0)


# =========================================================
# DB-level Functions
# =========================================================

def next_site_id(
    conn: Connection,
) -> str:
    row = fetch_one(
        conn,
        """
        SELECT site_id
        FROM sites
        WHERE site_id LIKE :pattern
        ORDER BY site_id DESC
        LIMIT 1
        """,
        {
            "pattern": "site-%",
        },
    )

    if not row:
        return "site-001"

    last = row["site_id"]

    try:
        n = int(
            last.split("-")[1]
        ) + 1

    except Exception:
        n = 1

    return f"site-{n:03d}"


def ensure_site_status(
    conn: Connection,
    site_id: str,
) -> None:
    exists = fetch_one(
        conn,
        """
        SELECT 1
        FROM site_status
        WHERE site_id = :site_id
        """,
        {
            "site_id": site_id,
        },
    )

    if exists:
        return

    execute_write(
        conn,
        """
        INSERT INTO site_status (
            site_id,
            status,
            active_count,
            headcount_status,
            sos_active,
            full_headcount_confirmed,
            last_event_at,
            last_sos_at,
            last_full_headcount_at
        )
        VALUES (
            :site_id,
            'idle',
            0,
            'pending',
            0,
            0,
            NULL,
            NULL,
            NULL
        )
        """,
        {
            "site_id": site_id,
        },
    )


def resolve_site_id(
    conn: Connection,
    site_id: Optional[str],
    device_id: str,
) -> str:
    supplied_site_id = (
        site_id or ""
    ).strip()

    supplied_device_id = (
        device_id or ""
    ).strip()

    if not supplied_device_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid device_id. "
                "The event was not recorded."
            ),
        )

    current_binding = fetch_one(
        conn,
        """
        SELECT
            site_id,
            operational_state
        FROM sites
        WHERE kiosk_id = :device_id
        """,
        {
            "device_id":
                supplied_device_id,
        },
    )

    if supplied_site_id:
        supplied_site = fetch_one(
            conn,
            """
            SELECT
                site_id,
                operational_state
            FROM sites
            WHERE site_id = :site_id
            """,
            {
                "site_id":
                    supplied_site_id,
            },
        )

        if not supplied_site:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Supplied site_id is "
                    "invalid. "
                    "The event was not "
                    "recorded."
                ),
            )

        if (
            supplied_site.get(
                "operational_state"
            )
            != "active"
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Archived site cannot "
                    "receive kiosk events. "
                    "The event was not "
                    "recorded."
                ),
            )

        if current_binding:
            if (
                current_binding.get(
                    "operational_state"
                )
                != "active"
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Archived site cannot "
                        "receive kiosk events. "
                        "The event was not "
                        "recorded."
                    ),
                )

            if (
                current_binding[
                    "site_id"
                ]
                != supplied_site[
                    "site_id"
                ]
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Kiosk binding mismatch. "
                        "The supplied site_id "
                        "does not match the "
                        "kiosk's current site "
                        "binding. "
                        "The event was not "
                        "recorded."
                    ),
                )

        return supplied_site[
            "site_id"
        ]

    if current_binding:
        if (
            current_binding.get(
                "operational_state"
            )
            != "active"
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Archived site cannot "
                    "receive kiosk events. "
                    "The event was not "
                    "recorded."
                ),
            )

        return current_binding[
            "site_id"
        ]

    raise HTTPException(
        status_code=400,
        detail=(
            "No site resolved. "
            "Bind kiosk to site or "
            "provide valid site_id. "
            "The event was not recorded."
        ),
    )
def insert_event(
    conn: Connection,
    *,
    site_id: str,
    device_id: str,
    event_type: str,
    occurred_at: str,
    payload: Dict[str, Any],
) -> int:
    created_at = utc_now_iso()

    event_params = {
        "site_id": site_id,
        "device_id": device_id,
        "event_type": event_type,
        "occurred_at": occurred_at,
        "payload_json": json_dumps(payload),
        "created_at": created_at,
    }

    if DB_MODE == "postgres":
        result = execute_write(
            conn,
            """
            INSERT INTO events (
                site_id,
                device_id,
                event_type,
                occurred_at,
                payload_json,
                created_at
            )
            VALUES (
                :site_id,
                :device_id,
                :event_type,
                :occurred_at,
                :payload_json,
                :created_at
            )
            RETURNING id
            """,
            event_params,
        )

        return int(
            result.scalar_one()
        )

    result = execute_write(
        conn,
        """
        INSERT INTO events (
            site_id,
            device_id,
            event_type,
            occurred_at,
            payload_json,
            created_at
        )
        VALUES (
            :site_id,
            :device_id,
            :event_type,
            :occurred_at,
            :payload_json,
            :created_at
        )
        """,
        event_params,
    )

    return int(
        result.lastrowid
    )


def recompute_site_state(
    conn: Connection,
    site_id: str,
) -> Dict[str, Any]:
    rows = fetch_all(
        conn,
        """
        SELECT
            event_type,
            occurred_at
        FROM events
        WHERE site_id = :site_id
        ORDER BY
            occurred_at ASC,
            id ASC
        """,
        {
            "site_id": site_id,
        },
    )

    active_count = 0
    sos_active = 0
    full_headcount_confirmed = 0

    last_event_at = None
    last_sos_at = None
    last_full_headcount_at = None

    for row in rows:
        event_type = row["event_type"]
        occurred_at = row["occurred_at"]

        last_event_at = occurred_at

        if event_type == "signin":
            active_count += 1
            full_headcount_confirmed = 0

        elif event_type == "signout":
            active_count = max(
                0,
                active_count - 1,
            )

        elif event_type == "sos":
            sos_active = 1
            full_headcount_confirmed = 0
            last_sos_at = occurred_at

        elif event_type == "full_headcount":
            sos_active = 0
            full_headcount_confirmed = 1
            last_full_headcount_at = occurred_at

    if sos_active:
        status = "emergency"
        headcount_status = "alert"

    elif active_count > 0:
        status = "live"

        headcount_status = (
            "safe"
            if full_headcount_confirmed
            else "pending"
        )

    else:
        status = "idle"

        headcount_status = (
            "safe"
            if full_headcount_confirmed
            else "pending"
        )

    ensure_site_status(
        conn,
        site_id,
    )

    execute_write(
        conn,
        """
        UPDATE site_status
        SET status = :status,
            active_count = :active_count,
            headcount_status = :headcount_status,
            sos_active = :sos_active,
            full_headcount_confirmed =
                :full_headcount_confirmed,
            last_event_at = :last_event_at,
            last_sos_at = :last_sos_at,
            last_full_headcount_at =
                :last_full_headcount_at
        WHERE site_id = :site_id
        """,
        {
            "status": status,
            "active_count": active_count,
            "headcount_status": headcount_status,
            "sos_active": sos_active,
            "full_headcount_confirmed":
                full_headcount_confirmed,
            "last_event_at": last_event_at,
            "last_sos_at": last_sos_at,
            "last_full_headcount_at":
                last_full_headcount_at,
            "site_id": site_id,
        },
    )

    return {
        "status": status,
        "active_count": active_count,
        "headcount_status": headcount_status,
        "sos_active": bool(sos_active),
        "full_headcount_confirmed":
            bool(full_headcount_confirmed),
        "last_event_at": last_event_at,
        "last_sos_at": last_sos_at,
        "last_full_headcount_at":
            last_full_headcount_at,
    }


def fetch_site_summary(
    conn: Connection,
    site_id: str,
) -> Dict[str, Any]:
    row = fetch_one(
        conn,
        """
        SELECT
            s.site_id,
            s.name,
            s.address,
            s.city,
            s.company_name,
            s.kiosk_id,
            s.latitude,
            s.longitude,
            s.site_policy_mode,
            s.allow_kiosk_headcount_display,
            s.allow_physical_headcount_display,
            s.operational_state,
            s.archived_at,
            s.unlinked_at,
            s.archive_note,
            s.last_kiosk_id,
            ss.status,
            ss.active_count,
            ss.headcount_status,
            ss.sos_active,
            ss.full_headcount_confirmed,
            ss.last_event_at,
            ss.last_sos_at,
            ss.last_full_headcount_at
        FROM sites s
        LEFT JOIN site_status ss
            ON ss.site_id = s.site_id
        WHERE s.site_id = :site_id
        """,
        {
            "site_id": site_id,
        },
    )

    if not row:
        raise HTTPException(
            status_code=404,
            detail="Site not found.",
        )

    row["allow_kiosk_headcount_display"] = bool(
        row.get("allow_kiosk_headcount_display")
    )
    row["allow_physical_headcount_display"] = bool(
        row.get("allow_physical_headcount_display")
    )

    return row


def list_active_crises(
    conn: Connection,
) -> List[Dict[str, Any]]:
    return fetch_all(
        conn,
        """
        SELECT *
        FROM crises
        WHERE status = 'active'
        ORDER BY created_at DESC
        """,
    )


def site_in_any_active_crisis(
    site: Dict[str, Any],
    crises: List[Dict[str, Any]],
) -> bool:
    lat = site.get("latitude")
    lon = site.get("longitude")

    if lat is None or lon is None:
        return False

    for crisis in crises:
        crisis_lat = crisis.get(
            "center_latitude"
        )
        crisis_lon = crisis.get(
            "center_longitude"
        )
        radius = crisis.get(
            "radius_km"
        )

        if (
            crisis_lat is None
            or crisis_lon is None
            or radius is None
        ):
            continue

        distance = haversine_km(
            float(lat),
            float(lon),
            float(crisis_lat),
            float(crisis_lon),
        )

        if distance <= float(radius):
            return True

    return False


def classify_ncm_site_state(
    site: Dict[str, Any],
    crises: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Shared NZ-NCM classification for every backend read
    surface that needs crisis interpretation.

    This preserves one red / orange / green law for both
    the NZ-NCM feed and the kiosk bootstrap contract.
    """
    in_crisis_area = site_in_any_active_crisis(
        site,
        crises,
    )

    ncm_state = None
    priority = None

    if site.get("sos_active"):
        ncm_state = "red"
        priority = 1

    elif (
        in_crisis_area
        and site.get(
            "active_count",
            0,
        ) > 0
        and not site.get(
            "full_headcount_confirmed"
        )
    ):
        ncm_state = "orange"
        priority = 2

    elif site.get(
        "full_headcount_confirmed"
    ):
        ncm_state = "green"
        priority = 3

    return {
        "ncm_state": ncm_state,
        "priority": priority,
        "in_active_crisis_area":
            in_crisis_area,
    }


# =========================================================
# Health
# =========================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": "1.8.0",
        "db_mode": DB_MODE,
        "db_path": (
            DB_PATH
            if DB_MODE == "sqlite"
            else None
        ),
        "time": utc_now_iso(),
    }


# =========================================================
# Admin - Sites
# =========================================================

@app.post("/admin/site")
async def admin_create_site(
    payload: CreateSiteRequest,
):
    with get_db() as conn:
        site_id = (
            payload.site_id
            or next_site_id(conn)
        )

        now = utc_now_iso()

        existing = fetch_one(
            conn,
            """
            SELECT 1
            FROM sites
            WHERE site_id = :site_id
            """,
            {
                "site_id": site_id,
            },
        )

        if existing:
            raise HTTPException(
                status_code=409,
                detail="Site ID already exists.",
            )

        if payload.kiosk_id:
            kiosk_exists = fetch_one(
                conn,
                """
                SELECT site_id
                FROM sites
                WHERE kiosk_id = :kiosk_id
                """,
                {
                    "kiosk_id":
                        payload.kiosk_id,
                },
            )

            if kiosk_exists:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Kiosk ID is already "
                        "bound to another site."
                    ),
                )

        execute_write(
            conn,
            """
            INSERT INTO sites (
                site_id,
                name,
                address,
                city,
                company_name,
                kiosk_id,
                latitude,
                longitude,
                site_policy_mode,
                allow_kiosk_headcount_display,
                allow_physical_headcount_display,
                created_at,
                updated_at
            )
            VALUES (
                :site_id,
                :name,
                :address,
                :city,
                :company_name,
                :kiosk_id,
                :latitude,
                :longitude,
                :site_policy_mode,
                :allow_kiosk_headcount_display,
                :allow_physical_headcount_display,
                :created_at,
                :updated_at
            )
            """,
            {
                "site_id": site_id,
                "name": payload.name.strip(),
                "address": payload.address,
                "city": payload.city,
                "company_name":
                    payload.company_name,
                "kiosk_id":
                    payload.kiosk_id,
                "latitude":
                    payload.latitude,
                "longitude":
                    payload.longitude,
                "site_policy_mode":
                    payload.site_policy_mode,
                "allow_kiosk_headcount_display":
                    1 if payload.allow_kiosk_headcount_display else 0,
                "allow_physical_headcount_display":
                    1 if payload.allow_physical_headcount_display else 0,
                "created_at": now,
                "updated_at": now,
            },
        )

        ensure_site_status(
            conn,
            site_id,
        )

        return fetch_site_summary(
            conn,
            site_id,
        )


@app.get("/admin/sites")
async def admin_list_sites():
    with get_db() as conn:
        rows = fetch_all(
            conn,
            """
            SELECT
                s.site_id,
                s.name,
                s.address,
                s.city,
                s.company_name,
                s.kiosk_id,
                s.latitude,
                s.longitude,
                s.site_policy_mode,
                s.allow_kiosk_headcount_display,
                s.allow_physical_headcount_display,
                s.operational_state,
                s.archived_at,
                s.unlinked_at,
                s.archive_note,
                s.last_kiosk_id,
                ss.status,
                ss.active_count,
                ss.headcount_status,
                ss.sos_active,
                ss.full_headcount_confirmed,
                ss.last_event_at,
                ss.last_sos_at,
                ss.last_full_headcount_at
            FROM sites s
            LEFT JOIN site_status ss
                ON ss.site_id = s.site_id
            ORDER BY
                s.created_at DESC,
                s.site_id DESC
            """
        )

        crises = list_active_crises(
            conn
        )

        results = []

        for site in rows:
            site[
                "allow_kiosk_headcount_display"
            ] = bool(
                site.get(
                    "allow_kiosk_headcount_display"
                )
            )

            site[
                "allow_physical_headcount_display"
            ] = bool(
                site.get(
                    "allow_physical_headcount_display"
                )
            )

            site[
                "in_active_crisis_area"
            ] = site_in_any_active_crisis(
                site,
                crises,
            )

            results.append(site)

        return results


@app.put("/admin/sites/{site_id}")
async def admin_update_site(
    site_id: str,
    payload: UpdateSiteRequest,
):
    with get_db() as conn:
        site = fetch_one(
            conn,
            """
            SELECT *
            FROM sites
            WHERE site_id = :site_id
            """,
            {
                "site_id": site_id,
            },
        )

        if not site:
            raise HTTPException(
                status_code=404,
                detail="Site not found.",
            )

        updates = payload.dict(
            exclude_unset=True
        )

        kiosk_change_requested = (
            "kiosk_id" in updates
        )

        requested_kiosk_id = updates.get(
            "kiosk_id"
        )

        if (
            kiosk_change_requested
            and requested_kiosk_id
        ):
            if (
                site.get(
                    "operational_state"
                )
                != "active"
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Archived site cannot "
                        "receive a kiosk. "
                        "Restore the site before "
                        "binding."
                    ),
                )

            other = fetch_one(
                conn,
                """
                SELECT site_id
                FROM sites
                WHERE kiosk_id = :kiosk_id
                  AND site_id <> :site_id
                """,
                {
                    "kiosk_id":
                        requested_kiosk_id,
                    "site_id":
                        site_id,
                },
            )

            if other:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Kiosk ID is already "
                        "bound to another site."
                    ),
                )

        merged = {
            **site,
            **updates,
            "updated_at": utc_now_iso(),
        }

        if (
            kiosk_change_requested
            and site.get(
                "operational_state"
            ) == "active"
        ):
            result = execute_write(
                conn,
                """
                UPDATE sites
                SET name = :name,
                    address = :address,
                    city = :city,
                    company_name =
                        :company_name,
                    kiosk_id = :kiosk_id,
                    latitude = :latitude,
                    longitude = :longitude,
                    site_policy_mode =
                        :site_policy_mode,
                    allow_kiosk_headcount_display =
                        :allow_kiosk_headcount_display,
                    allow_physical_headcount_display =
                        :allow_physical_headcount_display,
                    updated_at = :updated_at
                WHERE site_id = :site_id
                  AND operational_state =
                        'active'
                """,
                {
                    "name": merged["name"],
                    "address":
                        merged.get("address"),
                    "city":
                        merged.get("city"),
                    "company_name":
                        merged.get(
                            "company_name"
                        ),
                    "kiosk_id":
                        merged.get(
                            "kiosk_id"
                        ),
                    "latitude":
                        merged.get(
                            "latitude"
                        ),
                    "longitude":
                        merged.get(
                            "longitude"
                        ),
                    "site_policy_mode":
                        merged.get(
                            "site_policy_mode"
                        ),
                    "allow_kiosk_headcount_display":
                        1 if merged.get(
                            "allow_kiosk_headcount_display"
                        ) else 0,
                    "allow_physical_headcount_display":
                        1 if merged.get(
                            "allow_physical_headcount_display"
                        ) else 0,
                    "updated_at":
                        merged["updated_at"],
                    "site_id": site_id,
                },
            )

            if result.rowcount != 1:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Site lifecycle changed "
                        "before the update "
                        "completed. Refresh and "
                        "try again."
                    ),
                )

        else:
            # Metadata updates deliberately do
            # not write kiosk_id. This prevents
            # a concurrent archive action from
            # having its kiosk release reversed.
            execute_write(
                conn,
                """
                UPDATE sites
                SET name = :name,
                    address = :address,
                    city = :city,
                    company_name =
                        :company_name,
                    latitude = :latitude,
                    longitude = :longitude,
                    site_policy_mode =
                        :site_policy_mode,
                    allow_kiosk_headcount_display =
                        :allow_kiosk_headcount_display,
                    allow_physical_headcount_display =
                        :allow_physical_headcount_display,
                    updated_at = :updated_at
                WHERE site_id = :site_id
                """,
                {
                    "name":
                        merged["name"],
                    "address":
                        merged.get(
                            "address"
                        ),
                    "city":
                        merged.get("city"),
                    "company_name":
                        merged.get(
                            "company_name"
                        ),
                    "latitude":
                        merged.get(
                            "latitude"
                        ),
                    "longitude":
                        merged.get(
                            "longitude"
                        ),
                    "site_policy_mode":
                        merged.get(
                            "site_policy_mode"
                        ),
                    "allow_kiosk_headcount_display":
                        1 if merged.get(
                            "allow_kiosk_headcount_display"
                        ) else 0,
                    "allow_physical_headcount_display":
                        1 if merged.get(
                            "allow_physical_headcount_display"
                        ) else 0,
                    "updated_at":
                        merged["updated_at"],
                    "site_id": site_id,
                },
            )

        return fetch_site_summary(
            conn,
            site_id,
        )


@app.post(
    "/admin/sites/{site_id}/archive-and-release"
)
async def admin_archive_and_release_site(
    site_id: str,
    payload: ArchiveAndReleaseSiteRequest,
):
    with get_db() as conn:
        exists = fetch_one(
            conn,
            """
            SELECT site_id
            FROM sites
            WHERE site_id = :site_id
            """,
            {
                "site_id": site_id,
            },
        )

        if not exists:
            raise HTTPException(
                status_code=404,
                detail="Site not found.",
            )

        ensure_site_status(
            conn,
            site_id,
        )

        site = fetch_site_summary(
            conn,
            site_id,
        )

        if (
            site.get(
                "operational_state"
            )
            != "active"
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Site is already archived."
                ),
            )

        active_count = int(
            site.get(
                "active_count"
            )
            or 0
        )

        if active_count != 0:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Site cannot be archived "
                    "while people remain "
                    "signed in."
                ),
            )

        if bool(
            site.get("sos_active")
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Site cannot be archived "
                    "while SOS is active."
                ),
            )

        if (
            site.get("status")
            == "emergency"
            or site.get(
                "headcount_status"
            )
            == "alert"
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Site cannot be archived "
                    "while an emergency or "
                    "headcount alert is "
                    "unresolved."
                ),
            )

        now = utc_now_iso()

        current_kiosk_id = site.get(
            "kiosk_id"
        )

        last_kiosk_id = (
            current_kiosk_id
            or site.get(
                "last_kiosk_id"
            )
        )

        archive_note = (
            payload.archive_note
        )

        if archive_note is not None:
            archive_note = (
                archive_note.strip()
                or None
            )

        result = execute_write(
            conn,
            """
            UPDATE sites
            SET kiosk_id = NULL,
                operational_state =
                    'archived',
                archived_at =
                    :archived_at,
                unlinked_at =
                    :unlinked_at,
                archive_note =
                    :archive_note,
                last_kiosk_id =
                    :last_kiosk_id,
                updated_at =
                    :updated_at
            WHERE site_id = :site_id
              AND operational_state =
                    'active'
            """,
            {
                "archived_at": now,
                "unlinked_at": now,
                "archive_note":
                    archive_note,
                "last_kiosk_id":
                    last_kiosk_id,
                "updated_at": now,
                "site_id": site_id,
            },
        )

        if result.rowcount != 1:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Site lifecycle changed "
                    "before the archive action "
                    "completed. Refresh and "
                    "try again."
                ),
            )

        return fetch_site_summary(
            conn,
            site_id,
        )


@app.post("/admin/bind-kiosk")
async def admin_bind_kiosk(
    payload: BindKioskRequest,
):
    with get_db() as conn:
        site = fetch_one(
            conn,
            """
            SELECT *
            FROM sites
            WHERE site_id = :site_id
            """,
            {
                "site_id":
                    payload.site_id,
            },
        )

        if not site:
            raise HTTPException(
                status_code=404,
                detail="Site not found.",
            )

        if (
            site.get(
                "operational_state"
            )
            != "active"
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Archived site cannot "
                    "receive a kiosk. "
                    "Restore the site before "
                    "binding."
                ),
            )

        other = fetch_one(
            conn,
            """
            SELECT site_id
            FROM sites
            WHERE kiosk_id = :kiosk_id
              AND site_id <> :site_id
            """,
            {
                "kiosk_id":
                    payload.kiosk_id,
                "site_id":
                    payload.site_id,
            },
        )

        if other:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Kiosk ID is already "
                    "bound to another site."
                ),
            )

        result = execute_write(
            conn,
            """
            UPDATE sites
            SET kiosk_id = :kiosk_id,
                updated_at = :updated_at
            WHERE site_id = :site_id
              AND operational_state =
                    'active'
            """,
            {
                "kiosk_id":
                    payload.kiosk_id,
                "updated_at":
                    utc_now_iso(),
                "site_id":
                    payload.site_id,
            },
        )

        if result.rowcount != 1:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Site lifecycle changed "
                    "before the kiosk binding "
                    "completed. Refresh and "
                    "try again."
                ),
            )

        return {
            "status": "ok",
            "site_id":
                payload.site_id,
            "kiosk_id":
                payload.kiosk_id,
        }


@app.get(
    "/admin/sites/{site_id}/events"
)
async def admin_site_events(
    site_id: str,
    event_type: Optional[str] = None,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    before_occurred_at: Optional[str] = None,
    before_id: Optional[int] = Query(
        default=None,
        ge=1,
    ),
    from_timestamp: Optional[str] = Query(
        default=None,
        alias="from",
    ),
    to_timestamp: Optional[str] = Query(
        default=None,
        alias="to",
    ),
):
    """
    Return one stable raw-event page for a site.

    Backward-compatible requests may continue to use:

        event_type
        limit

    Phase 5G adds deliberate complete-history retrieval
    through a two-part keyset cursor:

        before_occurred_at
        before_id

    Optional from/to parameters are raw timestamp boundaries.
    The response remains the existing event array so current
    Admin and runtime consumers do not require an envelope
    migration.
    """
    cursor_timestamp_supplied = (
        before_occurred_at is not None
    )

    cursor_id_supplied = (
        before_id is not None
    )

    if (
        cursor_timestamp_supplied
        != cursor_id_supplied
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "before_occurred_at and "
                "before_id must be supplied "
                "together."
            ),
        )

    cursor_timestamp = parse_query_iso(
        before_occurred_at,
        "before_occurred_at",
    )

    from_value = parse_query_iso(
        from_timestamp,
        "from",
    )

    to_value = parse_query_iso(
        to_timestamp,
        "to",
    )

    if (
        from_value is not None
        and to_value is not None
        and from_value > to_value
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid event date range. "
                "The from timestamp must be "
                "earlier than or equal to the "
                "to timestamp."
            ),
        )

    with get_db() as conn:
        site = fetch_one(
            conn,
            """
            SELECT 1
            FROM sites
            WHERE site_id = :site_id
            """,
            {
                "site_id": site_id,
            },
        )

        if not site:
            raise HTTPException(
                status_code=404,
                detail="Site not found.",
            )

        sql = """
            SELECT
                id,
                site_id,
                device_id,
                event_type,
                occurred_at,
                payload_json,
                created_at
            FROM events
            WHERE site_id = :site_id
        """

        params: Dict[str, Any] = {
            "site_id": site_id,
        }

        if event_type:
            sql += (
                " AND event_type = "
                ":event_type"
            )

            params[
                "event_type"
            ] = event_type

        if from_value is not None:
            sql += (
                " AND occurred_at >= "
                ":from_timestamp"
            )

            params[
                "from_timestamp"
            ] = from_value

        if to_value is not None:
            sql += (
                " AND occurred_at <= "
                ":to_timestamp"
            )

            params[
                "to_timestamp"
            ] = to_value

        if cursor_timestamp is not None:
            sql += """
                AND (
                    occurred_at <
                        :before_occurred_at
                    OR (
                        occurred_at =
                            :before_occurred_at
                        AND id < :before_id
                    )
                )
            """

            params[
                "before_occurred_at"
            ] = cursor_timestamp

            params[
                "before_id"
            ] = before_id

        sql += (
            " ORDER BY "
            "occurred_at DESC, "
            "id DESC "
            "LIMIT :limit"
        )

        params["limit"] = limit

        rows = fetch_all(
            conn,
            sql,
            params,
        )

        result = []

        for row in rows:
            payload = json_loads(
                row["payload_json"]
            )

            result.append(
                {
                    "id":
                        row["id"],
                    "site_id":
                        row["site_id"],
                    "device_id":
                        row["device_id"],
                    "kiosk_id":
                        row["device_id"],
                    "event_type":
                        row["event_type"],
                    "timestamp":
                        row["occurred_at"],
                    "payload":
                        payload,
                    "worker_id":
                        payload.get(
                            "worker_id"
                        ),
                    "name":
                        payload.get(
                            "name"
                        ),
                    "company":
                        payload.get(
                            "company"
                        ),
                    "role":
                        payload.get(
                            "role"
                        ),
                    "description":
                        payload.get(
                            "description"
                        ),
                    "severity":
                        payload.get(
                            "severity"
                        ),
                    "injury":
                        payload.get(
                            "injury"
                        ),
                    "note":
                        payload.get(
                            "note"
                        ),
                }
            )

        return result


# =========================================================
# Admin - Crisis Control
# =========================================================

@app.get("/admin/crises")
async def admin_list_crises():
    with get_db() as conn:
        return list_active_crises(
            conn
        )


@app.post("/admin/crises")
async def admin_create_crisis(
    payload: CreateCrisisRequest,
):
    with get_db() as conn:
        crisis_id = (
            payload.crisis_id
            or (
                "crisis-"
                + datetime.now(
                    timezone.utc
                ).strftime(
                    "%Y%m%d%H%M%S"
                )
            )
        )

        now = utc_now_iso()

        exists = fetch_one(
            conn,
            """
            SELECT 1
            FROM crises
            WHERE crisis_id = :crisis_id
            """,
            {
                "crisis_id":
                    crisis_id,
            },
        )

        if exists:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Crisis ID already "
                    "exists."
                ),
            )

        execute_write(
            conn,
            """
            INSERT INTO crises (
                crisis_id,
                source,
                crisis_class,
                hazard_type,
                name,
                status,
                center_latitude,
                center_longitude,
                radius_km,
                created_at,
                updated_at,
                cleared_at
            )
            VALUES (
                :crisis_id,
                :source,
                :crisis_class,
                :hazard_type,
                :name,
                'active',
                :center_latitude,
                :center_longitude,
                :radius_km,
                :created_at,
                :updated_at,
                NULL
            )
            """,
            {
                "crisis_id":
                    crisis_id,
                "source":
                    payload.source,
                "crisis_class":
                    payload.crisis_class,
                "hazard_type":
                    payload.hazard_type,
                "name":
                    payload.name,
                "center_latitude":
                    payload.center_latitude,
                "center_longitude":
                    payload.center_longitude,
                "radius_km":
                    payload.radius_km,
                "created_at":
                    now,
                "updated_at":
                    now,
            },
        )

        return {
            "status": "ok",
            "crisis_id": crisis_id,
        }


@app.post(
    "/admin/crises/{crisis_id}/clear"
)
async def admin_clear_crisis(
    crisis_id: str,
):
    with get_db() as conn:
        exists = fetch_one(
            conn,
            """
            SELECT 1
            FROM crises
            WHERE crisis_id = :crisis_id
            """,
            {
                "crisis_id":
                    crisis_id,
            },
        )

        if not exists:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Crisis not found."
                ),
            )

        now = utc_now_iso()

        execute_write(
            conn,
            """
            UPDATE crises
            SET status = 'cleared',
                updated_at = :updated_at,
                cleared_at = :cleared_at
            WHERE crisis_id = :crisis_id
            """,
            {
                "updated_at": now,
                "cleared_at": now,
                "crisis_id":
                    crisis_id,
            },
        )

        return {
            "status": "ok",
            "crisis_id":
                crisis_id,
        }


# =========================================================
# NZ-NCM Feed
# =========================================================

@app.get("/ncm/sites")
async def ncm_sites():
    with get_db() as conn:
        sites = []

        crises = list_active_crises(
            conn
        )

        rows = fetch_all(
            conn,
            """
            SELECT
                s.site_id,
                s.name,
                s.address,
                s.city,
                s.company_name,
                s.kiosk_id,
                s.latitude,
                s.longitude,
                ss.status,
                ss.active_count,
                ss.headcount_status,
                ss.sos_active,
                ss.full_headcount_confirmed,
                ss.last_event_at,
                ss.last_sos_at,
                ss.last_full_headcount_at
            FROM sites s
            LEFT JOIN site_status ss
                ON ss.site_id = s.site_id
            WHERE s.operational_state =
                'active'
            ORDER BY s.created_at DESC
            """
        )

        for site in rows:
            classification = (
                classify_ncm_site_state(
                    site,
                    crises,
                )
            )

            if classification.get(
                "ncm_state"
            ):
                site.update(
                    classification
                )

                sites.append(site)

        sites.sort(
            key=lambda site: (
                site["priority"],
                site.get(
                    "last_event_at"
                )
                or "",
            )
        )

        return {
            "active_crises": crises,
            "sites": sites,
        }


# =========================================================
# Kiosk API
# =========================================================

@app.get("/kiosk/bootstrap/{kiosk_id}")
async def kiosk_bootstrap(
    kiosk_id: str,
):
    cleaned_kiosk_id = (
        kiosk_id or ""
    ).strip()

    if not cleaned_kiosk_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid kiosk_id.",
        )

    with get_db() as conn:
        binding = fetch_one(
            conn,
            """
            SELECT site_id
            FROM sites
            WHERE kiosk_id = :kiosk_id
              AND operational_state =
                    'active'
            """,
            {
                "kiosk_id":
                    cleaned_kiosk_id,
            },
        )

        if not binding:
            return {
                "kiosk_id":
                    cleaned_kiosk_id,
                "bound": False,
                "site": None,
                "status": None,
                "permissions": {
                    "allow_kiosk_headcount_display": False,
                    "allow_physical_headcount_display": False,
                },
                "ncm_state": None,
                "in_active_crisis_area":
                    False,
            }

        site_id = binding[
            "site_id"
        ]

        ensure_site_status(
            conn,
            site_id,
        )

        site = fetch_site_summary(
            conn,
            site_id,
        )

        crises = list_active_crises(
            conn
        )

        classification = (
            classify_ncm_site_state(
                site,
                crises,
            )
        )

        return {
            "kiosk_id":
                cleaned_kiosk_id,
            "bound": True,
            "site": {
                "site_id":
                    site["site_id"],
                "name":
                    site["name"],
                "site_policy_mode":
                    site.get(
                        "site_policy_mode"
                    )
                    or "standard",
                "operational_state":
                    site.get(
                        "operational_state"
                    )
                    or "active",
            },
            "status": {
                "status":
                    site.get(
                        "status"
                    )
                    or "idle",
                "active_count":
                    int(
                        site.get(
                            "active_count"
                        )
                        or 0
                    ),
                "headcount_status":
                    site.get(
                        "headcount_status"
                    )
                    or "pending",
                "sos_active": bool(
                    site.get(
                        "sos_active"
                    )
                ),
                "full_headcount_confirmed":
                    bool(
                        site.get(
                            "full_headcount_confirmed"
                        )
                    ),
                "last_event_at":
                    site.get(
                        "last_event_at"
                    ),
            },
            "permissions": {
                "allow_kiosk_headcount_display":
                    bool(
                        site.get(
                            "allow_kiosk_headcount_display"
                        )
                    ),
                "allow_physical_headcount_display":
                    bool(
                        site.get(
                            "allow_physical_headcount_display"
                        )
                    ),
            },
            "ncm_state":
                classification.get(
                    "ncm_state"
                ),
            "in_active_crisis_area":
                bool(
                    classification.get(
                        "in_active_crisis_area"
                    )
                ),
        }


@app.post("/api/signin")
async def api_signin(
    payload: SignEventRequest,
):
    sign_event_type = (
        normalise_sign_action(
            payload.type
        )
    )

    if (
        not payload.worker_id
        and not payload.name
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Minimum identity required: "
                "worker_id or name."
            ),
        )

    with get_db() as conn:
        site_id = resolve_site_id(
            conn,
            payload.site_id,
            payload.device_id,
        )

        occurred_at = parse_iso(
            payload.timestamp
        ).isoformat()

        event_payload = {
            "worker_id":
                payload.worker_id,
            "name":
                payload.name,
            "company":
                payload.company,
            "role":
                payload.role,
            "action":
                sign_event_type,
            "source":
                normalise_event_source(
                    payload.source
                ),
        }

        event_id = insert_event(
            conn,
            site_id=site_id,
            device_id=
                payload.device_id,
            event_type=
                sign_event_type,
            occurred_at=
                occurred_at,
            payload=
                event_payload,
        )

        state = recompute_site_state(
            conn,
            site_id,
        )

        return {
            "status": "received",
            "event_id": event_id,
            "site_id": site_id,
            **state,
        }


@app.post("/api/hazard")
async def api_hazard(
    payload: HazardEventRequest,
):
    with get_db() as conn:
        site_id = resolve_site_id(
            conn,
            payload.site_id,
            payload.device_id,
        )

        occurred_at = parse_iso(
            payload.timestamp
        ).isoformat()

        event_payload = {
            "worker_id":
                payload.worker_id,
            "name":
                payload.name,
            "description":
                payload.description,
            "severity":
                payload.severity,
            "source":
                normalise_event_source(
                    payload.source
                ),
        }

        event_id = insert_event(
            conn,
            site_id=site_id,
            device_id=
                payload.device_id,
            event_type="hazard",
            occurred_at=
                occurred_at,
            payload=
                event_payload,
        )

        state = recompute_site_state(
            conn,
            site_id,
        )

        return {
            "status": "received",
            "event_id": event_id,
            "site_id": site_id,
            **state,
        }


@app.post("/api/incident")
async def api_incident(
    payload: IncidentEventRequest,
):
    with get_db() as conn:
        site_id = resolve_site_id(
            conn,
            payload.site_id,
            payload.device_id,
        )

        occurred_at = parse_iso(
            payload.timestamp
        ).isoformat()

        event_payload = {
            "worker_id":
                payload.worker_id,
            "name":
                payload.name,
            "description":
                payload.description,
            "injury":
                payload.injury,
            "source":
                normalise_event_source(
                    payload.source
                ),
        }

        event_id = insert_event(
            conn,
            site_id=site_id,
            device_id=
                payload.device_id,
            event_type="incident",
            occurred_at=
                occurred_at,
            payload=
                event_payload,
        )

        state = recompute_site_state(
            conn,
            site_id,
        )

        return {
            "status": "received",
            "event_id": event_id,
            "site_id": site_id,
            **state,
        }


@app.post("/api/sos")
async def api_sos(
    payload: SOSEventRequest,
):
    with get_db() as conn:
        site_id = resolve_site_id(
            conn,
            payload.site_id,
            payload.device_id,
        )

        occurred_at = parse_iso(
            payload.timestamp
        ).isoformat()

        event_id = insert_event(
            conn,
            site_id=site_id,
            device_id=
                payload.device_id,
            event_type="sos",
            occurred_at=
                occurred_at,
            payload={
                "note": payload.note,
                "source":
                    normalise_event_source(
                        payload.source
                    ),
            },
        )

        state = recompute_site_state(
            conn,
            site_id,
        )

        return {
            "status": "received",
            "event_id": event_id,
            "site_id": site_id,
            **state,
        }


@app.post("/api/full-headcount")
async def api_full_headcount(
    payload: FullHeadcountEventRequest,
):
    with get_db() as conn:
        site_id = resolve_site_id(
            conn,
            payload.site_id,
            payload.device_id,
        )

        occurred_at = parse_iso(
            payload.timestamp
        ).isoformat()

        event_id = insert_event(
            conn,
            site_id=site_id,
            device_id=
                payload.device_id,
            event_type=
                "full_headcount",
            occurred_at=
                occurred_at,
            payload={
                "note": payload.note,
                "source":
                    normalise_event_source(
                        payload.source
                    ),
            },
        )

        state = recompute_site_state(
            conn,
            site_id,
        )

        return {
            "status": "received",
            "event_id": event_id,
            "site_id": site_id,
            **state,
        }