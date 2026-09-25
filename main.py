
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from math import radians, sin, cos, sqrt, atan2
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

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


# =========================================================
# H&I - Schema Init (HI.1)
# =========================================================

def ensure_site_hi_schema(
    conn: Connection,
) -> None:
    """
    Create the Site Hazards & Information persistence
    foundation without changing existing SignLog domains.

    HI.1 creates both the fast current projection and the
    immutable action-history table. Material action writes do
    not begin until HI.2.
    """
    execute_write(
        conn,
        """
        CREATE TABLE IF NOT EXISTS site_hi_state (
            site_id TEXT PRIMARY KEY,
            revision INTEGER NOT NULL DEFAULT 0,
            content_json TEXT NOT NULL,
            published_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(site_id)
                REFERENCES sites(site_id)
        )
        """,
    )

    if DB_MODE == "postgres":
        execute_write(
            conn,
            """
            CREATE TABLE IF NOT EXISTS site_hi_actions (
                id SERIAL PRIMARY KEY,
                site_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                item_id TEXT,
                actor_type TEXT NOT NULL,
                actor_ref TEXT,
                source_type TEXT NOT NULL,
                evidence_event_ids_json TEXT,
                action_payload_json TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                FOREIGN KEY(site_id)
                    REFERENCES sites(site_id)
            )
            """,
        )
    else:
        execute_write(
            conn,
            """
            CREATE TABLE IF NOT EXISTS site_hi_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                site_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                item_id TEXT,
                actor_type TEXT NOT NULL,
                actor_ref TEXT,
                source_type TEXT NOT NULL,
                evidence_event_ids_json TEXT,
                action_payload_json TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                FOREIGN KEY(site_id)
                    REFERENCES sites(site_id)
            )
            """,
        )

    execute_write(
        conn,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
            uq_site_hi_actions_site_revision
        ON site_hi_actions(site_id, revision)
        """,
    )

    execute_write(
        conn,
        """
        CREATE INDEX IF NOT EXISTS
            idx_site_hi_actions_site_id
        ON site_hi_actions(site_id)
        """,
    )

    execute_write(
        conn,
        """
        CREATE INDEX IF NOT EXISTS
            idx_site_hi_actions_site_time_id
        ON site_hi_actions(
            site_id,
            occurred_at,
            id
        )
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
        ensure_site_hi_schema(conn)

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


class IncidentEventRequest(BaseModel):
    site_id: Optional[str] = None

    device_id: str = Field(min_length=1)

    worker_id: Optional[str] = None
    name: Optional[str] = None

    description: str = Field(min_length=1)

    injury: bool = False

    timestamp: Optional[str] = None


class SOSEventRequest(BaseModel):
    site_id: Optional[str] = None

    device_id: str = Field(min_length=1)

    timestamp: Optional[str] = None
    note: Optional[str] = None


class FullHeadcountEventRequest(BaseModel):
    site_id: Optional[str] = None

    device_id: str = Field(min_length=1)

    timestamp: Optional[str] = None
    note: Optional[str] = None


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
# H&I Mutation Models (HI.2)
# =========================================================

class SiteHiItemMutationRequest(BaseModel):
    expected_revision: int = Field(ge=0)

    category: Literal[
        "current_hazard",
        "temporary_update",
        "site_information",
    ]

    title: str = Field(
        min_length=1,
        max_length=120,
    )
    message: str = Field(
        min_length=1,
        max_length=1500,
    )

    display_order: int = Field(
        ge=0,
        le=999,
    )

    valid_until: Optional[str] = None
    evidence_event_ids: Optional[List[int]] = None


class SiteHiClearRequest(BaseModel):
    expected_revision: int = Field(ge=0)

    # HI.2 correspondence correction:
    # clearing active field communication is a material action,
    # so a short reason is required even though it does NOT mean
    # the underlying hazard/evidence has been resolved.
    reason: str = Field(
        min_length=1,
        max_length=500,
    )


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


# =========================================================
# H&I - Authoritative Domain Helpers (HI.1 + HI.2)
# =========================================================

HI_CATEGORY_ORDER = {
    "current_hazard": 0,
    "temporary_update": 1,
    "site_information": 2,
}

HI_ITEM_SOURCE_TYPES = {
    "manual_operator",
    "evidence_linked",
}

HI_ACTION_TYPES = {
    "publish_item",
    "edit_item",
    "clear_item",
    "expire_items",
}

HI_ACTION_ACTOR_TYPES = {
    "admin_operator",
    "system",
}

HI_ACTION_SOURCE_TYPES = {
    "manual_operator",
    "evidence_linked",
    "system_expiry",
}


def require_hi_site(
    conn: Connection,
    site_id: str,
) -> Dict[str, Any]:
    site = fetch_one(
        conn,
        """
        SELECT
            site_id,
            operational_state,
            kiosk_id
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

    return site


def require_hi_mutation_site(
    conn: Connection,
    site_id: str,
) -> Dict[str, Any]:
    site = require_hi_site(
        conn,
        site_id,
    )

    if (
        site.get("operational_state")
        != "active"
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Archived Site H&I is read-only. "
                "Restore the Site before publishing "
                "or changing H&I."
            ),
        )

    return site


def empty_site_hi_snapshot(
    site_id: str,
) -> Dict[str, Any]:
    return {
        "site_id": site_id,
        "revision": 0,
        "published_at": None,
        "items": [],
    }


def hi_integrity_error(
    site_id: str,
    reason: str,
) -> HTTPException:
    return HTTPException(
        status_code=500,
        detail=(
            "H&I storage integrity error for "
            f"Site {site_id}: {reason}"
        ),
    )


def hi_json_dumps(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_hi_datetime(
    value: Any,
    *,
    field_name: str,
    site_id: Optional[str] = None,
    storage: bool = False,
) -> datetime:
    if not isinstance(value, str):
        if storage and site_id:
            raise hi_integrity_error(
                site_id,
                f"{field_name} is not a timestamp string.",
            )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name} timestamp.",
        )

    cleaned = value.strip()

    if not cleaned:
        if storage and site_id:
            raise hi_integrity_error(
                site_id,
                f"{field_name} is empty.",
            )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name} timestamp.",
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
        )

    except Exception as error:
        if storage and site_id:
            raise hi_integrity_error(
                site_id,
                f"{field_name} is not valid ISO time.",
            ) from error

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {field_name}. "
                "Provide a valid ISO timestamp."
            ),
        ) from error


def canonical_hi_items(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            HI_CATEGORY_ORDER[
                item["category"]
            ],
            item["display_order"],
            str(item.get("title") or ""),
            str(item.get("item_id") or ""),
        ),
    )


def validate_stored_hi_item(
    site_id: str,
    item: Dict[str, Any],
) -> None:
    item_id = item.get("item_id")
    category = item.get("category")
    title = item.get("title")
    message = item.get("message")
    display_order = item.get("display_order")
    source_type = item.get("source_type")
    evidence_event_ids = item.get(
        "evidence_event_ids"
    )

    if (
        not isinstance(item_id, str)
        or not item_id.strip()
        or not item_id.startswith("hi-")
    ):
        raise hi_integrity_error(
            site_id,
            "stored item has an invalid item_id.",
        )

    if category not in HI_CATEGORY_ORDER:
        raise hi_integrity_error(
            site_id,
            "stored item has an unknown category.",
        )

    if (
        not isinstance(title, str)
        or not title.strip()
        or len(title) > 120
    ):
        raise hi_integrity_error(
            site_id,
            "stored item has an invalid title.",
        )

    if (
        not isinstance(message, str)
        or not message.strip()
        or len(message) > 1500
    ):
        raise hi_integrity_error(
            site_id,
            "stored item has an invalid message.",
        )

    if (
        not isinstance(display_order, int)
        or isinstance(display_order, bool)
        or display_order < 0
        or display_order > 999
    ):
        raise hi_integrity_error(
            site_id,
            "stored item has an invalid display_order.",
        )

    if source_type not in HI_ITEM_SOURCE_TYPES:
        raise hi_integrity_error(
            site_id,
            "stored item has an invalid source_type.",
        )

    if not isinstance(evidence_event_ids, list):
        raise hi_integrity_error(
            site_id,
            "stored item evidence_event_ids must be an array.",
        )

    if len(evidence_event_ids) > 50:
        raise hi_integrity_error(
            site_id,
            "stored item has too many evidence references.",
        )

    seen_ids = set()

    for event_id in evidence_event_ids:
        if (
            not isinstance(event_id, int)
            or isinstance(event_id, bool)
            or event_id < 1
            or event_id in seen_ids
        ):
            raise hi_integrity_error(
                site_id,
                "stored item has invalid or duplicate evidence IDs.",
            )
        seen_ids.add(event_id)

    for field_name in (
        "valid_from",
        "created_at",
        "updated_at",
    ):
        parse_hi_datetime(
            item.get(field_name),
            field_name=field_name,
            site_id=site_id,
            storage=True,
        )

    valid_until = item.get("valid_until")

    if valid_until is not None:
        parse_hi_datetime(
            valid_until,
            field_name="valid_until",
            site_id=site_id,
            storage=True,
        )

    if (
        category == "temporary_update"
        and valid_until is None
    ):
        raise hi_integrity_error(
            site_id,
            "stored temporary update is missing valid_until.",
        )


def parse_site_hi_content(
    site_id: str,
    value: Optional[str],
) -> List[Dict[str, Any]]:
    """
    H&I current-state JSON is authority-bearing data.
    Malformed H&I storage must never be silently coerced into
    an empty state.
    """
    if value is None:
        raise hi_integrity_error(
            site_id,
            "content_json is missing.",
        )

    try:
        decoded = json.loads(value)
    except Exception as error:
        raise hi_integrity_error(
            site_id,
            "content_json is not valid JSON.",
        ) from error

    if not isinstance(decoded, dict):
        raise hi_integrity_error(
            site_id,
            "content_json must contain an object.",
        )

    items = decoded.get("items")

    if not isinstance(items, list):
        raise hi_integrity_error(
            site_id,
            "content_json.items must be an array.",
        )

    for item in items:
        if not isinstance(item, dict):
            raise hi_integrity_error(
                site_id,
                "every H&I item must be an object.",
            )

        validate_stored_hi_item(
            site_id,
            item,
        )

    return canonical_hi_items(
        items
    )


def load_site_hi_snapshot_raw(
    conn: Connection,
    site_id: str,
) -> Dict[str, Any]:
    row = fetch_one(
        conn,
        """
        SELECT
            site_id,
            revision,
            content_json,
            published_at,
            created_at,
            updated_at
        FROM site_hi_state
        WHERE site_id = :site_id
        """,
        {
            "site_id": site_id,
        },
    )

    if not row:
        return empty_site_hi_snapshot(
            site_id
        )

    revision = row.get("revision")

    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < 0
    ):
        raise hi_integrity_error(
            site_id,
            "revision is invalid.",
        )

    items = parse_site_hi_content(
        site_id,
        row.get("content_json"),
    )

    published_at = row.get(
        "published_at"
    )

    if revision == 0:
        if published_at is not None:
            raise hi_integrity_error(
                site_id,
                "revision 0 cannot have published_at.",
            )
    else:
        parse_hi_datetime(
            published_at,
            field_name="published_at",
            site_id=site_id,
            storage=True,
        )

    return {
        "site_id": site_id,
        "revision": revision,
        "published_at": published_at,
        "items": items,
    }


def normalise_hi_evidence_ids(
    values: Optional[List[int]],
) -> List[int]:
    if values is None:
        return []

    if len(values) > 50:
        raise HTTPException(
            status_code=400,
            detail=(
                "H&I evidence_event_ids may contain "
                "at most 50 event IDs."
            ),
        )

    result: List[int] = []
    seen = set()

    for value in values:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "H&I evidence_event_ids must contain "
                    "positive event IDs."
                ),
            )

        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def validate_hi_evidence_refs(
    conn: Connection,
    site_id: str,
    event_ids: List[int],
) -> None:
    for event_id in event_ids:
        event = fetch_one(
            conn,
            """
            SELECT id, site_id
            FROM events
            WHERE id = :event_id
            """,
            {
                "event_id": event_id,
            },
        )

        if not event:
            raise HTTPException(
                status_code=400,
                detail=(
                    "H&I evidence reference "
                    f"{event_id} does not exist."
                ),
            )

        if event.get("site_id") != site_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    "H&I evidence reference "
                    f"{event_id} does not belong to "
                    f"Site {site_id}."
                ),
            )


def build_hi_item_from_payload(
    conn: Connection,
    site_id: str,
    payload: SiteHiItemMutationRequest,
    *,
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    title = payload.title.strip()
    message = payload.message.strip()

    if not title:
        raise HTTPException(
            status_code=400,
            detail="H&I title cannot be blank.",
        )

    if not message:
        raise HTTPException(
            status_code=400,
            detail="H&I message cannot be blank.",
        )

    now_dt = datetime.now(
        timezone.utc
    )
    now = now_dt.isoformat()

    valid_until: Optional[str] = None

    if payload.valid_until is not None:
        parsed_until = parse_hi_datetime(
            payload.valid_until,
            field_name="valid_until",
        )

        if parsed_until <= now_dt:
            raise HTTPException(
                status_code=400,
                detail=(
                    "H&I valid_until must be later "
                    "than Backend publication time."
                ),
            )

        valid_until = parsed_until.isoformat()

    if (
        payload.category == "temporary_update"
        and valid_until is None
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Temporary Site Update requires "
                "valid_until."
            ),
        )

    evidence_event_ids = (
        normalise_hi_evidence_ids(
            payload.evidence_event_ids
        )
    )

    validate_hi_evidence_refs(
        conn,
        site_id,
        evidence_event_ids,
    )

    source_type = (
        "evidence_linked"
        if evidence_event_ids
        else "manual_operator"
    )

    item_id = (
        existing["item_id"]
        if existing
        else f"hi-{uuid4()}"
    )

    created_at = (
        existing["created_at"]
        if existing
        else now
    )

    # An edit is a republished current version. created_at keeps
    # original identity; valid_from records when this version
    # became effective.
    item = {
        "item_id": item_id,
        "category": payload.category,
        "title": title,
        "message": message,
        "display_order": payload.display_order,
        "valid_from": now,
        "valid_until": valid_until,
        "source_type": source_type,
        "evidence_event_ids":
            evidence_event_ids,
        "created_at": created_at,
        "updated_at": now,
    }

    if existing:
        comparable_fields = (
            "category",
            "title",
            "message",
            "display_order",
            "valid_until",
            "source_type",
            "evidence_event_ids",
        )

        if all(
            existing.get(field_name)
            == item.get(field_name)
            for field_name in comparable_fields
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "H&I edit contains no material change."
                ),
            )

    return item


def require_expected_hi_revision(
    snapshot: Dict[str, Any],
    expected_revision: int,
) -> None:
    if snapshot["revision"] != expected_revision:
        raise HTTPException(
            status_code=409,
            detail=(
                "H&I revision conflict. Refresh current "
                "Site H&I and retry."
            ),
        )


def find_current_hi_item(
    snapshot: Dict[str, Any],
    item_id: str,
) -> Dict[str, Any]:
    for item in snapshot["items"]:
        if item.get("item_id") == item_id:
            return item

    raise HTTPException(
        status_code=404,
        detail=(
            "H&I item is not present in the current "
            "Site projection."
        ),
    )


def insert_site_hi_action(
    conn: Connection,
    *,
    site_id: str,
    revision: int,
    action_type: str,
    item_id: Optional[str],
    actor_type: str,
    actor_ref: Optional[str],
    source_type: str,
    evidence_event_ids: List[int],
    action_payload: Dict[str, Any],
    snapshot: Dict[str, Any],
    occurred_at: str,
) -> None:
    if action_type not in HI_ACTION_TYPES:
        raise ValueError(
            "Invalid internal H&I action_type."
        )

    if actor_type not in HI_ACTION_ACTOR_TYPES:
        raise ValueError(
            "Invalid internal H&I actor_type."
        )

    if source_type not in HI_ACTION_SOURCE_TYPES:
        raise ValueError(
            "Invalid internal H&I source_type."
        )

    execute_write(
        conn,
        """
        INSERT INTO site_hi_actions (
            site_id,
            revision,
            action_type,
            item_id,
            actor_type,
            actor_ref,
            source_type,
            evidence_event_ids_json,
            action_payload_json,
            snapshot_json,
            occurred_at
        )
        VALUES (
            :site_id,
            :revision,
            :action_type,
            :item_id,
            :actor_type,
            :actor_ref,
            :source_type,
            :evidence_event_ids_json,
            :action_payload_json,
            :snapshot_json,
            :occurred_at
        )
        """,
        {
            "site_id": site_id,
            "revision": revision,
            "action_type": action_type,
            "item_id": item_id,
            "actor_type": actor_type,
            "actor_ref": actor_ref,
            "source_type": source_type,
            "evidence_event_ids_json":
                hi_json_dumps(
                    evidence_event_ids
                ),
            "action_payload_json":
                hi_json_dumps(
                    action_payload
                ),
            "snapshot_json":
                hi_json_dumps(
                    snapshot
                ),
            "occurred_at": occurred_at,
        },
    )


def persist_site_hi_revision(
    conn: Connection,
    *,
    site_id: str,
    expected_revision: int,
    items: List[Dict[str, Any]],
    action_type: str,
    item_id: Optional[str],
    actor_type: str,
    actor_ref: Optional[str],
    source_type: str,
    evidence_event_ids: List[int],
    action_payload: Dict[str, Any],
    occurred_at: str,
) -> Dict[str, Any]:
    new_revision = expected_revision + 1
    canonical_items = canonical_hi_items(
        items
    )

    snapshot = {
        "site_id": site_id,
        "revision": new_revision,
        "published_at": occurred_at,
        "items": canonical_items,
    }

    content_json = hi_json_dumps(
        {
            "items": canonical_items,
        }
    )

    result = execute_write(
        conn,
        """
        UPDATE site_hi_state
        SET revision = :new_revision,
            content_json = :content_json,
            published_at = :published_at,
            updated_at = :updated_at
        WHERE site_id = :site_id
          AND revision = :expected_revision
        """,
        {
            "new_revision": new_revision,
            "content_json": content_json,
            "published_at": occurred_at,
            "updated_at": occurred_at,
            "site_id": site_id,
            "expected_revision":
                expected_revision,
        },
    )

    if (
        result.rowcount == 0
        and expected_revision == 0
    ):
        result = execute_write(
            conn,
            """
            INSERT INTO site_hi_state (
                site_id,
                revision,
                content_json,
                published_at,
                created_at,
                updated_at
            )
            VALUES (
                :site_id,
                :revision,
                :content_json,
                :published_at,
                :created_at,
                :updated_at
            )
            ON CONFLICT(site_id) DO NOTHING
            """,
            {
                "site_id": site_id,
                "revision": new_revision,
                "content_json": content_json,
                "published_at": occurred_at,
                "created_at": occurred_at,
                "updated_at": occurred_at,
            },
        )

    if result.rowcount != 1:
        raise HTTPException(
            status_code=409,
            detail=(
                "H&I revision conflict. Refresh current "
                "Site H&I and retry."
            ),
        )

    insert_site_hi_action(
        conn,
        site_id=site_id,
        revision=new_revision,
        action_type=action_type,
        item_id=item_id,
        actor_type=actor_type,
        actor_ref=actor_ref,
        source_type=source_type,
        evidence_event_ids=evidence_event_ids,
        action_payload=action_payload,
        snapshot=snapshot,
        occurred_at=occurred_at,
    )

    return snapshot


def materialize_expired_hi_items(
    conn: Connection,
    site_id: str,
) -> Dict[str, Any]:
    """
    Apply already-authorised valid_until rules lazily.

    HI.2 correspondence correction:
    expiry is a Backend system action, not an admin_operator
    action. The action trail therefore records actor_type=system
    and source_type=system_expiry so the audit chain never claims
    a human cleared content they did not touch.
    """
    for _ in range(5):
        snapshot = load_site_hi_snapshot_raw(
            conn,
            site_id,
        )

        if snapshot["revision"] == 0:
            return snapshot

        now_dt = datetime.now(
            timezone.utc
        )
        expired_items: List[Dict[str, Any]] = []
        retained_items: List[Dict[str, Any]] = []

        for item in snapshot["items"]:
            valid_until = item.get(
                "valid_until"
            )

            if valid_until is None:
                retained_items.append(
                    item
                )
                continue

            expiry_dt = parse_hi_datetime(
                valid_until,
                field_name="valid_until",
                site_id=site_id,
                storage=True,
            )

            if expiry_dt <= now_dt:
                expired_items.append(
                    item
                )
            else:
                retained_items.append(
                    item
                )

        if not expired_items:
            return snapshot

        occurred_at = utc_now_iso()
        expected_revision = (
            snapshot["revision"]
        )
        new_revision = (
            expected_revision + 1
        )
        canonical_retained = (
            canonical_hi_items(
                retained_items
            )
        )

        next_snapshot = {
            "site_id": site_id,
            "revision": new_revision,
            "published_at": occurred_at,
            "items": canonical_retained,
        }

        result = execute_write(
            conn,
            """
            UPDATE site_hi_state
            SET revision = :new_revision,
                content_json = :content_json,
                published_at = :published_at,
                updated_at = :updated_at
            WHERE site_id = :site_id
              AND revision = :expected_revision
            """,
            {
                "new_revision": new_revision,
                "content_json":
                    hi_json_dumps(
                        {
                            "items":
                                canonical_retained,
                        }
                    ),
                "published_at": occurred_at,
                "updated_at": occurred_at,
                "site_id": site_id,
                "expected_revision":
                    expected_revision,
            },
        )

        if result.rowcount != 1:
            # Another transaction changed H&I first. Reload and
            # reassess instead of turning a read into a false
            # operator conflict.
            continue

        evidence_ids: List[int] = []
        seen_ids = set()

        for item in expired_items:
            for event_id in item.get(
                "evidence_event_ids",
                [],
            ):
                if event_id not in seen_ids:
                    seen_ids.add(event_id)
                    evidence_ids.append(
                        event_id
                    )

        insert_site_hi_action(
            conn,
            site_id=site_id,
            revision=new_revision,
            action_type="expire_items",
            item_id=None,
            actor_type="system",
            actor_ref=None,
            source_type="system_expiry",
            evidence_event_ids=evidence_ids,
            action_payload={
                "expired_item_ids": [
                    item["item_id"]
                    for item in expired_items
                ],
                "expired_items": expired_items,
                "materialized_at": occurred_at,
                "rule": (
                    "valid_until <= backend_utc_now"
                ),
            },
            snapshot=next_snapshot,
            occurred_at=occurred_at,
        )

        return next_snapshot

    raise HTTPException(
        status_code=409,
        detail=(
            "H&I changed repeatedly while expiry was being "
            "materialised. Retry the read."
        ),
    )


def load_site_hi_snapshot(
    conn: Connection,
    site_id: str,
) -> Dict[str, Any]:
    return materialize_expired_hi_items(
        conn,
        site_id,
    )


def parse_hi_action_json(
    site_id: str,
    field_name: str,
    value: Optional[str],
    expected_type: type,
) -> Any:
    if value is None:
        if expected_type is list:
            return []
        raise hi_integrity_error(
            site_id,
            f"{field_name} is missing.",
        )

    try:
        decoded = json.loads(value)
    except Exception as error:
        raise hi_integrity_error(
            site_id,
            f"{field_name} is not valid JSON.",
        ) from error

    if not isinstance(decoded, expected_type):
        raise hi_integrity_error(
            site_id,
            f"{field_name} has the wrong JSON shape.",
        )

    return decoded


def site_hi_action_from_row(
    site_id: str,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    action_id = row.get("id")
    revision = row.get("revision")
    action_type = row.get(
        "action_type"
    )
    actor_type = row.get(
        "actor_type"
    )
    source_type = row.get(
        "source_type"
    )
    item_id = row.get("item_id")

    if (
        not isinstance(action_id, int)
        or isinstance(action_id, bool)
        or action_id < 1
    ):
        raise hi_integrity_error(
            site_id,
            "action history contains invalid action ID.",
        )

    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < 1
    ):
        raise hi_integrity_error(
            site_id,
            "action history contains invalid revision.",
        )

    if action_type not in HI_ACTION_TYPES:
        raise hi_integrity_error(
            site_id,
            "action history contains unknown action_type.",
        )

    if actor_type not in HI_ACTION_ACTOR_TYPES:
        raise hi_integrity_error(
            site_id,
            "action history contains unknown actor_type.",
        )

    if source_type not in HI_ACTION_SOURCE_TYPES:
        raise hi_integrity_error(
            site_id,
            "action history contains unknown source_type.",
        )

    if action_type == "expire_items":
        if item_id is not None:
            raise hi_integrity_error(
                site_id,
                "expire_items action must not claim one item_id.",
            )
    elif (
        not isinstance(item_id, str)
        or not item_id.startswith("hi-")
    ):
        raise hi_integrity_error(
            site_id,
            "material item action has invalid item_id.",
        )

    parse_hi_datetime(
        row.get("occurred_at"),
        field_name="occurred_at",
        site_id=site_id,
        storage=True,
    )

    evidence_event_ids = (
        parse_hi_action_json(
            site_id,
            "evidence_event_ids_json",
            row.get(
                "evidence_event_ids_json"
            ),
            list,
        )
    )

    if len(evidence_event_ids) > 50:
        raise hi_integrity_error(
            site_id,
            "action history contains too many evidence IDs.",
        )

    seen_ids = set()
    for event_id in evidence_event_ids:
        if (
            not isinstance(event_id, int)
            or isinstance(event_id, bool)
            or event_id < 1
            or event_id in seen_ids
        ):
            raise hi_integrity_error(
                site_id,
                "action history contains invalid evidence IDs.",
            )
        seen_ids.add(event_id)

    action_payload = parse_hi_action_json(
        site_id,
        "action_payload_json",
        row.get("action_payload_json"),
        dict,
    )

    snapshot = parse_hi_action_json(
        site_id,
        "snapshot_json",
        row.get("snapshot_json"),
        dict,
    )

    if snapshot.get("site_id") != site_id:
        raise hi_integrity_error(
            site_id,
            "action snapshot Site identity does not match action row.",
        )

    if snapshot.get("revision") != revision:
        raise hi_integrity_error(
            site_id,
            "action snapshot revision does not match action row.",
        )

    parse_hi_datetime(
        snapshot.get("published_at"),
        field_name="snapshot.published_at",
        site_id=site_id,
        storage=True,
    )

    snapshot_items = snapshot.get("items")
    if not isinstance(snapshot_items, list):
        raise hi_integrity_error(
            site_id,
            "action snapshot items must be an array.",
        )

    for snapshot_item in snapshot_items:
        if not isinstance(snapshot_item, dict):
            raise hi_integrity_error(
                site_id,
                "action snapshot contains invalid item shape.",
            )
        validate_stored_hi_item(
            site_id,
            snapshot_item,
        )

    snapshot["items"] = canonical_hi_items(
        snapshot_items
    )

    return {
        "id": action_id,
        "site_id": site_id,
        "revision": revision,
        "action_type": action_type,
        "item_id": item_id,
        "actor_type": actor_type,
        "actor_ref": row.get("actor_ref"),
        "source_type": source_type,
        "evidence_event_ids":
            evidence_event_ids,
        "action_payload": action_payload,
        "snapshot": snapshot,
        "occurred_at": row.get(
            "occurred_at"
        ),
    }


def resolve_kiosk_hi_site_id(
    conn: Connection,
    kiosk_id: str,
) -> str:
    cleaned_kiosk_id = (
        kiosk_id or ""
    ).strip()

    if not cleaned_kiosk_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid kiosk_id.",
        )

    site = fetch_one(
        conn,
        """
        SELECT site_id
        FROM sites
        WHERE kiosk_id = :kiosk_id
          AND operational_state = 'active'
        """,
        {
            "kiosk_id": cleaned_kiosk_id,
        },
    )

    if not site:
        raise HTTPException(
            status_code=404,
            detail=(
                "Kiosk is not bound to an "
                "active Site."
            ),
        )

    return str(site["site_id"])


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
# Admin - Site H&I Authority (HI.1 + HI.2)
# =========================================================

@app.get(
    "/admin/sites/{site_id}/hi"
)
async def admin_site_hi(
    site_id: str,
):
    with get_db() as conn:
        # Archived Sites remain readable. Lazy expiry is a
        # Backend materialisation of a pre-authorised time rule,
        # not an Admin mutation.
        require_hi_site(
            conn,
            site_id,
        )

        return load_site_hi_snapshot(
            conn,
            site_id,
        )


@app.get(
    "/admin/sites/{site_id}/hi/actions"
)
async def admin_site_hi_actions(
    site_id: str,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    before_revision: Optional[int] = Query(
        default=None,
        ge=1,
    ),
):
    with get_db() as conn:
        require_hi_site(
            conn,
            site_id,
        )

        # "Every H&I read" includes history reads. Materialise
        # any due expiry first so the returned action trail is
        # current and complete to the read point.
        load_site_hi_snapshot(
            conn,
            site_id,
        )

        clauses = [
            "site_id = :site_id"
        ]
        params: Dict[str, Any] = {
            "site_id": site_id,
            "limit": limit,
        }

        if before_revision is not None:
            clauses.append(
                "revision < :before_revision"
            )
            params[
                "before_revision"
            ] = before_revision

        rows = fetch_all(
            conn,
            f"""
            SELECT
                id,
                site_id,
                revision,
                action_type,
                item_id,
                actor_type,
                actor_ref,
                source_type,
                evidence_event_ids_json,
                action_payload_json,
                snapshot_json,
                occurred_at
            FROM site_hi_actions
            WHERE {' AND '.join(clauses)}
            ORDER BY revision DESC
            LIMIT :limit
            """,
            params,
        )

        return [
            site_hi_action_from_row(
                site_id,
                row,
            )
            for row in rows
        ]


@app.post(
    "/admin/sites/{site_id}/hi/items"
)
async def admin_create_site_hi_item(
    site_id: str,
    payload: SiteHiItemMutationRequest,
):
    with get_db() as conn:
        require_hi_mutation_site(
            conn,
            site_id,
        )

        snapshot = load_site_hi_snapshot(
            conn,
            site_id,
        )

        require_expected_hi_revision(
            snapshot,
            payload.expected_revision,
        )

        item = build_hi_item_from_payload(
            conn,
            site_id,
            payload,
        )

        occurred_at = item[
            "updated_at"
        ]

        return persist_site_hi_revision(
            conn,
            site_id=site_id,
            expected_revision=
                snapshot["revision"],
            items=[
                *snapshot["items"],
                item,
            ],
            action_type="publish_item",
            item_id=item["item_id"],
            actor_type="admin_operator",
            actor_ref=None,
            source_type=
                item["source_type"],
            evidence_event_ids=
                item[
                    "evidence_event_ids"
                ],
            action_payload={
                "after": item,
            },
            occurred_at=occurred_at,
        )


@app.put(
    "/admin/sites/{site_id}/hi/items/{item_id}"
)
async def admin_update_site_hi_item(
    site_id: str,
    item_id: str,
    payload: SiteHiItemMutationRequest,
):
    with get_db() as conn:
        require_hi_mutation_site(
            conn,
            site_id,
        )

        snapshot = load_site_hi_snapshot(
            conn,
            site_id,
        )

        require_expected_hi_revision(
            snapshot,
            payload.expected_revision,
        )

        existing = find_current_hi_item(
            snapshot,
            item_id,
        )

        item = build_hi_item_from_payload(
            conn,
            site_id,
            payload,
            existing=existing,
        )

        next_items = [
            item
            if current.get("item_id")
            == item_id
            else current
            for current in snapshot["items"]
        ]

        occurred_at = item[
            "updated_at"
        ]

        return persist_site_hi_revision(
            conn,
            site_id=site_id,
            expected_revision=
                snapshot["revision"],
            items=next_items,
            action_type="edit_item",
            item_id=item_id,
            actor_type="admin_operator",
            actor_ref=None,
            source_type=
                item["source_type"],
            evidence_event_ids=
                item[
                    "evidence_event_ids"
                ],
            action_payload={
                "before": existing,
                "after": item,
            },
            occurred_at=occurred_at,
        )


@app.post(
    "/admin/sites/{site_id}/hi/items/{item_id}/clear"
)
async def admin_clear_site_hi_item(
    site_id: str,
    item_id: str,
    payload: SiteHiClearRequest,
):
    with get_db() as conn:
        require_hi_mutation_site(
            conn,
            site_id,
        )

        snapshot = load_site_hi_snapshot(
            conn,
            site_id,
        )

        require_expected_hi_revision(
            snapshot,
            payload.expected_revision,
        )

        existing = find_current_hi_item(
            snapshot,
            item_id,
        )

        reason = payload.reason.strip()

        if not reason:
            raise HTTPException(
                status_code=400,
                detail=(
                    "H&I clear reason cannot be blank."
                ),
            )

        next_items = [
            current
            for current in snapshot["items"]
            if current.get("item_id")
            != item_id
        ]

        occurred_at = utc_now_iso()

        return persist_site_hi_revision(
            conn,
            site_id=site_id,
            expected_revision=
                snapshot["revision"],
            items=next_items,
            action_type="clear_item",
            item_id=item_id,
            actor_type="admin_operator",
            actor_ref=None,
            source_type="manual_operator",
            evidence_event_ids=list(
                existing.get(
                    "evidence_event_ids",
                    [],
                )
            ),
            action_payload={
                "before": existing,
                "reason": reason,
                "resolution_claimed": False,
            },
            occurred_at=occurred_at,
        )


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
            in_crisis_area = (
                site_in_any_active_crisis(
                    site,
                    crises,
                )
            )

            ncm_state = None
            priority = None

            if site.get(
                "sos_active"
            ):
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

            if ncm_state:
                site[
                    "ncm_state"
                ] = ncm_state

                site[
                    "priority"
                ] = priority

                site[
                    "in_active_crisis_area"
                ] = in_crisis_area

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

@app.get(
    "/api/kiosks/{kiosk_id}/hi"
)
async def api_kiosk_hi(
    kiosk_id: str,
):
    with get_db() as conn:
        site_id = resolve_kiosk_hi_site_id(
            conn,
            kiosk_id,
        )

        return load_site_hi_snapshot(
            conn,
            site_id,
        )


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





