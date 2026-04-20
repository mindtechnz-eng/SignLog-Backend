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
from sqlalchemy import create_engine, text
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


def normalise_sign_action(value: str) -> Literal["signin", "signout"]:
    v = (value or "").strip().lower()
    if v in {"in", "signin", "sign-in", "sign_in"}:
        return "signin"
    if v in {"out", "signout", "sign-out", "sign_out"}:
        return "signout"
    raise HTTPException(status_code=400, detail="Invalid sign action. Use in/out or signin/signout.")


def json_dumps(data: Any) -> str:
    return json.dumps(data or {}, ensure_ascii=False)


def json_loads(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        raw = json.loads(value)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


@contextmanager
def get_db():
    with ENGINE.begin() as conn:
        yield conn


def fetch_one(conn: Connection, sql: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    result: Result = conn.execute(text(sql), params or {})
    row = result.mappings().first()
    return dict(row) if row else None


def fetch_all(conn: Connection, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    result: Result = conn.execute(text(sql), params or {})
    return [dict(row) for row in result.mappings().all()]


def execute_write(conn: Connection, sql: str, params: Optional[Dict[str, Any]] = None) -> Result:
    return conn.execute(text(sql), params or {})


# =========================================================
# Database Init
# =========================================================

def init_db() -> None:
    with get_db() as conn:
        if DB_MODE == "sqlite":
            conn.execute(text("PRAGMA journal_mode=WAL;"))

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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
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
                    FOREIGN KEY(site_id) REFERENCES sites(site_id)
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
                        FOREIGN KEY(site_id) REFERENCES sites(site_id)
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
                        FOREIGN KEY(site_id) REFERENCES sites(site_id)
                    )
                    """
                )
            )

        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_site_id ON events(site_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type)"))

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
    site_policy_mode: Literal["lenient", "standard", "strict"] = "standard"


class UpdateSiteRequest(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    company_name: Optional[str] = None
    kiosk_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    site_policy_mode: Optional[Literal["lenient", "standard", "strict"]] = None


class BindKioskRequest(BaseModel):
    kiosk_id: str = Field(min_length=1)
    site_id: str = Field(min_length=1)


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
    severity: Literal["low", "medium", "high"] = "medium"
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
    source: Literal["manual", "official_feed", "inferred_cluster"] = "manual"
    crisis_class: Literal["local_crisis", "regional_crisis", "soe"]
    hazard_type: Optional[str] = None
    name: str = Field(min_length=1)
    center_latitude: float
    center_longitude: float
    radius_km: float = Field(gt=0)


# =========================================================
# DB-level Functions
# =========================================================

def next_site_id(conn: Connection) -> str:
    row = fetch_one(
        conn,
        """
        SELECT site_id
        FROM sites
        WHERE site_id LIKE :pattern
        ORDER BY site_id DESC
        LIMIT 1
        """,
        {"pattern": "site-%"},
    )

    if not row:
        return "site-001"

    last = row["site_id"]
    try:
        n = int(last.split("-")[1]) + 1
    except Exception:
        n = 1
    return f"site-{n:03d}"


def ensure_site_status(conn: Connection, site_id: str) -> None:
    exists = fetch_one(
        conn,
        "SELECT 1 FROM site_status WHERE site_id = :site_id",
        {"site_id": site_id},
    )
    if not exists:
        execute_write(
            conn,
            """
            INSERT INTO site_status (
                site_id, status, active_count, headcount_status,
                sos_active, full_headcount_confirmed,
                last_event_at, last_sos_at, last_full_headcount_at
            )
            VALUES (:site_id, 'idle', 0, 'pending', 0, 0, NULL, NULL, NULL)
            """,
            {"site_id": site_id},
        )


def resolve_site_id(conn: Connection, site_id: Optional[str], device_id: str) -> str:
    row = fetch_one(
        conn,
        "SELECT site_id FROM sites WHERE kiosk_id = :device_id",
        {"device_id": device_id},
    )
    if row:
        return row["site_id"]

    if site_id:
        row = fetch_one(
            conn,
            "SELECT site_id FROM sites WHERE site_id = :site_id",
            {"site_id": site_id},
        )
        if row:
            return row["site_id"]

    raise HTTPException(
        status_code=400,
        detail="No site resolved. Bind kiosk to site or provide valid site_id."
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

    if DB_MODE == "postgres":
        result = execute_write(
            conn,
            """
            INSERT INTO events (
                site_id, device_id, event_type, occurred_at, payload_json, created_at
            )
            VALUES (:site_id, :device_id, :event_type, :occurred_at, :payload_json, :created_at)
            RETURNING id
            """,
            {
                "site_id": site_id,
                "device_id": device_id,
                "event_type": event_type,
                "occurred_at": occurred_at,
                "payload_json": json_dumps(payload),
                "created_at": created_at,
            },
        )
        return int(result.scalar_one())

    result = execute_write(
        conn,
        """
        INSERT INTO events (
            site_id, device_id, event_type, occurred_at, payload_json, created_at
        )
        VALUES (:site_id, :device_id, :event_type, :occurred_at, :payload_json, :created_at)
        """,
        {
            "site_id": site_id,
            "device_id": device_id,
            "event_type": event_type,
            "occurred_at": occurred_at,
            "payload_json": json_dumps(payload),
            "created_at": created_at,
        },
    )
    return int(result.lastrowid)


def recompute_site_state(conn: Connection, site_id: str) -> Dict[str, Any]:
    rows = fetch_all(
        conn,
        """
        SELECT event_type, occurred_at
        FROM events
        WHERE site_id = :site_id
        ORDER BY occurred_at ASC, id ASC
        """,
        {"site_id": site_id},
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
            active_count = max(0, active_count - 1)
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
        headcount_status = "safe" if full_headcount_confirmed else "pending"
    else:
        status = "idle"
        headcount_status = "safe" if full_headcount_confirmed else "pending"

    ensure_site_status(conn, site_id)

    execute_write(
        conn,
        """
        UPDATE site_status
        SET status = :status,
            active_count = :active_count,
            headcount_status = :headcount_status,
            sos_active = :sos_active,
            full_headcount_confirmed = :full_headcount_confirmed,
            last_event_at = :last_event_at,
            last_sos_at = :last_sos_at,
            last_full_headcount_at = :last_full_headcount_at
        WHERE site_id = :site_id
        """,
        {
            "status": status,
            "active_count": active_count,
            "headcount_status": headcount_status,
            "sos_active": sos_active,
            "full_headcount_confirmed": full_headcount_confirmed,
            "last_event_at": last_event_at,
            "last_sos_at": last_sos_at,
            "last_full_headcount_at": last_full_headcount_at,
            "site_id": site_id,
        },
    )

    return {
        "status": status,
        "active_count": active_count,
        "headcount_status": headcount_status,
        "sos_active": bool(sos_active),
        "full_headcount_confirmed": bool(full_headcount_confirmed),
        "last_event_at": last_event_at,
        "last_sos_at": last_sos_at,
        "last_full_headcount_at": last_full_headcount_at,
    }


def fetch_site_summary(conn: Connection, site_id: str) -> Dict[str, Any]:
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
            ss.status,
            ss.active_count,
            ss.headcount_status,
            ss.sos_active,
            ss.full_headcount_confirmed,
            ss.last_event_at,
            ss.last_sos_at,
            ss.last_full_headcount_at
        FROM sites s
        LEFT JOIN site_status ss ON ss.site_id = s.site_id
        WHERE s.site_id = :site_id
        """,
        {"site_id": site_id},
    )

    if not row:
        raise HTTPException(status_code=404, detail="Site not found.")

    return row


def list_active_crises(conn: Connection) -> List[Dict[str, Any]]:
    return fetch_all(
        conn,
        """
        SELECT *
        FROM crises
        WHERE status = 'active'
        ORDER BY created_at DESC
        """,
    )


def site_in_any_active_crisis(site: Dict[str, Any], crises: List[Dict[str, Any]]) -> bool:
    lat = site.get("latitude")
    lon = site.get("longitude")
    if lat is None or lon is None:
        return False

    for crisis in crises:
        clat = crisis.get("center_latitude")
        clon = crisis.get("center_longitude")
        radius = crisis.get("radius_km")
        if clat is None or clon is None or radius is None:
            continue
        if haversine_km(float(lat), float(lon), float(clat), float(clon)) <= float(radius):
            return True
    return False


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
        "db_path": DB_PATH if DB_MODE == "sqlite" else None,
        "time": utc_now_iso(),
    }


# =========================================================
# Admin - Sites
# =========================================================

@app.post("/admin/site")
async def admin_create_site(payload: CreateSiteRequest):
    with get_db() as conn:
        site_id = payload.site_id or next_site_id(conn)
        now = utc_now_iso()

        existing = fetch_one(
            conn,
            "SELECT 1 FROM sites WHERE site_id = :site_id",
            {"site_id": site_id},
        )
        if existing:
            raise HTTPException(status_code=409, detail="Site ID already exists.")

        if payload.kiosk_id:
            kiosk_exists = fetch_one(
                conn,
                "SELECT site_id FROM sites WHERE kiosk_id = :kiosk_id",
                {"kiosk_id": payload.kiosk_id},
            )
            if kiosk_exists:
                raise HTTPException(status_code=409, detail="Kiosk ID is already bound to another site.")

        execute_write(
            conn,
            """
            INSERT INTO sites (
                site_id, name, address, city, company_name, kiosk_id,
                latitude, longitude, site_policy_mode, created_at, updated_at
            )
            VALUES (
                :site_id, :name, :address, :city, :company_name, :kiosk_id,
                :latitude, :longitude, :site_policy_mode, :created_at, :updated_at
            )
            """,
            {
                "site_id": site_id,
                "name": payload.name.strip(),
                "address": payload.address,
                "city": payload.city,
                "company_name": payload.company_name,
                "kiosk_id": payload.kiosk_id,
                "latitude": payload.latitude,
                "longitude": payload.longitude,
                "site_policy_mode": payload.site_policy_mode,
                "created_at": now,
                "updated_at": now,
            },
        )

        ensure_site_status(conn, site_id)
        return fetch_site_summary(conn, site_id)


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
                ss.status,
                ss.active_count,
                ss.headcount_status,
                ss.sos_active,
                ss.full_headcount_confirmed,
                ss.last_event_at,
                ss.last_sos_at,
                ss.last_full_headcount_at
            FROM sites s
            LEFT JOIN site_status ss ON ss.site_id = s.site_id
            ORDER BY s.created_at DESC, s.site_id DESC
            """
        )

        crises = list_active_crises(conn)
        results = []
        for site in rows:
            site["in_active_crisis_area"] = site_in_any_active_crisis(site, crises)
            results.append(site)

        return results


@app.put("/admin/sites/{site_id}")
async def admin_update_site(site_id: str, payload: UpdateSiteRequest):
    with get_db() as conn:
        site = fetch_one(
            conn,
            "SELECT * FROM sites WHERE site_id = :site_id",
            {"site_id": site_id},
        )
        if not site:
            raise HTTPException(status_code=404, detail="Site not found.")

        updates = payload.dict(exclude_unset=True)

        if "kiosk_id" in updates and updates["kiosk_id"]:
            row = fetch_one(
                conn,
                "SELECT site_id FROM sites WHERE kiosk_id = :kiosk_id AND site_id <> :site_id",
                {"kiosk_id": updates["kiosk_id"], "site_id": site_id},
            )
            if row:
                raise HTTPException(status_code=409, detail="Kiosk ID is already bound to another site.")

        merged = {**site, **updates, "updated_at": utc_now_iso()}

        execute_write(
            conn,
            """
            UPDATE sites
            SET name = :name,
                address = :address,
                city = :city,
                company_name = :company_name,
                kiosk_id = :kiosk_id,
                latitude = :latitude,
                longitude = :longitude,
                site_policy_mode = :site_policy_mode,
                updated_at = :updated_at
            WHERE site_id = :site_id
            """,
            {
                "name": merged["name"],
                "address": merged.get("address"),
                "city": merged.get("city"),
                "company_name": merged.get("company_name"),
                "kiosk_id": merged.get("kiosk_id"),
                "latitude": merged.get("latitude"),
                "longitude": merged.get("longitude"),
                "site_policy_mode": merged.get("site_policy_mode"),
                "updated_at": merged["updated_at"],
                "site_id": site_id,
            },
        )

        return fetch_site_summary(conn, site_id)


@app.post("/admin/bind-kiosk")
async def admin_bind_kiosk(payload: BindKioskRequest):
    with get_db() as conn:
        site = fetch_one(
            conn,
            "SELECT * FROM sites WHERE site_id = :site_id",
            {"site_id": payload.site_id},
        )
        if not site:
            raise HTTPException(status_code=404, detail="Site not found.")

        other = fetch_one(
            conn,
            "SELECT site_id FROM sites WHERE kiosk_id = :kiosk_id AND site_id <> :site_id",
            {"kiosk_id": payload.kiosk_id, "site_id": payload.site_id},
        )
        if other:
            raise HTTPException(status_code=409, detail="Kiosk ID is already bound to another site.")

        execute_write(
            conn,
            "UPDATE sites SET kiosk_id = :kiosk_id, updated_at = :updated_at WHERE site_id = :site_id",
            {
                "kiosk_id": payload.kiosk_id,
                "updated_at": utc_now_iso(),
                "site_id": payload.site_id,
            },
        )
        return {
            "status": "ok",
            "site_id": payload.site_id,
            "kiosk_id": payload.kiosk_id,
        }


@app.get("/admin/sites/{site_id}/events")
async def admin_site_events(
    site_id: str,
    event_type: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
):
    with get_db() as conn:
        site = fetch_one(
            conn,
            "SELECT 1 FROM sites WHERE site_id = :site_id",
            {"site_id": site_id},
        )
        if not site:
            raise HTTPException(status_code=404, detail="Site not found.")

        sql = """
            SELECT id, site_id, device_id, event_type, occurred_at, payload_json, created_at
            FROM events
            WHERE site_id = :site_id
        """
        params: Dict[str, Any] = {"site_id": site_id}

        if event_type:
            sql += " AND event_type = :event_type"
            params["event_type"] = event_type

        sql += " ORDER BY occurred_at DESC, id DESC LIMIT :limit"
        params["limit"] = limit

        rows = fetch_all(conn, sql, params)
        result = []
        for row in rows:
            payload = json_loads(row["payload_json"])
            result.append(
                {
                    "id": row["id"],
                    "site_id": row["site_id"],
                    "device_id": row["device_id"],
                    "kiosk_id": row["device_id"],
                    "event_type": row["event_type"],
                    "timestamp": row["occurred_at"],
                    "payload": payload,
                    "worker_id": payload.get("worker_id"),
                    "name": payload.get("name"),
                    "company": payload.get("company"),
                    "role": payload.get("role"),
                    "description": payload.get("description"),
                    "severity": payload.get("severity"),
                    "injury": payload.get("injury"),
                    "note": payload.get("note"),
                }
            )
        return result


# =========================================================
# Admin - Crisis Control
# =========================================================

@app.get("/admin/crises")
async def admin_list_crises():
    with get_db() as conn:
        return list_active_crises(conn)


@app.post("/admin/crises")
async def admin_create_crisis(payload: CreateCrisisRequest):
    with get_db() as conn:
        crisis_id = payload.crisis_id or f"crisis-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        now = utc_now_iso()

        exists = fetch_one(
            conn,
            "SELECT 1 FROM crises WHERE crisis_id = :crisis_id",
            {"crisis_id": crisis_id},
        )
        if exists:
            raise HTTPException(status_code=409, detail="Crisis ID already exists.")

        execute_write(
            conn,
            """
            INSERT INTO crises (
                crisis_id, source, crisis_class, hazard_type, name, status,
                center_latitude, center_longitude, radius_km,
                created_at, updated_at, cleared_at
            )
            VALUES (
                :crisis_id, :source, :crisis_class, :hazard_type, :name, 'active',
                :center_latitude, :center_longitude, :radius_km,
                :created_at, :updated_at, NULL
            )
            """,
            {
                "crisis_id": crisis_id,
                "source": payload.source,
                "crisis_class": payload.crisis_class,
                "hazard_type": payload.hazard_type,
                "name": payload.name,
                "center_latitude": payload.center_latitude,
                "center_longitude": payload.center_longitude,
                "radius_km": payload.radius_km,
                "created_at": now,
                "updated_at": now,
            },
        )

        return {"status": "ok", "crisis_id": crisis_id}


@app.post("/admin/crises/{crisis_id}/clear")
async def admin_clear_crisis(crisis_id: str):
    with get_db() as conn:
        exists = fetch_one(
            conn,
            "SELECT 1 FROM crises WHERE crisis_id = :crisis_id",
            {"crisis_id": crisis_id},
        )
        if not exists:
            raise HTTPException(status_code=404, detail="Crisis not found.")

        now = utc_now_iso()
        execute_write(
            conn,
            """
            UPDATE crises
            SET status = 'cleared', updated_at = :updated_at, cleared_at = :cleared_at
            WHERE crisis_id = :crisis_id
            """,
            {
                "updated_at": now,
                "cleared_at": now,
                "crisis_id": crisis_id,
            },
        )
        return {"status": "ok", "crisis_id": crisis_id}


# =========================================================
# NZ-NCM Feed
# =========================================================

@app.get("/ncm/sites")
async def ncm_sites():
    with get_db() as conn:
        sites = []
        crises = list_active_crises(conn)

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
            LEFT JOIN site_status ss ON ss.site_id = s.site_id
            ORDER BY s.created_at DESC
            """
        )

        for site in rows:
            in_crisis_area = site_in_any_active_crisis(site, crises)

            ncm_state = None
            priority = None

            if site.get("sos_active"):
                ncm_state = "red"
                priority = 1
            elif in_crisis_area and site.get("active_count", 0) > 0 and not site.get("full_headcount_confirmed"):
                ncm_state = "orange"
                priority = 2
            elif site.get("full_headcount_confirmed"):
                ncm_state = "green"
                priority = 3

            if ncm_state:
                site["ncm_state"] = ncm_state
                site["priority"] = priority
                site["in_active_crisis_area"] = in_crisis_area
                sites.append(site)

        sites.sort(key=lambda s: (s["priority"], s.get("last_event_at") or ""))
        return {
            "active_crises": crises,
            "sites": sites,
        }


# =========================================================
# Kiosk API
# =========================================================

@app.post("/api/signin")
async def api_signin(payload: SignEventRequest):
    sign_event_type = normalise_sign_action(payload.type)

    if not payload.worker_id and not payload.name:
        raise HTTPException(
            status_code=400,
            detail="Minimum identity required: worker_id or name."
        )

    with get_db() as conn:
        site_id = resolve_site_id(conn, payload.site_id, payload.device_id)
        occurred_at = parse_iso(payload.timestamp).isoformat()

        event_payload = {
            "worker_id": payload.worker_id,
            "name": payload.name,
            "company": payload.company,
            "role": payload.role,
            "action": sign_event_type,
        }

        event_id = insert_event(
            conn,
            site_id=site_id,
            device_id=payload.device_id,
            event_type=sign_event_type,
            occurred_at=occurred_at,
            payload=event_payload,
        )

        state = recompute_site_state(conn, site_id)

        return {
            "status": "received",
            "event_id": event_id,
            "site_id": site_id,
            **state,
        }


@app.post("/api/hazard")
async def api_hazard(payload: HazardEventRequest):
    with get_db() as conn:
        site_id = resolve_site_id(conn, payload.site_id, payload.device_id)
        occurred_at = parse_iso(payload.timestamp).isoformat()

        event_payload = {
            "worker_id": payload.worker_id,
            "name": payload.name,
            "description": payload.description,
            "severity": payload.severity,
        }

        event_id = insert_event(
            conn,
            site_id=site_id,
            device_id=payload.device_id,
            event_type="hazard",
            occurred_at=occurred_at,
            payload=event_payload,
        )

        state = recompute_site_state(conn, site_id)

        return {
            "status": "received",
            "event_id": event_id,
            "site_id": site_id,
            **state,
        }


@app.post("/api/incident")
async def api_incident(payload: IncidentEventRequest):
    with get_db() as conn:
        site_id = resolve_site_id(conn, payload.site_id, payload.device_id)
        occurred_at = parse_iso(payload.timestamp).isoformat()

        event_payload = {
            "worker_id": payload.worker_id,
            "name": payload.name,
            "description": payload.description,
            "injury": payload.injury,
        }

        event_id = insert_event(
            conn,
            site_id=site_id,
            device_id=payload.device_id,
            event_type="incident",
            occurred_at=occurred_at,
            payload=event_payload,
        )

        state = recompute_site_state(conn, site_id)

        return {
            "status": "received",
            "event_id": event_id,
            "site_id": site_id,
            **state,
        }


@app.post("/api/sos")
async def api_sos(payload: SOSEventRequest):
    with get_db() as conn:
        site_id = resolve_site_id(conn, payload.site_id, payload.device_id)
        occurred_at = parse_iso(payload.timestamp).isoformat()

        event_id = insert_event(
            conn,
            site_id=site_id,
            device_id=payload.device_id,
            event_type="sos",
            occurred_at=occurred_at,
            payload={"note": payload.note},
        )

        state = recompute_site_state(conn, site_id)

        return {
            "status": "received",
            "event_id": event_id,
            "site_id": site_id,
            **state,
        }


@app.post("/api/full-headcount")
async def api_full_headcount(payload: FullHeadcountEventRequest):
    with get_db() as conn:
        site_id = resolve_site_id(conn, payload.site_id, payload.device_id)
        occurred_at = parse_iso(payload.timestamp).isoformat()

        event_id = insert_event(
            conn,
            site_id=site_id,
            device_id=payload.device_id,
            event_type="full_headcount",
            occurred_at=occurred_at,
            payload={"note": payload.note},
        )

        state = recompute_site_state(conn, site_id)

        return {
            "status": "received",
            "event_id": event_id,
            "site_id": site_id,
            **state,
        }