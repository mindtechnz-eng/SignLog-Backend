from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Optional, List

# -----------------------------------------
# FastAPI App
# -----------------------------------------
app = FastAPI()

# -----------------------------------------
# CORS (Allow front-end apps & kiosks)
# -----------------------------------------
origins = ["*"]  # later we tighten this for production

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------
# KIOSK EVENT MODELS
# -----------------------------------------
class SignInEvent(BaseModel):
    site_id: Optional[str]  # kiosk may still send this for now
    device_id: str          # kiosk ID / hardware ID
    name: str
    company: Optional[str] = None
    role: Optional[str] = None
    type: str              # "in" or "out"
    timestamp: datetime


class HazardEvent(BaseModel):
    site_id: Optional[str]
    device_id: str
    description: str
    severity: str
    timestamp: datetime


class IncidentEvent(BaseModel):
    site_id: Optional[str]
    device_id: str
    description: str
    injury: bool
    timestamp: datetime


class SOSEvent(BaseModel):
    site_id: Optional[str]
    device_id: str
    timestamp: datetime


# -----------------------------------------
# SIMPLE IN-MEMORY ADMIN STORAGE
# -----------------------------------------

# site_id -> JobSite
SITES: Dict[str, "JobSite"] = {}

# kiosk_id (device_id printed on hardware) -> site_id
KIOSK_BINDINGS: Dict[str, str] = {}

# simple counter – auto-generate site IDs if admin doesn’t supply one
NEXT_SITE_NUM: int = 1

# simple in-memory event store for admin dashboards
# each item: { "id", "event_type", "site_id", "device_id", "timestamp", "payload" }
EVENTS: List[dict] = []
NEXT_EVENT_ID: int = 1


class JobSite(BaseModel):
    """
    Minimal site model for admin + dashboard.
    This lives only in memory for now.
    """
    site_id: str
    name: str
    address: Optional[str] = None
    city: Optional[str] = None
    company_name: Optional[str] = None

    # hardware link
    kiosk_id: Optional[str] = None

    # status for traffic-light dashboard
    #   "idle"      -> orange  (non-active site)
    #   "live"      -> green   (active + safe)
    #   "emergency" -> red     (SOS / SoE)
    status: str = "idle"

    # basic headcount + headcount status
    active_count: int = 0
    headcount_status: str = "pending"  # pending | safe | incomplete

    # timestamp of last event (for info)
    last_event_at: Optional[datetime] = None


class CreateSiteRequest(BaseModel):
    """
    Payload the admin dashboard sends when creating a site.
    site_id is optional – if missing we auto-generate one.
    """
    name: str
    address: Optional[str] = None
    city: Optional[str] = None
    company_name: Optional[str] = None
    kiosk_id: Optional[str] = None
    site_id: Optional[str] = None  # optional friendly ID from dashboard


class BindKioskRequest(BaseModel):
    """
    Payload for binding a kiosk to a site after both exist.
    """
    kiosk_id: str
    site_id: str


# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------
def ensure_site_exists_for_event(site_id: str) -> JobSite:
    """
    Ensure there is at least a placeholder JobSite for this site_id.
    Used when kiosk events arrive before admin has created a proper site.
    """
    if site_id not in SITES:
        SITES[site_id] = JobSite(
            site_id=site_id,
            name=f"Unregistered site ({site_id})",
            address=None,
            city=None,
            company_name=None,
            kiosk_id=None,
            status="idle",
            active_count=0,
            headcount_status="pending",
            last_event_at=None,
        )
    return SITES[site_id]


def resolve_site_id_from_event(site_id: Optional[str], device_id: str) -> str:
    """
    Decide which site_id to attach to an event:
    - Prefer KIOSK_BINDINGS[device_id] if present.
    - Fall back to site_id provided by kiosk.
    - If none, use 'unassigned'.
    """
    # 1) If kiosk is bound to a site, that wins.
    if device_id in KIOSK_BINDINGS:
        return KIOSK_BINDINGS[device_id]

    # 2) If kiosk sent a site_id, use it.
    if site_id:
        return site_id

    # 3) Otherwise unassigned bucket.
    return "unassigned"


def store_event(event_type: str, site_id: str, device_id: str, payload: dict) -> dict:
    """
    Store a minimal record of an event for admin dashboards.
    """
    global NEXT_EVENT_ID

    record = {
        "id": NEXT_EVENT_ID,
        "event_type": event_type,
        "site_id": site_id,
        "device_id": device_id,
        "timestamp": datetime.utcnow(),
        "payload": payload,
    }
    EVENTS.append(record)
    NEXT_EVENT_ID += 1
    return record


# -----------------------------------------
# ADMIN ROUTES – for dashboard / command centre
# -----------------------------------------

@app.post("/admin/site")
async def admin_create_site(payload: CreateSiteRequest):
    """
    Create a site from the admin dashboard.
    If site_id is not provided, we generate one like 'site-001'.
    """
    global NEXT_SITE_NUM

    site_id = payload.site_id or f"site-{NEXT_SITE_NUM:03d}"
    NEXT_SITE_NUM += 1

    site = JobSite(
        site_id=site_id,
        name=payload.name,
        address=payload.address,
        city=payload.city,
        company_name=payload.company_name,
        kiosk_id=payload.kiosk_id,
        status="idle",
        active_count=0,
        headcount_status="pending",
        last_event_at=None,
    )

    SITES[site_id] = site

    if site.kiosk_id:
        KIOSK_BINDINGS[site.kiosk_id] = site_id

    return site


@app.get("/admin/sites")
async def admin_list_sites():
    """
    Return all sites – for the 'All Sites' traffic-light view.
    """
    return list(SITES.values())


@app.post("/admin/bind-kiosk")
async def admin_bind_kiosk(payload: BindKioskRequest):
    """
    Attach a kiosk (by its printed ID / device_id) to a site.
    Admin uses this when they lease a unit and want it feeding a particular site.
    """
    site = SITES.get(payload.site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    site.kiosk_id = payload.kiosk_id
    KIOSK_BINDINGS[payload.kiosk_id] = payload.site_id

    return {
        "status": "ok",
        "site_id": payload.site_id,
        "kiosk_id": payload.kiosk_id,
    }


@app.get("/admin/sites/{site_id}/events")
async def admin_site_events(site_id: str, event_type: Optional[str] = None, limit: int = 100):
    """
    Return recent events for a specific site.
    event_type can filter on 'signin', 'hazard', 'incident', 'sos'.
    """
    filtered = [e for e in EVENTS if e["site_id"] == site_id]
    if event_type:
        filtered = [e for e in filtered if e["event_type"] == event_type]

    # newest first, up to limit
    filtered = sorted(filtered, key=lambda e: e["id"], reverse=True)[:limit]
    return filtered


# -----------------------------------------
# ROUTES – KIOSK API
# -----------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "signlog-backend"}


@app.post("/api/signin")
async def create_signin(event: SignInEvent):
    # resolve site
    site_id = resolve_site_id_from_event(event.site_id, event.device_id)
    site = ensure_site_exists_for_event(site_id)

    # update headcount
    if event.type.lower() == "in":
        site.active_count += 1
    elif event.type.lower() == "out" and site.active_count > 0:
        site.active_count -= 1

    # set status based on headcount (no emergency)
    if site.active_count > 0:
        # active + safe (green)
        site.status = "live"
    else:
        # non-active site (orange)
        site.status = "idle"

    site.last_event_at = datetime.utcnow()

    # store event for admin dashboards
    store_event(
        "signin",
        site_id,
        event.device_id,
        event.dict(),
    )

    print("SignIn:", event.dict(), "-> site:", site_id, "active_count:", site.active_count)
    return {"status": "received", "site_id": site_id, "active_count": site.active_count}


@app.post("/api/hazard")
async def create_hazard(event: HazardEvent):
    site_id = resolve_site_id_from_event(event.site_id, event.device_id)
    site = ensure_site_exists_for_event(site_id)

    site.last_event_at = datetime.utcnow()

    store_event(
        "hazard",
        site_id,
        event.device_id,
        event.dict(),
    )

    print("Hazard:", event.dict(), "-> site:", site_id)
    return {"status": "received", "site_id": site_id}


@app.post("/api/incident")
async def create_incident(event: IncidentEvent):
    site_id = resolve_site_id_from_event(event.site_id, event.device_id)
    site = ensure_site_exists_for_event(site_id)

    site.last_event_at = datetime.utcnow()

    store_event(
        "incident",
        site_id,
        event.device_id,
        event.dict(),
    )

    print("Incident:", event.dict(), "-> site:", site_id)
    return {"status": "received", "site_id": site_id}


@app.post("/api/sos")
async def create_sos(event: SOSEvent):
    site_id = resolve_site_id_from_event(event.site_id, event.device_id)
    site = ensure_site_exists_for_event(site_id)

    # mark emergency (red)
    site.status = "emergency"
    site.last_event_at = datetime.utcnow()

    record = store_event(
        "sos",
        site_id,
        event.device_id,
        event.dict(),
    )

    # CDEM hook placeholder – for now, just log what would be sent
    print("[CDEM] Emergency signal would be sent for:", {
        "site_id": site_id,
        "site_name": site.name,
        "status": site.status,
        "event_id": record["id"],
    })

    print("SOS Trigger:", event.dict(), "-> site:", site_id)
    return {"status": "received", "site_id": site_id}
