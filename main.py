from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

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
# MODELS
# -----------------------------------------
class SignInEvent(BaseModel):
    site_id: str
    device_id: str
    name: str
    company: str | None = None
    role: str | None = None
    type: str              # "in" or "out"
    timestamp: datetime


class HazardEvent(BaseModel):
    site_id: str
    device_id: str
    description: str
    severity: str
    timestamp: datetime


class IncidentEvent(BaseModel):
    site_id: str
    device_id: str
    description: str
    injury: bool
    timestamp: datetime


class SOSEvent(BaseModel):
    site_id: str
    device_id: str
    timestamp: datetime


# -----------------------------------------
# ROUTES
# -----------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "signlog-backend"}


@app.post("/api/signin")
async def create_signin(event: SignInEvent):
    print("SignIn:", event.dict())
    return {"status": "received"}


@app.post("/api/hazard")
async def create_hazard(event: HazardEvent):
    print("Hazard:", event.dict())
    return {"status": "received"}


@app.post("/api/incident")
async def create_incident(event: IncidentEvent):
    print("Incident:", event.dict())
    return {"status": "received"}


@app.post("/api/sos")
async def create_sos(event: SOSEvent):
    print("SOS Trigger:", event.dict())
    return {"status": "received"}
