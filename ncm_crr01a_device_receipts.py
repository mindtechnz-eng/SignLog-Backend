"""NZ-NCM CRR-01A C2-B1: bound device credentials + authenticated renderer claims.

No public Kiosk credential. No secret in browser JavaScript. The trusted
appliance agent is the future caller; this component is its Backend contract.
A verified device credential authenticates the reporting *device*, not human
awareness, true pixel display, rescue, response causation or SOS/FHC attribution.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

CONTRACT = 'ncm-crr01a.renderer-claim.v1'
MAX_CLOCK_SKEW_SECONDS = 120


@dataclass(frozen=True)
class DevicePrincipal:
    kiosk_id: str
    site_id: str
    provisioned_at: str


def utc_datetime(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Explicit timezone-aware timestamp required')
    try:
        parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except (ValueError, OverflowError) as exc:
        raise ValueError('Invalid ISO timestamp') from exc
    if parsed.tzinfo is None:
        raise ValueError('Timestamp must contain timezone')
    return parsed.astimezone(timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def _binding(conn: Connection, kiosk_id: str) -> dict[str, Any] | None:
    row = conn.execute(text('''
        SELECT site_id, kiosk_id, operational_state FROM sites
        WHERE kiosk_id=:kiosk AND operational_state='active'
    '''), {'kiosk': kiosk_id}).mappings().first()
    return dict(row) if row else None


def ensure_crr01a_device_receipt_schema(conn: Connection) -> None:
    """Idempotent cross-engine additive schema, independent of emergency events."""
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS ncm_crr01a_device_credentials (
            kiosk_id TEXT PRIMARY KEY,
            bound_site_id TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            credential_version INTEGER NOT NULL CHECK(credential_version >= 1),
            provisioned_at TEXT NOT NULL,
            revoked_at TEXT,
            FOREIGN KEY(bound_site_id) REFERENCES sites(site_id)
        )
    '''))
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS ncm_crr01a_device_credential_actions (
            action_id TEXT PRIMARY KEY,
            kiosk_id TEXT NOT NULL,
            site_id TEXT NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('provision','rotate','revoke')),
            credential_version INTEGER NOT NULL,
            occurred_at TEXT NOT NULL
        )
    '''))
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS ncm_crr01a_renderer_receipts (
            receipt_id TEXT PRIMARY KEY,
            context_id TEXT NOT NULL,
            crisis_id TEXT NOT NULL,
            site_id TEXT NOT NULL,
            kiosk_id TEXT NOT NULL,
            credential_version INTEGER NOT NULL,
            renderer_version TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            displayed_at_claim TEXT NOT NULL,
            backend_received_at TEXT NOT NULL,
            receipt_basis TEXT NOT NULL CHECK(
                receipt_basis='device_authenticated_renderer_claim_not_pixel_proof'
            ),
            UNIQUE(context_id, kiosk_id),
            UNIQUE(kiosk_id, claim_id),
            FOREIGN KEY(context_id) REFERENCES ncm_crisis_attention_contexts(context_id),
            FOREIGN KEY(crisis_id) REFERENCES crises(crisis_id),
            FOREIGN KEY(site_id) REFERENCES sites(site_id)
        )
    '''))
    conn.execute(text('''CREATE INDEX IF NOT EXISTS idx_ncm_c2b1_receipts_crisis_site_time
        ON ncm_crr01a_renderer_receipts(crisis_id,site_id,backend_received_at)'''))


def provision_credential(conn: Connection, *, kiosk_id: str, now: str,
                         rotate: bool = False) -> dict[str, Any]:
    """Privileged one-time enrollment; return token ONCE, never persist plaintext."""
    from uuid import uuid4
    kiosk = kiosk_id.strip() if isinstance(kiosk_id, str) else ''
    if not kiosk or len(kiosk) > 200:
        raise ValueError('Invalid kiosk identity')
    moment = utc_datetime(now).isoformat()
    binding = _binding(conn, kiosk)
    if binding is None:
        raise LookupError('Kiosk must be bound to an active Site')
    existing = conn.execute(text('''SELECT credential_version FROM ncm_crr01a_device_credentials
        WHERE kiosk_id=:kiosk'''), {'kiosk': kiosk}).mappings().first()
    if existing and not rotate:
        raise RuntimeError('Device is already provisioned; explicitly request rotation')
    if not existing and rotate:
        raise RuntimeError('No existing credential to rotate; provision first')

    secret = 'slkc1.' + secrets.token_urlsafe(48)
    version = int(existing['credential_version']) + 1 if existing else 1
    conn.execute(text('''
        INSERT INTO ncm_crr01a_device_credentials (
            kiosk_id,bound_site_id,token_hash,credential_version,provisioned_at,revoked_at
        ) VALUES(:kiosk,:site,:hash,:version,:at,NULL)
        ON CONFLICT(kiosk_id) DO UPDATE SET bound_site_id=excluded.bound_site_id,
            token_hash=excluded.token_hash,
            credential_version=excluded.credential_version,
            provisioned_at=excluded.provisioned_at, revoked_at=NULL
    '''), {'kiosk': kiosk,'site': binding['site_id'],'hash': _token_hash(secret),
           'version': version,'at': moment})
    conn.execute(text('''INSERT INTO ncm_crr01a_device_credential_actions
        (action_id,kiosk_id,site_id,action,credential_version,occurred_at)
        VALUES(:id,:kiosk,:site,:action,:version,:at)'''),
        {'id': str(uuid4()), 'kiosk': kiosk,'site': binding['site_id'],
         'action': 'rotate' if existing else 'provision','version': version,'at': moment})
    return {'kiosk_id': kiosk, 'site_id': binding['site_id'], 'credential_version': version,
            'provisioned_at': moment, 'credential': secret,
            'notice': 'One-time private appliance credential: never put in kiosk web root, JS, Git or project files'}


def revoke_credential(conn: Connection, *, kiosk_id: str, now: str) -> dict[str, Any]:
    from uuid import uuid4
    moment = utc_datetime(now).isoformat()
    row = conn.execute(text('''SELECT bound_site_id,credential_version,revoked_at
        FROM ncm_crr01a_device_credentials WHERE kiosk_id=:kiosk'''),
        {'kiosk': kiosk_id}).mappings().first()
    if not row:
        raise LookupError('Device credential not found')
    if row['revoked_at'] is not None:
        return {'kiosk_id': kiosk_id, 'revoked_at': row['revoked_at'], 'already_revoked': True}
    conn.execute(text('''UPDATE ncm_crr01a_device_credentials SET revoked_at=:at WHERE kiosk_id=:kiosk
        AND revoked_at IS NULL'''), {'at': moment,'kiosk': kiosk_id})
    conn.execute(text('''INSERT INTO ncm_crr01a_device_credential_actions
        (action_id,kiosk_id,site_id,action,credential_version,occurred_at)
        VALUES(:id,:kiosk,:site,'revoke',:version,:at)'''),
        {'id': str(uuid4()),'kiosk': kiosk_id,'site': row['bound_site_id'],
         'version':row['credential_version'],'at':moment})
    return {'kiosk_id': kiosk_id,'revoked_at':moment, 'already_revoked': False}


def authenticate_device(conn: Connection, *, bearer: str | None) -> DevicePrincipal:
    """High-entropy bearer only; no device_id string or public bootstrap trust."""
    if not isinstance(bearer, str) or not bearer.startswith('Bearer '):
        raise PermissionError('Authenticated appliance credential required')
    token = bearer[len('Bearer '):]
    if not token.startswith('slkc1.') or not (60 <= len(token) <= 200):
        raise PermissionError('Invalid appliance credential')
    row = conn.execute(text('''
        SELECT kiosk_id,bound_site_id,token_hash,provisioned_at,revoked_at
        FROM ncm_crr01a_device_credentials WHERE token_hash=:hash
    '''), {'hash': _token_hash(token)}).mappings().first()
    if not row or row['revoked_at'] is not None:
        raise PermissionError('Invalid or revoked appliance credential')
    # Constant-time equality for the matched hash; token is not logged anywhere.
    if not secrets.compare_digest(row['token_hash'], _token_hash(token)):
        raise PermissionError('Invalid appliance credential')
    binding = _binding(conn, row['kiosk_id'])
    if not binding or binding['site_id'] != row['bound_site_id']:
        raise PermissionError('Appliance binding changed; re-provision required')
    return DevicePrincipal(kiosk_id=row['kiosk_id'], site_id=row['bound_site_id'],
                           provisioned_at=row['provisioned_at'])


def accept_renderer_claim(conn: Connection, *, principal: DevicePrincipal,
                          context_id: str, displayed_at: str, received_at: str,
                          claim_id: str, renderer_version: str) -> dict[str, Any]:
    """Accept a bounded device-authenticated claim, NOT physical display proof.

    Does not change original issued row or crisis state. Returns the first
    accepted original receipt on retries; refuses conflicting duplicate claims.
    """
    from uuid import uuid4
    now = utc_datetime(received_at)
    claimed = utc_datetime(displayed_at)
    if not isinstance(context_id, str) or not (8 <= len(context_id) <= 128):
        raise ValueError('Invalid context_id')
    if not isinstance(claim_id, str) or not (8 <= len(claim_id) <= 128):
        raise ValueError('Invalid claim_id')
    if not isinstance(renderer_version, str) or not (1 <= len(renderer_version.strip()) <= 80):
        raise ValueError('Invalid renderer version')
    row = conn.execute(text('''SELECT context_id,crisis_id,site_id,kiosk_id,issued_at,expires_at
        FROM ncm_crisis_attention_contexts WHERE context_id=:id'''),
        {'id': context_id}).mappings().first()
    if not row:
        raise LookupError('Unknown issued context')
    if row['kiosk_id'] != principal.kiosk_id or row['site_id'] != principal.site_id:
        raise PermissionError('Context belongs to another appliance or Site')
    # A previously accepted identical claim is idempotent even after expiry.
    existing = conn.execute(text('''SELECT receipt_id,claim_id,displayed_at_claim,renderer_version,
        backend_received_at,receipt_basis FROM ncm_crr01a_renderer_receipts
        WHERE context_id=:context AND kiosk_id=:kiosk'''),
        {'context': context_id,'kiosk':principal.kiosk_id}).mappings().first()
    if existing:
        if (existing['claim_id'] != claim_id or existing['displayed_at_claim'] != claimed.isoformat()
            or existing['renderer_version'] != renderer_version.strip()):
            raise RuntimeError('A different display claim is already recorded for this context')
        return {**dict(existing),'context_id':context_id,'kiosk_id':principal.kiosk_id,
                'site_id':principal.site_id,'crisis_id':row['crisis_id'],'duplicate':True}
    reused_claim = conn.execute(text('''SELECT context_id FROM ncm_crr01a_renderer_receipts
        WHERE kiosk_id=:kiosk AND claim_id=:claim'''),
        {'kiosk':principal.kiosk_id,'claim':claim_id}).mappings().first()
    if reused_claim is not None:
        raise RuntimeError('A renderer claim ID cannot be reused for a different crisis context')
    issued = utc_datetime(row['issued_at'])
    expires = utc_datetime(row['expires_at'])
    slack = timedelta(seconds=MAX_CLOCK_SKEW_SECONDS)
    if not (issued - slack <= claimed <= now + slack and claimed <= expires
            and now <= expires + slack):
        raise ValueError('Stale, early, or future display claim; no verified receipt recorded')
    if now < issued - slack:
        raise ValueError('Backend receipt clock predates issuance')
    cred = conn.execute(text('''SELECT credential_version FROM ncm_crr01a_device_credentials
        WHERE kiosk_id=:kiosk AND revoked_at IS NULL'''), {'kiosk': principal.kiosk_id}).mappings().one()
    receipt_id = str(uuid4())
    basis='device_authenticated_renderer_claim_not_pixel_proof'
    conn.execute(text('''INSERT INTO ncm_crr01a_renderer_receipts (
        receipt_id,context_id,crisis_id,site_id,kiosk_id,credential_version,
        renderer_version,claim_id,displayed_at_claim,backend_received_at,receipt_basis
        ) VALUES (:id,:context,:crisis,:site,:kiosk,:version,:renderer,:claim,:displayed,:received,:basis)
        ON CONFLICT(context_id,kiosk_id) DO NOTHING'''),
        {'id':receipt_id,'context':context_id,'crisis':row['crisis_id'],
         'site':principal.site_id,'kiosk':principal.kiosk_id,'version':cred['credential_version'],
         'renderer':renderer_version.strip(),'claim':claim_id,
         'displayed':claimed.isoformat(),'received':now.isoformat(),'basis':basis})
    stored = conn.execute(text('''SELECT receipt_id,claim_id,displayed_at_claim,renderer_version,
        backend_received_at,receipt_basis FROM ncm_crr01a_renderer_receipts
        WHERE context_id=:context AND kiosk_id=:kiosk'''),
        {'context':context_id,'kiosk':principal.kiosk_id}).mappings().one()
    if stored['claim_id'] != claim_id:
        raise RuntimeError('Concurrent, conflicting renderer claim')
    return {**dict(stored),'context_id':context_id,'kiosk_id':principal.kiosk_id,
            'site_id':principal.site_id,'crisis_id':row['crisis_id'],
            'duplicate': stored['receipt_id'] != receipt_id}
