"""NZ-NCM CW-LOG-01B2: Backend-owned, per-crisis Site signal journal.

Records the NZ-NCM-derived AS transition when its existing qualification
becomes true, and links subsequent ORIGINAL SOS/FHC Site event IDs. No UI
polling clock, human acknowledgement inference, duplicate authority or server.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid5, NAMESPACE_URL
from sqlalchemy import text, inspect
from sqlalchemy.engine import Connection

CONTRACT = "ncm-crisis-signal-journal.v1"


def ensure_signal_journal_schema(conn: Connection) -> None:
    conn.execute(text('''CREATE TABLE IF NOT EXISTS ncm_crisis_site_signal_journal (
        signal_id TEXT PRIMARY KEY,
        crisis_id TEXT NOT NULL,
        site_id TEXT NOT NULL,
        signal_state TEXT NOT NULL CHECK (signal_state IN ('as','sos','fhc')),
        recorded_at TEXT NOT NULL,
        original_event_id INTEGER,
        original_event_type TEXT,
        original_occurred_at TEXT,
        source_basis TEXT NOT NULL,
        active_count_snapshot INTEGER,
        contract_version TEXT NOT NULL,
        FOREIGN KEY (crisis_id) REFERENCES crises(crisis_id),
        FOREIGN KEY (site_id) REFERENCES sites(site_id)
    )'''))
    conn.execute(text('''CREATE INDEX IF NOT EXISTS idx_ncm_crisis_site_signal_journal_crisis
        ON ncm_crisis_site_signal_journal (crisis_id, recorded_at, site_id)'''))


def record_crisis_site_state(conn: Connection, *, crisis: dict, site: dict,
                             qualified_pending: bool, site_in_radius: bool,
                             observed_at: str, source_event_id: int | None = None,
                             source_event_type: str | None = None,
                             source_occurred_at: str | None = None) -> str | None:
    """Record only an actual AS qualification or original SOS/FHC event.

    Caller is the existing Backend mutation transaction. Repeated observations
    of unchanged AS do not create rows; original event_id is deduplicated.
    """
    if crisis.get('status') != 'active' or not site_in_radius:
        return None
    crisis_id, site_id = str(crisis['crisis_id']), str(site['site_id'])
    count = int(site.get('active_count') or 0)
    if source_event_type == 'sos' and bool(site.get('sos_active')):
        state = 'sos'
    elif source_event_type == 'full_headcount' and bool(site.get('full_headcount_confirmed')):
        state = 'fhc'
    elif count > 0 and qualified_pending and not bool(site.get('sos_active')):
        state = 'as'
    else:
        return None

    parsed = datetime.fromisoformat(observed_at.replace('Z','+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Backend observation must carry UTC offset')
    recorded_at = parsed.astimezone(timezone.utc).isoformat()
    if state in ('sos','fhc'):
        if not isinstance(source_event_id, int) or source_event_id <= 0 or not source_occurred_at:
            return None
        source_time = datetime.fromisoformat(source_occurred_at.replace('Z','+00:00'))
        crisis_time = datetime.fromisoformat(str(crisis['created_at']).replace('Z','+00:00'))
        if source_time.tzinfo is None or crisis_time.tzinfo is None or source_time < crisis_time:
            return None  # older original event cannot become response to later crisis
    else:
        last = conn.execute(text('''SELECT signal_state FROM ncm_crisis_site_signal_journal
            WHERE crisis_id=:c AND site_id=:s ORDER BY recorded_at DESC, signal_id DESC LIMIT 1'''),
            {'c':crisis_id,'s':site_id}).mappings().first()
        if last and last['signal_state'] == 'as':
            return None

    trigger = str(source_event_id) if source_event_id else 'crisis-created'
    signal_id = str(uuid5(NAMESPACE_URL, f'ncm/cw-log01b2/{crisis_id}/{site_id}/{state}/{trigger}'))
    conn.execute(text('''INSERT INTO ncm_crisis_site_signal_journal
        (signal_id, crisis_id, site_id, signal_state, recorded_at,
         original_event_id, original_event_type, original_occurred_at,
         source_basis, active_count_snapshot, contract_version)
        VALUES (:id,:crisis,:site,:state,:clock,:event,:event_type,:event_clock,:basis,:count,:contract)
        ON CONFLICT (signal_id) DO NOTHING'''), {
            'id':signal_id,'crisis':crisis_id,'site':site_id,'state':state,
            'clock':recorded_at,'event':source_event_id,'event_type':source_event_type,
            'event_clock':source_occurred_at,
            'basis':'ncm_backend_as_qualification' if state=='as' else 'original_site_event_under_named_active_crisis',
            'count':count,'contract':CONTRACT,
        })
    return signal_id


def read_crisis_signal_journal(conn: Connection, crisis_id: str, *, max_rows: int=5000) -> dict:
    if not 1 <= max_rows <= 5000:
        raise ValueError('Invalid bounded read')
    if 'ncm_crisis_site_signal_journal' not in inspect(conn).get_table_names():
        return {'contract':CONTRACT,'capture_status':'schema_not_installed','complete':False,'rows':[]}
    rows=conn.execute(text('''SELECT signal_id,crisis_id,site_id,signal_state,recorded_at,
        original_event_id,original_event_type,original_occurred_at,source_basis,
        active_count_snapshot,contract_version FROM ncm_crisis_site_signal_journal
        WHERE crisis_id=:id ORDER BY recorded_at ASC,signal_id ASC LIMIT :limit'''),
        {'id':crisis_id,'limit':max_rows+1}).mappings().all()
    if len(rows)>max_rows:
        raise RuntimeError('Journal exceeds bounded historical read')
    return {'contract':CONTRACT,'capture_status':'recorded','complete':True,
            'rows':[dict(row) for row in rows]}
