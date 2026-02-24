"""
database.py
============
SQLite-backed transcript logging for research and historical review.
Each completed triage session is saved with the full interview Q&A,
demographics, prediction, risk percentages, specialist info, and
triage summary.
"""

import sqlite3
import json
import csv
import io
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "transcripts.db"


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT UNIQUE NOT NULL,
            timestamp       TEXT NOT NULL,
            patient_name    TEXT,
            age             INTEGER,
            sex             TEXT,
            zip_code        TEXT,
            answering_for   TEXT,
            symptom_text    TEXT,
            pmh_text        TEXT,
            selected_symptoms TEXT,
            pmh             TEXT,
            interview_history TEXT,
            prediction_level INTEGER,
            prediction_label TEXT,
            risk_pcts       TEXT,
            specialist_info TEXT,
            reassurance     TEXT,
            escalation      TEXT,
            triage_summary  TEXT,
            red_flag        TEXT,
            risk_factors    TEXT
        )
    """)
    conn.commit()
    conn.close()


def generate_session_id():
    return str(uuid.uuid4())


def save_transcript(session_id, state, prediction, evidence):
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO transcripts (
                session_id, timestamp, patient_name, age, sex, zip_code,
                answering_for, symptom_text, pmh_text, selected_symptoms,
                pmh, interview_history, prediction_level, prediction_label,
                risk_pcts, specialist_info, reassurance, escalation,
                triage_summary, red_flag, risk_factors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.now(timezone.utc).isoformat(),
            state.name,
            state.age,
            state.sex,
            state.zip_code,
            state.answering_for,
            state.symptom_text,
            state.pmh_text,
            json.dumps(state.selected_symptoms),
            json.dumps(state.pmh),
            json.dumps(state.interview_history),
            prediction.get("level"),
            prediction.get("label"),
            json.dumps(evidence.get("risk_pcts", {})),
            json.dumps(prediction.get("specialist", {})),
            evidence.get("reassurance", ""),
            json.dumps(evidence.get("escalation", [])),
            evidence.get("triage_summary", ""),
            json.dumps(prediction.get("red_flag")) if prediction.get("red_flag") else None,
            json.dumps(prediction.get("risk_factors", [])),
        ))
        conn.commit()
    finally:
        conn.close()


def get_transcripts(page=1, per_page=25):
    conn = _get_conn()
    offset = (page - 1) * per_page

    total = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    rows = conn.execute(
        "SELECT * FROM transcripts ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (per_page, offset),
    ).fetchall()
    conn.close()

    total_pages = max(1, (total + per_page - 1) // per_page)
    return [dict(r) for r in rows], total, total_pages


def get_transcript_by_id(transcript_id):
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM transcripts WHERE id = ?", (transcript_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def export_all_json():
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM transcripts ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()

    records = []
    for r in rows:
        d = dict(r)
        for key in ("selected_symptoms", "pmh", "interview_history",
                     "risk_pcts", "specialist_info", "escalation",
                     "red_flag", "risk_factors"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        records.append(d)
    return json.dumps(records, indent=2)


def export_all_csv():
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM transcripts ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()

    if not rows:
        return ""

    output = io.StringIO()
    columns = rows[0].keys()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for r in rows:
        writer.writerow(dict(r))
    return output.getvalue()
