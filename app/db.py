import json
import asyncio
from pathlib import Path

import libsql

from app.config import TURSO_DATABASE_URL, TURSO_AUTH_TOKEN

# Conexión oficial recomendada por Turso:
# local file + sync remoto (embedded replica)
DB_FILE = "local.db"

conn = libsql.connect(DB_FILE, sync_url=TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)


async def init_db():
    schema_path = Path(__file__).parent / "sql" / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")

    # Ejecutar script completo
    conn.executescript(schema_sql)
    conn.commit()

    # Sincroniza con Turso
    conn.sync()


async def create_case(image_name: str, image_path: str, original_description: str | None = None) -> int:
    cur = conn.execute(
        """
        INSERT INTO cases (image_name, image_path, original_description)
        VALUES (?, ?, ?)
        RETURNING id_case
        """,
        [image_name, image_path, original_description]
    )
    row = cur.fetchone()
    conn.commit()
    conn.sync()
    return row[0]


async def save_patterns_result(
    id_case: int,
    model_version: str,
    pred1_label: str,
    pred1_prob: float,
    pred2_label: str | None,
    pred2_prob: float | None,
    probs_json: dict,
    gradcam_json: dict | None,
    report_json: dict | None,
):
    conn.execute(
        """
        INSERT INTO results_patterns
        (id_case, model_version, pred1_label, pred1_prob, pred2_label, pred2_prob, probs_json, gradcam_json, report_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            id_case,
            model_version,
            pred1_label,
            pred1_prob,
            pred2_label,
            pred2_prob,
            json.dumps(probs_json, ensure_ascii=False),
            json.dumps(gradcam_json or {}, ensure_ascii=False),
            json.dumps(report_json or {}, ensure_ascii=False),
        ]
    )
    conn.commit()
    conn.sync()


async def save_diseases_result(
    id_case: int,
    model_version: str,
    pred1_label: str,
    pred1_prob: float,
    pred2_label: str | None,
    pred2_prob: float | None,
    probs_json: dict,
):
    conn.execute(
        """
        INSERT INTO results_diseases
        (id_case, model_version, pred1_label, pred1_prob, pred2_label, pred2_prob, probs_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            id_case,
            model_version,
            pred1_label,
            pred1_prob,
            pred2_label,
            pred2_prob,
            json.dumps(probs_json, ensure_ascii=False),
        ]
    )
    conn.commit()
    conn.sync()


async def get_history(limit: int = 20):
    cur = conn.execute(
        """
        SELECT
          c.id_case,
          c.image_name,
          c.image_path,
          c.created_at,
          rp.pred1_label,
          rp.pred1_prob,
          rd.pred1_label,
          rd.pred1_prob
        FROM cases c
        LEFT JOIN results_patterns rp ON rp.id_case = c.id_case
        LEFT JOIN results_diseases rd ON rd.id_case = c.id_case
        ORDER BY c.id_case DESC
        LIMIT ?
        """,
        [limit]
    )
    return cur.fetchall()