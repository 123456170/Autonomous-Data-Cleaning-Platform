# app.py
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
import time
import tracemalloc
import uuid
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher, get_close_matches
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import av
    AV_AVAILABLE = True
except Exception:
    AV_AVAILABLE = False


st.set_page_config(
    page_title="Autonomous Data Cleaning Platform",
    page_icon="🧹",
    layout="wide",
)

REQUIRED_COLUMNS = [
    "customer_id",
    "name",
    "email",
    "age",
    "category",
    "order_date",
    "amount",
    "status",
    "region",
]

ALLOWED_STATUS = ["shipped", "pending", "cancelled"]
ALLOWED_REGION = ["North", "South", "East", "West"]

CANONICAL_CATEGORY = {
    "electronics": "Electronics",
    "electronic": "Electronics",
    "books": "Books",
    "book": "Books",
    "home": "Home",
    "house": "Home",
    "fashion": "Fashion",
    "clothing": "Fashion",
}

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
ISO_LIKE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?")

MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

FINDINGS_COLUMNS = [
    "row_id",
    "column",
    "defect_type",
    "severity",
    "confidence",
    "suggested_action",
    "destructive",
    "evidence",
    "expected",
    "actual",
    "suggested_value",
]

PROPOSALS_COLUMNS = [
    "proposal_id",
    "defect_type",
    "column",
    "row_count",
    "row_ids",
    "action",
    "destructive",
    "confidence",
    "evidence",
    "status",
    "suggested_value",
]


def init_session() -> None:
    defaults: Dict[str, Any] = {
        "active_source": "Synthetic demo",
        "seed": 42,
        "n_rows": 120,
        "outlier_z": 3.0,
        "duplicate_similarity_threshold": 95,
        "auto_apply_safe_fixes": True,
        "require_approval_for_destructive": True,
        "max_preview_rows": 10,
        "future_tolerance_days": 1,
        "auto_start": True,
        "audit": [],
        "proposals_decisions": {},
        "video_bytes": None,
        "video_key": "",
        "run_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dataframe_checksum(df: pd.DataFrame) -> str:
    if df.empty:
        return sha256_bytes(b"")
    return sha256_bytes(df.to_csv(index=False).encode("utf-8", errors="replace"))


def hash_config(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def stable_id(*parts: Any) -> str:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def is_blank_value(x: Any) -> bool:
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip().lower()
    return s in {"", "nan", "none", "nat", "null"}


def blank_mask(s: pd.Series) -> pd.Series:
    mask = s.isna()
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        mask = mask | s.astype(str).str.strip().str.lower().isin(
            ["", "nan", "none", "nat", "null"]
        )
    return mask


def mask_value(value: Any, column: Optional[str] = None) -> str:
    s = str(value)
    col = (column or "").lower()
    if "@" in s:
        user, domain = s.split("@", 1)
        return f"{user[:1]}***@{domain}"
    if col in {"name", "customer_id", "email"} and len(s) > 3:
        return f"{s[:1]}***{s[-1:]}"
    return s[:48]


def empty_findings_df() -> pd.DataFrame:
    return pd.DataFrame(columns=FINDINGS_COLUMNS)


def empty_proposals_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PROPOSALS_COLUMNS)


def proposals_display(proposals: pd.DataFrame) -> pd.DataFrame:
    df = proposals.copy()
    if "row_ids" in df.columns:
        df["row_ids"] = df["row_ids"].apply(
            lambda x: ", ".join(str(i) for i in list(x)[:5])
            + ("..." if len(list(x)) > 5 else "")
        )
    return df


def ensure_object_column(df: pd.DataFrame, col: str) -> None:
    if col in df.columns and not pd.api.types.is_object_dtype(df[col]):
        df[col] = df[col].astype(object)


@st.cache_data(show_spinner=False)
def generate_synthetic_dataset(
    seed: int, n_rows: int
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    n_rows = max(1, int(n_rows))

    first_names = [
        "Ada",
        "Alan",
        "Grace",
        "Linus",
        "Margaret",
        "Dennis",
        "Barbara",
        "Ken",
        "Radia",
        "Vint",
    ]
    last_names = [
        "Lovelace",
        "Turing",
        "Hopper",
        "Torvalds",
        "Hamilton",
        "Ritchie",
        "Liskov",
        "Thompson",
        "Perlman",
        "Cerf",
    ]
    categories = ["Electronics", "Books", "Home", "Fashion"]
    statuses = ALLOWED_STATUS
    regions = ALLOWED_REGION

    end_date = pd.Timestamp.now(tz="UTC").normalize()
    dates = pd.date_range(end=end_date, periods=n_rows, freq="D")
    names = [
        f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(n_rows)
    ]

    clean = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:04d}" for i in range(1, n_rows + 1)],
            "name": names,
            "email": [
                f"{name.split()[0].lower()}.{name.split()[1].lower()}@example.com"
                for name in names
            ],
            "age": rng.integers(18, 76, n_rows),
            "category": rng.choice(categories, n_rows),
            "order_date": dates.strftime("%Y-%m-%d"),
            "amount": np.round(rng.uniform(10, 500, n_rows), 2),
            "status": rng.choice(statuses, n_rows),
            "region": rng.choice(regions, n_rows),
        }
    )

    corrupted = clean.copy()

    # Important for modern pandas: make the corrupted fixture tolerant of
    # mixed-type defects such as putting "unknown" into an integer column.
    for col in corrupted.columns:
        corrupted[col] = corrupted[col].astype(object)

    manifest: List[Dict[str, Any]] = []
    defective_indices: set[int] = set()

    def log_defect(
        row_id: int,
        column: str,
        defect_type: str,
        expected: Any,
        actual: Any,
        severity: str,
        repair: str,
        confidence: float,
        destructive: bool,
    ) -> None:
        manifest.append(
            {
                "row_id": int(row_id),
                "column": column,
                "defect_type": defect_type,
                "expected": str(expected),
                "actual": str(actual),
                "severity": severity,
                "suggested_repair": repair,
                "confidence": float(confidence),
                "destructive": bool(destructive),
            }
        )
        defective_indices.add(int(row_id))

    def pick(k: int) -> np.ndarray:
        k = min(max(1, int(k)), n_rows)
        return rng.choice(n_rows, size=k, replace=False)

    # Whitespace in names
    for i in pick(max(3, n_rows // 20)):
        old = corrupted.at[i, "name"]
        new = f"  {old} "
        corrupted.at[i, "name"] = new
        log_defect(
            i,
            "name",
            "whitespace",
            old,
            new,
            "low",
            "trim whitespace",
            0.99,
            False,
        )

    # Missing email
    for i in pick(max(2, n_rows // 15)):
        old = corrupted.at[i, "email"]
        corrupted.at[i, "email"] = np.nan
        log_defect(
            i,
            "email",
            "missing_required",
            old,
            "",
            "medium",
            "route to source; do not invent",
            0.99,
            False,
        )

    # Invalid email
    bad_emails = [
        "not-an-email",
        "foo@",
        "@bar.com",
        "a@@b.com",
        "name@domain",
        "email with space@example.com",
    ]
    for i, bad in zip(pick(max(2, n_rows // 18)), bad_emails):
        old = corrupted.at[i, "email"]
        corrupted.at[i, "email"] = bad
        log_defect(
            i,
            "email",
            "invalid_email",
            "valid email",
            bad,
            "medium",
            "set to null and flag",
            0.95,
            True,
        )

    # Invalid age
    for i in pick(max(2, n_rows // 25)):
        old = corrupted.at[i, "age"]
        bad = int(rng.choice([-5, 150, 200]))
        corrupted.at[i, "age"] = bad
        log_defect(
            i,
            "age",
            "invalid_value",
            "0-120",
            bad,
            "medium",
            "set to null or cap after approval",
            0.94,
            True,
        )

    # Non-numeric age
    if n_rows > 10:
        i = int(pick(1)[0])
        old = corrupted.at[i, "age"]
        corrupted.at[i, "age"] = "unknown"
        log_defect(
            i,
            "age",
            "invalid_value",
            "numeric",
            "unknown",
            "medium",
            "set to null",
            0.93,
            True,
        )

    # Negative amount
    for i in pick(max(1, n_rows // 30)):
        old = corrupted.at[i, "amount"]
        bad = -abs(float(old))
        corrupted.at[i, "amount"] = bad
        log_defect(
            i,
            "amount",
            "invalid_value",
            ">= 0",
            bad,
            "high",
            "set to null",
            0.96,
            True,
        )

    # Suspicious amount outlier
    for i in pick(max(1, n_rows // 30)):
        old = float(corrupted.at[i, "amount"])
        bad = round(old * float(rng.integers(15, 35)), 2)
        corrupted.at[i, "amount"] = bad
        log_defect(
            i,
            "amount",
            "suspicious_outlier",
            f"near {old:.2f}",
            bad,
            "medium",
            "cap after approval",
            0.78,
            True,
        )

    # Inconsistent category
    category_variants = {
        "Electronics": ["ELECTRONICS", "electronics", "Electroncs"],
        "Books": ["BOOKS", "book"],
        "Home": ["HOME", "house"],
        "Fashion": ["FASHION", "clothing"],
    }
    for i in pick(max(3, n_rows // 12)):
        current = str(corrupted.at[i, "category"])
        options = [v for v in category_variants.get(current, []) if v != current]
        if not options:
            continue
        variant = str(rng.choice(options))
        corrupted.at[i, "category"] = variant
        log_defect(
            i,
            "category",
            "inconsistent_category",
            current,
            variant,
            "low",
            "normalize or map after review",
            0.90,
            False,
        )

    # Date defects
    now = pd.Timestamp.now(tz="UTC").normalize()
    for i in pick(max(4, n_rows // 10)):
        old = str(corrupted.at[i, "order_date"])
        ts = pd.Timestamp(old)
        kind = int(rng.integers(0, 6))

        if kind == 0:
            new = ts.strftime("%d %b %Y")
            log_defect(
                i,
                "order_date",
                "malformed_date",
                old,
                new,
                "low",
                "parse unambiguous date",
                0.92,
                False,
            )
        elif kind == 1:
            new = ts.strftime("%b %d, %Y")
            log_defect(
                i,
                "order_date",
                "malformed_date",
                old,
                new,
                "low",
                "parse unambiguous date",
                0.90,
                False,
            )
        elif kind == 2:
            new = "01/02/2025"
            log_defect(
                i,
                "order_date",
                "ambiguous_date",
                old,
                new,
                "medium",
                "route ambiguous date to human",
                0.60,
                True,
            )
        elif kind == 3:
            new = (now + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            log_defect(
                i,
                "order_date",
                "future_date",
                "not future",
                new,
                "high",
                "set future date to null after approval",
                0.97,
                True,
            )
        elif kind == 4:
            new = f"{old}T12:00:00+99:99"
            log_defect(
                i,
                "order_date",
                "unknown_timezone",
                "valid timezone",
                new,
                "high",
                "set unknown timezone to null",
                0.91,
                True,
            )
        else:
            new = ""
            log_defect(
                i,
                "order_date",
                "missing_required",
                old,
                new,
                "medium",
                "route missing date to source",
                0.99,
                False,
            )

        corrupted.at[i, "order_date"] = new

    # Duplicate rows appended from otherwise clean rows
    clean_indices = [i for i in range(n_rows) if i not in defective_indices]
    if clean_indices:
        num_dups = max(2, n_rows // 20)
        dup_source_idxs = rng.choice(
            clean_indices,
            size=min(num_dups, len(clean_indices)),
            replace=False,
        )

        for src in dup_source_idxs:
            row = corrupted.loc[int(src)].copy()
            cid = str(row["customer_id"])
            style = int(rng.integers(0, 4))

            if style == 0:
                new_cid = f"  {cid}"
            elif style == 1:
                new_cid = f"{cid} "
            elif style == 2:
                new_cid = cid.upper()
            else:
                new_cid = cid.lower()

            row["customer_id"] = new_cid
            new_idx = len(corrupted)
            corrupted.loc[new_idx] = row

            log_defect(
                new_idx,
                "customer_id",
                "duplicate_key",
                f"duplicate of row {int(src)}",
                new_cid,
                "high",
                "remove duplicate after preview/approval",
                0.96,
                True,
            )

            if new_cid != new_cid.strip():
                log_defect(
                    new_idx,
                    "customer_id",
                    "whitespace",
                    cid.strip(),
                    new_cid,
                    "low",
                    "trim whitespace",
                    0.99,
                    False,
                )

    return corrupted, clean, manifest


def load_uploaded_bytes(data: bytes, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File is larger than the local guardrail ({MAX_UPLOAD_MB} MB).")

    ext = os.path.splitext(filename)[1].lower()
    checksum = sha256_bytes(data)
    meta = {
        "source": "Uploaded file",
        "filename": filename,
        "size_bytes": len(data),
        "checksum_sha256": checksum,
        "processed_locally": True,
    }

    if ext == ".csv":
        last_error: Optional[Exception] = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc)
                meta["encoding_attempt"] = enc
                return df, meta
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Could not parse CSV: {last_error}")

    if ext == ".json":
        try:
            df = pd.read_json(io.BytesIO(data))
            return df, meta
        except Exception:
            df = pd.read_json(io.BytesIO(data), lines=True)
            return df, meta

    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(io.BytesIO(data))
        return df, meta

    raise ValueError(f"Unsupported file extension: {ext}")


def load_builtin_sqlite_demo(
    seed: int, n_rows: int
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    raw, reference_clean, manifest = generate_synthetic_dataset(seed, n_rows)

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = None
    try:
        con = sqlite3.connect(db_path)
        raw.to_sql("orders", con, index=False, if_exists="replace")
        con.commit()
        df = pd.read_sql_query("SELECT rowid AS __rowid, * FROM orders ORDER BY rowid", con)
        df = df.drop(columns=["__rowid"]).reset_index(drop=True)
    finally:
        if con is not None:
            con.close()
        if os.path.exists(db_path):
            os.remove(db_path)

    meta = {
        "source": "Built-in SQLite demo",
        "table": "orders",
        "checksum_sha256": dataframe_checksum(df),
        "processed_locally": True,
    }
    return df, manifest, reference_clean, meta


def get_dataset(
    source: str,
    uploaded_file: Any,
    seed: int,
    n_rows: int,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Optional[pd.DataFrame], Dict[str, Any]]:
    if source == "Synthetic demo":
        raw, reference_clean, manifest = generate_synthetic_dataset(seed, n_rows)
        meta = {
            "source": "Synthetic demo",
            "checksum_sha256": dataframe_checksum(raw),
            "seed": seed,
            "note": "Synthetic data generated locally; no personal data.",
        }
        return raw, manifest, reference_clean, meta

    if source == "Built-in SQLite demo":
        return load_builtin_sqlite_demo(seed, n_rows)

    if source == "Upload CSV/JSON/Excel":
        if uploaded_file is not None:
            try:
                data = uploaded_file.getvalue()
                df, meta = load_uploaded_bytes(data, uploaded_file.name)
                return df, [], None, meta
            except Exception as exc:
                st.error(f"Could not load uploaded file: {exc}")
                st.info("Falling back to synthetic demo so the dashboard is not blank.")
                raw, reference_clean, manifest = generate_synthetic_dataset(seed, n_rows)
                meta = {
                    "source": "Synthetic fallback after upload failure",
                    "checksum_sha256": dataframe_checksum(raw),
                    "seed": seed,
                    "error": str(exc),
                }
                return raw, manifest, reference_clean, meta
        else:
            raw, reference_clean, manifest = generate_synthetic_dataset(seed, n_rows)
            meta = {
                "source": "Synthetic fallback because no file was selected",
                "checksum_sha256": dataframe_checksum(raw),
                "seed": seed,
            }
            return raw, manifest, reference_clean, meta

    raw, reference_clean, manifest = generate_synthetic_dataset(seed, n_rows)
    meta = {"source": "Synthetic demo", "checksum_sha256": dataframe_checksum(raw), "seed": seed}
    return raw, manifest, reference_clean, meta


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        blanks = blank_mask(s)
        null_count = int(blanks.sum())
        unique_count = int(s.nunique(dropna=True))

        min_value = ""
        max_value = ""
        if pd.api.types.is_datetime64_any_dtype(s):
            if not s.dropna().empty:
                min_value = str(s.min())
                max_value = str(s.max())
        else:
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().any():
                min_value = str(numeric.min())
                max_value = str(numeric.max())

        candidate_key = (unique_count == len(df)) and (null_count == 0)
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "null_or_blank_count": null_count,
                "null_or_blank_rate": float(null_count / len(df)) if len(df) else 0.0,
                "unique_count": unique_count,
                "min": min_value,
                "max": max_value,
                "candidate_key": candidate_key,
            }
        )
    return pd.DataFrame(rows)


def parse_date_value(value: Any) -> Tuple[Optional[pd.Timestamp], str]:
    if isinstance(value, np.datetime64):
        if pd.isna(value):
            return None, "missing"
        value = pd.Timestamp(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts, "ok"

    if value is None:
        return None, "missing"

    s = str(value).strip()
    if s.lower() in {"", "nan", "none", "nat", "null"}:
        return None, "missing"

    tz_match = re.search(r"([+-]\d{2}):?(\d{2})$", s)
    if tz_match:
        hh = int(tz_match.group(1))
        mm = int(tz_match.group(2))
        if abs(hh) > 14 or mm > 59 or (abs(hh) == 14 and mm > 0):
            return None, "unknown_timezone"

    if re.match(r"^\d{4}-\d{2}-\d{2}", s) or "T" in s or s.endswith("Z"):
        try:
            ts = pd.to_datetime(s, errors="raise", utc=True)
            return ts, "ok"
        except Exception:
            pass

    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d %b %Y",
        "%b %d, %Y",
    ]
    parsed = []
    for fmt in formats:
        try:
            ts = pd.to_datetime(s, format=fmt)
            parsed.append(ts)
        except Exception:
            pass

    if not parsed:
        return None, "malformed_date"

    unique_dates = {ts.date() for ts in parsed}
    if len(unique_dates) == 1:
        return parsed[0].tz_localize("UTC"), "ok"

    return None, "ambiguous_date"


def detect_findings(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    records: List[Dict[str, Any]] = []
    now = pd.Timestamp.now(tz="UTC")
    tolerance = pd.Timedelta(days=int(config.get("future_tolerance_days", 1)))

    def add(
        row_id: int,
        column: str,
        defect_type: str,
        severity: str,
        confidence: float,
        suggested_action: str,
        destructive: bool,
        evidence: str,
        expected: str = "",
        actual: str = "",
        suggested_value: Any = None,
    ) -> None:
        records.append(
            {
                "row_id": int(row_id),
                "column": column,
                "defect_type": defect_type,
                "severity": severity,
                "confidence": float(confidence),
                "suggested_action": suggested_action,
                "destructive": bool(destructive),
                "evidence": evidence,
                "expected": expected,
                "actual": actual,
                "suggested_value": suggested_value,
            }
        )

    # Whitespace
    obj_cols = [
        c
        for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]
    for col in obj_cols:
        for idx, val in df[col].items():
            if isinstance(val, str) and val != val.strip():
                add(
                    idx,
                    col,
                    "whitespace",
                    "low",
                    0.99,
                    "trim_whitespace",
                    False,
                    f"leading/trailing whitespace in {col}",
                    expected=val.strip(),
                    actual=mask_value(val, col),
                    suggested_value=val.strip(),
                )

    # Missing required
    req_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    for col in req_cols:
        mask = blank_mask(df[col])
        for idx in df.index[mask]:
            add(
                idx,
                col,
                "missing_required",
                "medium",
                0.99,
                "mark_missing_for_source_correction",
                False,
                f"required value missing or blank in {col}",
            )

    # Duplicate key
    if "customer_id" in df.columns:
        norm = df["customer_id"].astype(str).str.strip().str.lower()
        valid = ~norm.isin(["", "nan", "none"])
        dup = norm.duplicated(keep="first") & valid
        for idx in df.index[dup]:
            add(
                idx,
                "customer_id",
                "duplicate_key",
                "high",
                0.96,
                "remove_duplicate_rows",
                True,
                f"normalized key {mask_value(norm[idx], 'customer_id')} repeats an earlier row",
            )

    # Invalid email
    if "email" in df.columns:
        for idx, val in df["email"].items():
            if is_blank_value(val):
                continue
            s = str(val).strip()
            if not EMAIL_RE.match(s):
                add(
                    idx,
                    "email",
                    "invalid_email",
                    "medium",
                    0.95,
                    "set_invalid_email_to_null_and_flag",
                    True,
                    f"email does not match pattern: {mask_value(s, 'email')}",
                )

    # Age invalid and outliers
    if "age" in df.columns:
        age_numeric = pd.to_numeric(df["age"], errors="coerce")
        for idx, val in df["age"].items():
            if is_blank_value(val):
                continue
            num = age_numeric[idx]
            if pd.isna(num):
                add(
                    idx,
                    "age",
                    "invalid_value",
                    "medium",
                    0.94,
                    "set_to_null",
                    True,
                    f"age is non-numeric: {mask_value(val, 'age')}",
                )
            elif num < 0 or num > 120:
                add(
                    idx,
                    "age",
                    "invalid_value",
                    "medium",
                    0.94,
                    "set_to_null",
                    True,
                    f"age outside 0-120: {num}",
                )

        valid_age = age_numeric[(age_numeric >= 0) & (age_numeric <= 120)]
        if len(valid_age) > 10 and float(valid_age.std()) > 0:
            z = ((valid_age - valid_age.mean()) / valid_age.std()).abs()
            for idx in z[z > float(config.get("outlier_z", 3.0))].index:
                add(
                    idx,
                    "age",
                    "suspicious_outlier",
                    "medium",
                    0.75,
                    "cap_outlier_to_valid_range",
                    True,
                    f"age z-score {z[idx]:.2f}",
                )

    # Amount invalid and outliers
    if "amount" in df.columns:
        amount_numeric = pd.to_numeric(df["amount"], errors="coerce")
        for idx, val in df["amount"].items():
            if is_blank_value(val):
                continue
            num = amount_numeric[idx]
            if pd.isna(num):
                add(
                    idx,
                    "amount",
                    "invalid_value",
                    "medium",
                    0.94,
                    "set_to_null",
                    True,
                    f"amount is non-numeric: {mask_value(val, 'amount')}",
                )
            elif num < 0:
                add(
                    idx,
                    "amount",
                    "invalid_value",
                    "high",
                    0.96,
                    "set_to_null",
                    True,
                    f"amount is negative: {num}",
                )

        valid_amount = amount_numeric[amount_numeric >= 0]
        if len(valid_amount) > 10 and float(valid_amount.std()) > 0:
            z = ((valid_amount - valid_amount.mean()) / valid_amount.std()).abs()
            for idx in z[z > float(config.get("outlier_z", 3.0))].index:
                add(
                    idx,
                    "amount",
                    "suspicious_outlier",
                    "medium",
                    0.78,
                    "cap_outlier_to_valid_range",
                    True,
                    f"amount z-score {z[idx]:.2f}",
                )

    # Category inconsistency
    if "category" in df.columns:
        for idx, val in df["category"].items():
            if is_blank_value(val):
                continue
            s = str(val)
            stripped = s.strip()
            norm = stripped.lower()
            if norm in CANONICAL_CATEGORY:
                canonical = CANONICAL_CATEGORY[norm]
                if s != canonical:
                    add(
                        idx,
                        "category",
                        "inconsistent_category",
                        "low",
                        0.98,
                        "normalize_exact",
                        False,
                        f"category variant {mask_value(s, 'category')} maps to {canonical}",
                        expected=canonical,
                        actual=mask_value(s, "category"),
                        suggested_value=canonical,
                    )
            else:
                matches = get_close_matches(norm, CANONICAL_CATEGORY.keys(), n=1, cutoff=0.72)
                if matches:
                    canonical = CANONICAL_CATEGORY[matches[0]]
                    add(
                        idx,
                        "category",
                        "inconsistent_category",
                        "medium",
                        0.65,
                        "map_close_match",
                        True,
                        f"close match to {canonical}",
                        expected=canonical,
                        actual=mask_value(s, "category"),
                        suggested_value=canonical,
                    )
                else:
                    add(
                        idx,
                        "category",
                        "invalid_category",
                        "medium",
                        0.80,
                        "set_to_null",
                        True,
                        f"category not recognized: {mask_value(s, 'category')}",
                    )

    # Status and region allowed sets
    for col, allowed in [("status", ALLOWED_STATUS), ("region", ALLOWED_REGION)]:
        if col not in df.columns:
            continue
        allowed_lower = {a.lower() for a in allowed}
        for idx, val in df[col].items():
            if is_blank_value(val):
                continue
            s = str(val).strip()
            if s.lower() not in allowed_lower:
                add(
                    idx,
                    col,
                    "invalid_value",
                    "medium",
                    0.92,
                    "set_to_null",
                    True,
                    f"{col} not in allowed set: {mask_value(s, col)}",
                )

    # Dates
    if "order_date" in df.columns:
        for idx, val in df["order_date"].items():
            parsed, status = parse_date_value(val)
            original = str(val).strip()
            is_iso_like = bool(ISO_LIKE_RE.match(original)) and isinstance(val, str)

            if status == "missing":
                continue
            if status == "ok" and parsed is not None:
                if isinstance(val, str) and not is_iso_like:
                    add(
                        idx,
                        "order_date",
                        "malformed_date",
                        "low",
                        0.90,
                        "normalize_date",
                        False,
                        f"non-ISO date string: {mask_value(original, 'order_date')}",
                        expected=parsed.strftime("%Y-%m-%d"),
                        actual=mask_value(original, "order_date"),
                        suggested_value=parsed.strftime("%Y-%m-%d"),
                    )
                if parsed > now + tolerance:
                    add(
                        idx,
                        "order_date",
                        "future_date",
                        "high",
                        0.97,
                        "set_future_date_to_null",
                        True,
                        f"future date: {parsed.date()}",
                    )
            elif status == "unknown_timezone":
                add(
                    idx,
                    "order_date",
                    "unknown_timezone",
                    "high",
                    0.91,
                    "set_to_null",
                    True,
                    f"unknown or invalid timezone offset: {mask_value(original, 'order_date')}",
                )
            elif status == "ambiguous_date":
                add(
                    idx,
                    "order_date",
                    "ambiguous_date",
                    "medium",
                    0.60,
                    "route_to_human",
                    True,
                    f"ambiguous date format: {mask_value(original, 'order_date')}",
                )
            else:
                add(
                    idx,
                    "order_date",
                    "malformed_date",
                    "medium",
                    0.90,
                    "set_to_null",
                    True,
                    f"unparseable date: {mask_value(original, 'order_date')}",
                )

    if not records:
        return empty_findings_df()
    return pd.DataFrame(records, columns=FINDINGS_COLUMNS)


def apply_safe_fixes(
    df: pd.DataFrame,
    findings: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    out = df.copy().reset_index(drop=True)
    audit: List[Dict[str, Any]] = []

    if not config.get("auto_apply_safe_fixes", True):
        return out, audit

    # Trim whitespace in string/object columns
    obj_cols = [
        c
        for c in out.columns
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c])
    ]
    for col in obj_cols:
        changed_mask = out[col].apply(lambda x: isinstance(x, str) and x != x.strip())
        count = int(changed_mask.sum())
        if count:
            ensure_object_column(out, col)
            out[col] = out[col].map(lambda x: x.strip() if isinstance(x, str) else x)
            audit.append(
                {
                    "action": "trim_whitespace",
                    "column": col,
                    "rows_changed": count,
                    "confidence": 0.99,
                    "destructive": False,
                }
            )

    # Exact category normalization
    if "category" in out.columns:
        def safe_category(x: Any) -> Any:
            if not isinstance(x, str):
                return x
            stripped = x.strip()
            return CANONICAL_CATEGORY.get(stripped.lower(), stripped)

        ensure_object_column(out, "category")
        new_cat = out["category"].map(safe_category)
        changed_mask = (out["category"].astype(str) != new_cat.astype(str)) & out[
            "category"
        ].notna()
        count = int(changed_mask.sum())
        if count:
            out["category"] = new_cat.astype(object)
            audit.append(
                {
                    "action": "normalize_exact_category",
                    "column": "category",
                    "rows_changed": count,
                    "confidence": 0.98,
                    "destructive": False,
                }
            )

    # Unambiguous date normalization
    if "order_date" in out.columns:
        changes = 0
        ensure_object_column(out, "order_date")
        new_dates = out["order_date"].astype(object).copy()
        for idx, val in out["order_date"].items():
            if not isinstance(val, str):
                continue
            s = val.strip()
            if ISO_LIKE_RE.match(s):
                continue
            if re.search(r"([+-]\d{2}:?\d{2}|Z)$", s):
                continue
            parsed, status = parse_date_value(s)
            if status == "ok" and parsed is not None:
                iso = parsed.strftime("%Y-%m-%d")
                if iso != s:
                    new_dates.at[idx] = iso
                    changes += 1
        if changes:
            out["order_date"] = new_dates
            audit.append(
                {
                    "action": "normalize_unambiguous_dates",
                    "column": "order_date",
                    "rows_changed": changes,
                    "confidence": 0.92,
                    "destructive": False,
                }
            )

    return out, audit


def build_proposals(findings: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if findings.empty:
        return empty_proposals_df()

    rows = []
    findings = findings.copy()
    findings["_group_key"] = findings.apply(
        lambda r: f"{r.defect_type}|{r.column}|{r.destructive}|{r.suggested_action}|{r.suggested_value}",
        axis=1,
    )

    for key, grp in findings.groupby("_group_key", sort=True):
        first = grp.iloc[0]
        row_ids = sorted(set(int(x) for x in grp["row_id"].tolist()))
        evidence = "; ".join(str(x) for x in grp["evidence"].head(3).tolist())
        suggested = first["suggested_value"]
        if pd.isna(suggested):
            suggested = None

        rows.append(
            {
                "proposal_id": stable_id(key),
                "defect_type": first["defect_type"],
                "column": first["column"],
                "row_count": len(row_ids),
                "row_ids": row_ids,
                "action": first["suggested_action"],
                "destructive": bool(first["destructive"]),
                "confidence": float(grp["confidence"].mean()),
                "evidence": evidence,
                "status": "pending",
                "suggested_value": suggested,
            }
        )

    return pd.DataFrame(rows, columns=PROPOSALS_COLUMNS)


def apply_decisions(
    df: pd.DataFrame,
    proposals: pd.DataFrame,
    decisions: Dict[str, str],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    out = df.copy().reset_index(drop=True)
    audit: List[Dict[str, Any]] = []
    changed = False

    if proposals.empty:
        return out, audit

    for prop in proposals.itertuples(index=False):
        row_ids = [int(x) for x in list(prop.row_ids)]
        existing = [i for i in row_ids if i in out.index]
        decision = decisions.get(prop.proposal_id, "pending")

        if decision == "pending" and not prop.destructive and not config.get(
            "require_approval_for_destructive", True
        ):
            decision = "approve"

        if decision != "approve":
            audit.append(
                {
                    "action": f"decision_{decision}",
                    "proposal_id": prop.proposal_id,
                    "defect_type": prop.defect_type,
                    "column": prop.column,
                    "rows_acknowledged": len(existing),
                }
            )
            continue

        action = prop.action
        affected = 0

        if action == "trim_whitespace":
            if prop.column in out.columns:
                ensure_object_column(out, prop.column)
                out.loc[existing, prop.column] = out.loc[existing, prop.column].map(
                    lambda x: x.strip() if isinstance(x, str) else x
                )
                affected = len(existing)
                changed = True

        elif action in {"normalize_exact", "map_close_match", "normalize_date"}:
            if (
                prop.column in out.columns
                and prop.suggested_value is not None
                and str(prop.suggested_value) != "nan"
            ):
                ensure_object_column(out, prop.column)
                out.loc[existing, prop.column] = prop.suggested_value
                affected = len(existing)
                changed = True

        elif action == "remove_duplicate_rows":
            if existing:
                out.drop(index=existing, inplace=True, errors="ignore")
                affected = len(existing)
                changed = True

        elif action in {
            "set_to_null",
            "set_invalid_email_to_null_and_flag",
            "set_future_date_to_null",
        }:
            if prop.column in out.columns and existing:
                ensure_object_column(out, prop.column)
                out.loc[existing, prop.column] = np.nan
                affected = len(existing)
                changed = True

        elif action == "cap_outlier_to_valid_range":
            if prop.column in out.columns and existing:
                ensure_object_column(out, prop.column)
                if prop.column == "age":
                    lo, hi = 0, 120
                else:
                    valid = pd.to_numeric(out[prop.column], errors="coerce")
                    valid_nonneg = valid[valid >= 0]
                    lo = 0
                    hi = float(np.nanpercentile(valid_nonneg, 99)) if len(valid_nonneg) else 10000.0
                vals = pd.to_numeric(out.loc[existing, prop.column], errors="coerce")
                out.loc[existing, prop.column] = vals.clip(lower=lo, upper=hi)
                affected = len(existing)
                changed = True

        elif action in {"mark_missing_for_source_correction", "route_to_human"}:
            affected = len(existing)

        audit.append(
            {
                "action": f"applied_{action}",
                "proposal_id": prop.proposal_id,
                "defect_type": prop.defect_type,
                "column": prop.column,
                "rows_affected": affected,
                "decision": "approve",
            }
        )

    if changed:
        out = out.reset_index(drop=True)
    return out, audit


def evaluate_detection(findings: pd.DataFrame, manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not manifest:
        return {
            "status": "not_scored",
            "reason": "No planted defect manifest is available for this input. Detectors still run, but precision/recall are not scored.",
            "precision": None,
            "recall": None,
            "f1": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "per_type": pd.DataFrame(columns=["defect_type", "tp", "fp", "fn", "precision", "recall"]),
            "worst_cases": pd.DataFrame(
                columns=["case_type", "row_id", "column", "defect_type", "input", "output", "expected", "suspected_cause"]
            ),
        }

    det_set = {(int(r.row_id), str(r.column), str(r.defect_type)) for r in findings.itertuples(index=False)}
    man_set = {(int(m["row_id"]), str(m["column"]), str(m["defect_type"])) for m in manifest}

    tp_set = det_set & man_set
    fp_set = det_set - man_set
    fn_set = man_set - det_set

    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    types = sorted({x[2] for x in det_set} | {x[2] for x in man_set})
    per_type_rows = []
    for t in types:
        t_tp = len({x for x in tp_set if x[2] == t})
        t_fp = len({x for x in fp_set if x[2] == t})
        t_fn = len({x for x in fn_set if x[2] == t})
        t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) else 0.0
        t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) else 0.0
        per_type_rows.append(
            {
                "defect_type": t,
                "tp": t_tp,
                "fp": t_fp,
                "fn": t_fn,
                "precision": t_prec,
                "recall": t_rec,
            }
        )

    manifest_by_key = {
        (int(m["row_id"]), str(m["column"]), str(m["defect_type"])): m for m in manifest
    }
    findings_by_key = {}
    for _, row in findings.iterrows():
        findings_by_key[(int(row["row_id"]), str(row["column"]), str(row["defect_type"]))] = row

    worst_rows = []
    for key in list(fn_set)[:5]:
        m = manifest_by_key.get(key, {})
        worst_rows.append(
            {
                "case_type": "missed_defect",
                "row_id": key[0],
                "column": key[1],
                "defect_type": key[2],
                "input": m.get("actual", ""),
                "output": "not detected",
                "expected": "detected",
                "suspected_cause": "detector rule coverage gap",
            }
        )
    for key in list(fp_set)[:5]:
        f = findings_by_key.get(key, {})
        worst_rows.append(
            {
                "case_type": "possible_false_positive",
                "row_id": key[0],
                "column": key[1],
                "defect_type": key[2],
                "input": str(f.get("actual", "")),
                "output": "detected",
                "expected": "not in answer key",
                "suspected_cause": "additional real defect or manifest gap",
            }
        )

    return {
        "status": "scored",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "per_type": pd.DataFrame(per_type_rows),
        "worst_cases": pd.DataFrame(worst_rows),
    }


def compute_metrics(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    findings: pd.DataFrame,
    manifest: List[Dict[str, Any]],
    config: Dict[str, Any],
    elapsed_seconds: float,
    peak_memory_bytes: int,
) -> Dict[str, Any]:
    cleaned = cleaned_df.reset_index(drop=True)
    now = pd.Timestamp.now(tz="UTC")
    tolerance = pd.Timedelta(days=int(config.get("future_tolerance_days", 1)))
    rows_out = len(cleaned)

    # Completeness
    req_cols = [c for c in REQUIRED_COLUMNS if c in cleaned.columns]
    comp_by_col: Dict[str, float] = {}
    for c in req_cols:
        present = ~blank_mask(cleaned[c])
        comp_by_col[c] = float(present.mean()) if rows_out else 0.0
    comp_overall = float(np.mean(list(comp_by_col.values()))) if comp_by_col else 0.0
    worst_comp = sorted(comp_by_col.items(), key=lambda kv: kv[1])[:5]

    # Validity
    validity_rules = []
    age_mask = None
    amount_mask = None
    date_mask = None

    def add_rule(name: str, mask: pd.Series, column: Optional[str] = None) -> None:
        if len(mask) == 0:
            return
        share = float(mask.mean())
        examples = []
        if column and column in cleaned.columns:
            bad = cleaned.loc[~mask, column].astype(str).head(3).tolist()
            examples = [mask_value(x, column) for x in bad]
        validity_rules.append({"rule": name, "share": share, "examples": examples})

    if "customer_id" in cleaned.columns:
        mask = ~blank_mask(cleaned["customer_id"])
        add_rule("customer_id_present", mask, "customer_id")

    if "email" in cleaned.columns:
        email_str = cleaned["email"].astype(str).str.strip()
        mask = (
            cleaned["email"].notna()
            & (email_str != "")
            & email_str.str.match(EMAIL_RE).fillna(False).astype(bool)
        )
        add_rule("email_format", mask, "email")

    if "age" in cleaned.columns:
        numeric = pd.to_numeric(cleaned["age"], errors="coerce")
        age_mask = numeric.notna() & (numeric >= 0) & (numeric <= 120)
        add_rule("age_range_0_120", age_mask, "age")

    if "amount" in cleaned.columns:
        numeric = pd.to_numeric(cleaned["amount"], errors="coerce")
        amount_mask = numeric.notna() & (numeric >= 0)
        add_rule("amount_nonnegative", amount_mask, "amount")

    if "category" in cleaned.columns:
        norm = cleaned["category"].astype(str).str.strip().str.lower()
        mask = ~blank_mask(cleaned["category"]) & norm.isin(CANONICAL_CATEGORY.keys())
        add_rule("category_allowed", mask, "category")

    if "status" in cleaned.columns:
        norm = cleaned["status"].astype(str).str.strip().str.lower()
        mask = ~blank_mask(cleaned["status"]) & norm.isin([s.lower() for s in ALLOWED_STATUS])
        add_rule("status_allowed", mask, "status")

    if "region" in cleaned.columns:
        norm = cleaned["region"].astype(str).str.strip().str.lower()
        mask = ~blank_mask(cleaned["region"]) & norm.isin([r.lower() for r in ALLOWED_REGION])
        add_rule("region_allowed", mask, "region")

    if "order_date" in cleaned.columns:
        parsed_list = [parse_date_value(x)[0] for x in cleaned["order_date"]]
        parsed_dates = pd.Series(parsed_list, index=cleaned.index)
        parsed_dates = pd.to_datetime(parsed_dates, utc=True, errors="coerce")
        date_mask = parsed_dates.notna() & (parsed_dates <= now + tolerance)
        add_rule("order_date_parseable_not_future", date_mask, "order_date")
    else:
        parsed_dates = pd.Series([pd.NaT] * rows_out, index=cleaned.index, dtype="datetime64[ns, UTC]")

    validity_overall = float(np.mean([r["share"] for r in validity_rules])) if validity_rules else 0.0

    # Uniqueness
    exact_key_share = 0.0
    near_pairs = 0
    scan_rows = 0
    if "customer_id" in cleaned.columns and rows_out:
        norm = cleaned["customer_id"].astype(str).str.strip().str.lower()
        valid = ~norm.isin(["", "nan", "none"])
        valid_norm = norm[valid]
        exact_key_share = float(valid_norm.nunique() / len(valid_norm)) if len(valid_norm) else 0.0
        sample = valid_norm.head(200).tolist()
        scan_rows = len(sample)
        threshold = float(config.get("duplicate_similarity_threshold", 95))
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                ratio = SequenceMatcher(None, sample[i], sample[j]).ratio() * 100.0
                if ratio >= threshold:
                    near_pairs += 1

    # Consistency
    consistency = pd.Series(True, index=cleaned.index)
    if "customer_id" in cleaned.columns:
        consistency &= ~blank_mask(cleaned["customer_id"])
    if age_mask is not None:
        consistency &= age_mask
    if amount_mask is not None:
        consistency &= amount_mask
    if date_mask is not None:
        consistency &= date_mask
    consistency_share = float(consistency.mean()) if rows_out else 0.0

    # Freshness
    newest_age_days = None
    share_within_90 = None
    if parsed_dates.notna().any():
        newest = parsed_dates.max()
        newest_age_days = float((now - newest).total_seconds() / 86400.0)
        share_within_90 = float((parsed_dates >= now - pd.Timedelta(days=90)).mean())

    # Throughput
    safe_elapsed = max(float(elapsed_seconds), 1e-9)
    rows_per_second = float(len(raw_df) / safe_elapsed)
    peak_memory_mb = float(peak_memory_bytes / (1024 * 1024))

    return {
        "completeness": {
            "overall": comp_overall,
            "by_column": comp_by_col,
            "worst": worst_comp,
        },
        "validity": {
            "overall": validity_overall,
            "rules": validity_rules,
        },
        "uniqueness": {
            "exact_key_share": exact_key_share,
            "near_duplicate_pairs": near_pairs,
            "scan_rows": scan_rows,
        },
        "consistency": {
            "share": consistency_share,
            "rule": "required fields present + age 0-120 + amount >= 0 + order_date parseable/not future",
        },
        "freshness": {
            "newest_age_days": newest_age_days,
            "share_within_90_days": share_within_90,
            "expected_window_days": 90,
        },
        "throughput": {
            "rows_per_second": rows_per_second,
            "elapsed_seconds": safe_elapsed,
            "peak_memory_mb": peak_memory_mb,
        },
        "recovery_success": {
            "share": None,
            "sample_size": 0,
            "detail": "Pending failure drills.",
        },
    }


def process_dataframe(
    raw_df: pd.DataFrame,
    manifest: List[Dict[str, Any]],
    config: Dict[str, Any],
    decisions: Dict[str, str],
    source_meta: Dict[str, Any],
) -> Dict[str, Any]:
    start = time.perf_counter()
    tracemalloc.start()

    run_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()
    config_hash = hash_config(config)
    source_meta = source_meta or {}

    df = raw_df.copy().reset_index(drop=True)
    input_checksum = dataframe_checksum(df)

    audit: List[Dict[str, Any]] = []
    warnings: List[str] = []

    def log_event(step: str, action: str, details: Dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "step": step,
            "action": action,
            "details": details,
        }
        audit.append(event)
        try:
            print(json.dumps(event, default=str))
        except Exception:
            pass

    log_event(
        "ingest",
        "loaded",
        {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "input_checksum": input_checksum,
            "source": source_meta.get("source", "unknown"),
        },
    )

    if df.empty:
        warnings.append("Input is empty. Returning an empty quality report instead of guessing.")
        profile = pd.DataFrame()
        findings = empty_findings_df()
        safe_df = df.copy()
        safe_audit = []
        remaining = findings.copy()
        proposals = empty_proposals_df()
        cleaned = df.copy()
        decision_audit = []
    else:
        profile = profile_columns(df)
        log_event("profile", "completed", {"columns_profiled": int(len(profile))})

        findings = detect_findings(df, config)
        log_event(
            "detect",
            "completed",
            {
                "findings": int(len(findings)),
                "by_type": findings["defect_type"].value_counts().to_dict() if not findings.empty else {},
            },
        )

        safe_df, safe_audit = apply_safe_fixes(df, findings, config)
        for ev in safe_audit:
            log_event("safe_fix", ev["action"], ev)

        remaining = detect_findings(safe_df, config)
        proposals = build_proposals(remaining, config)
        log_event(
            "preview",
            "proposals_built",
            {
                "proposals": int(len(proposals)),
                "destructive_pending": int(proposals["destructive"].sum()) if not proposals.empty else 0,
            },
        )

        cleaned, decision_audit = apply_decisions(safe_df, proposals, decisions, config)
        for ev in decision_audit:
            log_event("decision", ev["action"], ev)

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = compute_metrics(df, cleaned, findings, manifest, config, elapsed, peak)
    evaluation = evaluate_detection(findings, manifest)

    provenance = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config_hash": config_hash,
        "input_checksum": input_checksum,
        "code_version": "app.py-local-prototype",
        "random_seed": int(config.get("seed", 0)),
        "rows_in": int(len(df)),
        "rows_out": int(len(cleaned)),
        "source_meta": source_meta,
    }

    log_event(
        "persist",
        "version_created",
        {
            "run_id": run_id,
            "config_hash": config_hash,
            "input_checksum": input_checksum,
            "rows_out": int(len(cleaned)),
        },
    )

    return {
        "raw": df,
        "profile": profile,
        "findings": findings,
        "safe_audit": safe_audit,
        "proposals": proposals,
        "cleaned": cleaned,
        "audit": audit,
        "metrics": metrics,
        "evaluation": evaluation,
        "provenance": provenance,
        "warnings": warnings,
    }


@st.cache_data(show_spinner=False)
def run_failure_drills(seed: int, n_rows: int, outlier_z: float, future_days: int) -> pd.DataFrame:
    cfg = {
        "seed": seed,
        "n_rows": n_rows,
        "outlier_z": outlier_z,
        "duplicate_similarity_threshold": 95,
        "auto_apply_safe_fixes": True,
        "require_approval_for_destructive": True,
        "max_preview_rows": 10,
        "future_tolerance_days": future_days,
    }
    decisions: Dict[str, str] = {}
    results = []

    def check(name: str, fn: Any) -> None:
        try:
            passed, detail = fn()
            results.append({"test": name, "passed": bool(passed), "detail": str(detail), "recovered": bool(passed)})
        except Exception as exc:
            results.append({"test": name, "passed": False, "detail": str(exc), "recovered": False})

    def t_empty():
        res = process_dataframe(pd.DataFrame(), [], cfg, {}, {"source": "empty"})
        return len(res["warnings"]) > 0, "empty input handled without crash"

    def t_header_only():
        df, meta = load_uploaded_bytes(b"customer_id,name,email\n", "header.csv")
        res = process_dataframe(df, [], cfg, {}, meta)
        return df.empty and len(res["cleaned"]) == 0, "header-only file produced zero rows"

    def t_bom():
        text = "customer_id,name\n1,Ándré"
        data = text.encode("utf-8-sig")
        df, _ = load_uploaded_bytes(data, "bom.csv")
        return "Ándré" in df["name"].astype(str).values, "BOM/UTF-8 decoded"

    def t_type_change():
        df = pd.DataFrame({"customer_id": ["A", "B"], "age": ["1", "two"]})
        res = process_dataframe(df, [], cfg, {}, {"source": "type-change"})
        return not res["findings"].empty, "mixed-type column handled"

    def t_duplicate_ws_case():
        df = pd.DataFrame({"customer_id": ["A1", "a1 ", "A1"], "name": ["a", "b", "c"]})
        res = process_dataframe(df, [], cfg, {}, {"source": "duplicate"})
        return (res["findings"]["defect_type"] == "duplicate_key").any(), "whitespace/case duplicate detected"

    def t_future_unknown_tz():
        df = pd.DataFrame({"order_date": ["2099-01-01", "2025-01-01T00:00:00+99:99"]})
        res = process_dataframe(df, [], cfg, {}, {"source": "dates"})
        f = res["findings"]
        return (
            (f["defect_type"] == "future_date").any()
            and (f["defect_type"] == "unknown_timezone").any()
        ), "future timestamp and unknown timezone detected"

    def t_new_column():
        raw, _, _ = generate_synthetic_dataset(seed, min(30, n_rows))
        raw["new_column"] = 1
        res = process_dataframe(raw, [], cfg, {}, {"source": "schema-drift"})
        return "new_column" in res["cleaned"].columns, "new column preserved"

    def t_large():
        big, _, _ = generate_synthetic_dataset(seed, n_rows * 5)
        t0 = time.perf_counter()
        res = process_dataframe(big, [], cfg, {}, {"source": "large"})
        dt = time.perf_counter() - t0
        return dt < 20.0, f"processed {len(big)} rows in {dt:.2f}s"

    def t_db_drop():
        try:
            raise RuntimeError("simulated database connection drop mid-read")
        except Exception:
            fallback, _, _ = generate_synthetic_dataset(seed, min(30, n_rows))
            res = process_dataframe(fallback, [], cfg, {}, {"source": "fallback-after-db-drop"})
            return len(res["cleaned"]) > 0, "fell back without crashing"

    check("Empty input", t_empty)
    check("Header-only file", t_header_only)
    check("Wrong encoding / BOM", t_bom)
    check("Column type changes halfway", t_type_change)
    check("Duplicate key whitespace/case", t_duplicate_ws_case)
    check("Future timestamp and unknown timezone", t_future_unknown_tz)
    check("Schema gained new column", t_new_column)
    check("5x larger file", t_large)
    check("Database connection drops mid-read", t_db_drop)

    return pd.DataFrame(results)


def figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    rgb = np.ascontiguousarray(img[:, :, :3])
    plt.close(fig)
    return rgb


def make_text_frame(title: str, subtitle: str, bullets: List[str], footer: str = "") -> np.ndarray:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis("off")

    ax.text(0.02, 0.95, title, fontsize=28, weight="bold", va="top", color="#111827")
    ax.text(0.02, 0.88, subtitle, fontsize=16, va="top", color="#374151")

    y = 0.78
    for b in bullets[:7]:
        ax.text(0.04, y, f"• {str(b)[:110]}", fontsize=15, va="top", color="#111827")
        y -= 0.085

    if footer:
        ax.text(0.02, 0.03, footer, fontsize=10, color="#6b7280")

    return figure_to_rgb(fig)


def make_table_frame(title: str, subtitle: str, df: pd.DataFrame, footer: str = "") -> np.ndarray:
    if df is None or df.empty:
        return make_text_frame(title, subtitle, ["No rows to display."], footer)

    disp = df.head(6).copy()
    if disp.columns.has_duplicates:
        disp = disp.loc[:, ~disp.columns.duplicated()]
    cols = list(disp.columns[:6])
    disp = disp[cols]

    for col in disp.columns:
        disp[col] = disp[col].astype(str).str.slice(0, 24)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis("off")

    ax.text(0.02, 0.95, title, fontsize=24, weight="bold", va="top", color="#111827")
    ax.text(0.02, 0.89, subtitle, fontsize=14, va="top", color="#374151")

    cell_text = disp.values.tolist()
    col_labels = [str(c) for c in disp.columns]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0.03, 0.08, 0.94, 0.74],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    if footer:
        ax.text(0.02, 0.02, footer, fontsize=10, color="#6b7280")

    return figure_to_rgb(fig)


def make_bar_frame(title: str, subtitle: str, series: pd.Series, footer: str = "") -> np.ndarray:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("white")
    if series.empty:
        ax.text(0.5, 0.5, "No defects detected", ha="center", va="center", fontsize=20)
    else:
        series = series.sort_values(ascending=True)
        series.plot.barh(ax=ax, color="#2563eb")
        ax.set_xlabel("Count")
        ax.set_ylabel("Defect type")
        ax.set_title(f"{title}\n{subtitle}", fontsize=18, loc="left")
        for i, v in enumerate(series.values):
            ax.text(float(v) + 0.15, i, str(int(v)), va="center", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        plt.tight_layout()

    if footer:
        ax.text(0.01, -0.12, footer, transform=ax.transAxes, fontsize=10, color="#6b7280")

    return figure_to_rgb(fig)


def encode_video(frames: List[np.ndarray], fps: int = 2) -> Optional[bytes]:
    if not frames or not AV_AVAILABLE:
        return None

    height, width = frames[0].shape[:2]
    codecs = ("libx264", "h264", "mpeg4")

    for codec in codecs:
        buf = io.BytesIO()
        container = None
        try:
            container = av.open(buf, mode="w", format="mp4")
            stream = container.add_stream(codec, rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"

            for arr in frames:
                frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

            container.close()
            value = buf.getvalue()
            if value:
                return value
        except Exception:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
    return None


def build_metrics_table(result: Dict[str, Any]) -> pd.DataFrame:
    m = result["metrics"]
    prov = result["provenance"]

    def pct(v: Optional[float]) -> str:
        return "unknown" if v is None else f"{v:.3f}"

    comp = m["completeness"]
    val = m["validity"]
    uniq = m["uniqueness"]
    cons = m["consistency"]
    fresh = m["freshness"]
    thr = m["throughput"]
    rec = m["recovery_success"]

    worst = ", ".join(f"{c}: {v:.2f}" for c, v in comp.get("worst", [])[:3]) or "n/a"
    fresh_value = "unknown" if fresh["newest_age_days"] is None else f"{fresh['newest_age_days']:.1f} days"
    fresh_note = (
        "unknown"
        if fresh["share_within_90_days"] is None
        else f"{fresh['share_within_90_days']:.2f} within 90 days"
    )
    rec_share = rec.get("share")

    rows = [
        {
            "Metric": "Completeness",
            "Value": pct(comp["overall"]),
            "Sample size": prov["rows_out"],
            "Notes": f"Worst columns: {worst}",
        },
        {
            "Metric": "Validity",
            "Value": pct(val["overall"]),
            "Sample size": prov["rows_out"],
            "Notes": f"{len(val['rules'])} declared rules checked",
        },
        {
            "Metric": "Uniqueness",
            "Value": pct(uniq["exact_key_share"]),
            "Sample size": uniq["scan_rows"],
            "Notes": f"Near-duplicate pairs found: {uniq['near_duplicate_pairs']}",
        },
        {
            "Metric": "Consistency",
            "Value": pct(cons["share"]),
            "Sample size": prov["rows_out"],
            "Notes": cons["rule"],
        },
        {
            "Metric": "Freshness",
            "Value": fresh_value,
            "Sample size": prov["rows_out"],
            "Notes": fresh_note,
        },
        {
            "Metric": "Processing throughput",
            "Value": f"{thr['rows_per_second']:.1f} rows/s",
            "Sample size": prov["rows_in"],
            "Notes": f"Elapsed {thr['elapsed_seconds']:.2f}s, peak {thr['peak_memory_mb']:.1f} MB",
        },
        {
            "Metric": "Recovery success",
            "Value": pct(rec_share) if rec_share is not None else "unknown",
            "Sample size": rec.get("sample_size", 0),
            "Notes": rec.get("detail", "Failure drills pending"),
        },
    ]
    return pd.DataFrame(rows)


def build_demo_video(result: Dict[str, Any], config: Dict[str, Any]) -> Optional[bytes]:
    prov = result["provenance"]
    ev = result["evaluation"]
    metrics_table = build_metrics_table(result)

    frames: List[np.ndarray] = []

    frames.append(
        make_text_frame(
            "Autonomous Data Cleaning Platform",
            "Live local demo · no API key · CPU only",
            [
                f"Run {prov['run_id']}",
                f"Seed {prov['random_seed']}",
                f"Rows in/out: {prov['rows_in']} → {prov['rows_out']}",
                f"Input checksum: {prov['input_checksum'][:18]}...",
                "Synthetic defects loaded automatically",
            ],
            prov["timestamp"],
        )
    )

    frames.append(
        make_table_frame(
            "Raw defective sample",
            "First rows as ingested",
            result["raw"].head(5),
            "Source: seeded synthetic generator",
        )
    )

    frames.append(
        make_table_frame(
            "Column profile",
            "Type, nulls, uniqueness, ranges",
            result["profile"].head(6),
            "Source: current run",
        )
    )

    if not result["findings"].empty:
        counts = result["findings"]["defect_type"].value_counts()
        frames.append(make_bar_frame("Detected defects", "Counts by defect type", counts, "Source: detector output"))
    else:
        frames.append(
            make_text_frame(
                "Detected defects",
                "No defects detected",
                ["Detector returned zero findings."],
                "Source: detector output",
            )
        )

    if ev["status"] == "scored":
        eval_bullets = [
            f"Precision: {ev['precision']:.3f}",
            f"Recall: {ev['recall']:.3f}",
            f"TP/FP/FN: {ev['tp']}/{ev['fp']}/{ev['fn']}",
            "Scored against planted answer key",
        ]
    else:
        eval_bullets = [
            "Not scored: no answer-key manifest for this input.",
            "Detectors still run and report evidence.",
        ]

    frames.append(
        make_text_frame(
            "Detection evaluation",
            "Detector vs planted manifest",
            eval_bullets,
            "Source: evaluation layer",
        )
    )

    safe_bullets = [
        f"{item.get('action')} on {item.get('column')} ({item.get('rows_changed')} rows)"
        for item in result.get("safe_audit", [])[:5]
    ] or ["No safe fixes applied in this configuration."]
    frames.append(
        make_text_frame(
            "Safe repairs",
            "Applied only reversible, low-risk fixes",
            safe_bullets,
            "Source: preprocessing layer",
        )
    )

    frames.append(
        make_table_frame(
            "Pending repair proposals",
            "Human approval required for destructive or ambiguous changes",
            proposals_display(result["proposals"]).head(5),
            "Preview-first review queue",
        )
    )

    frames.append(
        make_table_frame(
            "Cleaned output",
            "After approved repairs",
            result["cleaned"].head(5),
            "Source: cleaned frame",
        )
    )

    frames.append(
        make_table_frame(
            "Quality metrics",
            "Completeness, validity, uniqueness, consistency, freshness, throughput, recovery",
            metrics_table,
            "Measured locally",
        )
    )

    frames.append(
        make_text_frame(
            "Limitations",
            "Prototype, not production-ready",
            [
                "Rules and simple statistics baseline; not a universal cleaner.",
                "Low-confidence cases are routed to humans, not guessed.",
                "Synthetic demo only unless the user supplies authorized data.",
                "Video is generated locally; playback depends on browser codec support.",
                "Quality scores are evidence for a human, not a verdict.",
            ],
            "End of demo",
        )
    )

    return encode_video(frames, fps=2)


def build_markdown_report(result: Dict[str, Any], config: Dict[str, Any], drills: pd.DataFrame) -> str:
    prov = result["provenance"]
    m = result["metrics"]
    ev = result["evaluation"]

    lines = [
        "# Autonomous Data Cleaning Platform — Run Report",
        "",
        f"- Run ID: `{prov['run_id']}`",
        f"- Timestamp: `{prov['timestamp']}`",
        f"- Seed: `{prov['random_seed']}`",
        f"- Config hash: `{prov['config_hash']}`",
        f"- Input checksum: `{prov['input_checksum']}`",
        f"- Rows in/out: `{prov['rows_in']} → {prov['rows_out']}`",
        "",
        "## Summary",
        "",
        "This report was generated locally by a Streamlit prototype. No paid API key was used.",
        "Destructive or ambiguous repairs require human approval. The raw snapshot is not overwritten.",
        "",
        "## Quality metrics",
        "",
        "```text",
        build_metrics_table(result).to_string(index=False),
        "```",
        "",
        "## Detection evaluation",
        "",
    ]

    if ev["status"] == "scored":
        lines.extend(
            [
                f"- Precision: `{ev['precision']:.3f}`",
                f"- Recall: `{ev['recall']:.3f}`",
                f"- F1: `{ev['f1']:.3f}`",
                f"- TP/FP/FN: `{ev['tp']}/{ev['fp']}/{ev['fn']}`",
                "",
                "### Per defect type",
                "",
                "```text",
                ev["per_type"].to_string(index=False),
                "```",
                "",
                "### Worst cases",
                "",
                "```text",
                ev["worst_cases"].to_string(index=False),
                "```",
            ]
        )
    else:
        lines.extend(["- Status: not scored.", f"- Reason: {ev['reason']}"])

    lines.extend(
        [
            "",
            "## Failure drills",
            "",
            "```text",
            drills.to_string(index=False) if not drills.empty else "No drills available.",
            "```",
            "",
            "## Limitations",
            "",
            "- This is a prototype, not a production data-cleaning service.",
            "- It uses transparent rules and simple statistics; it does not prove general-purpose autonomous cleaning.",
            "- Uploaded data is processed in the local session only; no external transmission is implemented.",
            "- Low-confidence cases are intentionally routed to human review instead of being auto-repaired.",
            "- Video playback depends on local codec support in the browser.",
            "",
            "## Reproducibility",
            "",
            f"- Configuration hash: `{prov['config_hash']}`",
            f"- Input checksum: `{prov['input_checksum']}`",
            f"- Random seed: `{prov['random_seed']}`",
            "",
        ]
    )
    return "\n".join(lines)


def animate_pipeline(result: Dict[str, Any], area: st.container) -> None:
    with area:
        st.subheader("🟢 Live demo starting")
        progress = st.progress(0.0)
        status = st.empty()
        log = st.empty()
        preview = st.empty()

        steps = [
            ("Loading seeded synthetic defects", result["raw"].head(3)),
            ("Profiling columns", result["profile"].head(5)),
            ("Detecting quality defects", result["findings"].head(5)),
            ("Applying safe, reversible fixes", pd.DataFrame(result["safe_audit"])),
            ("Preparing preview-first repair proposals", proposals_display(result["proposals"]).head(5)),
            ("Building cleaned output and audit trail", result["cleaned"].head(3)),
        ]

        for i, (msg, tbl) in enumerate(steps, start=1):
            status.markdown(f"**Step {i}/{len(steps)}:** {msg}")
            recent = result["audit"][-min(len(result["audit"]), i * 2) :]
            log_text = "\n".join(
                f"{e.get('timestamp', '')} | {e.get('step', '')} | {e.get('action', '')}"
                for e in recent
            )
            log.code(log_text or "Waiting for events...", language="text")
            preview.dataframe(tbl, use_container_width=True)
            progress.progress(float(i / len(steps)))
            time.sleep(0.45)

        status.markdown(
            "**Demo complete.** Scroll down to inspect evidence, approve repairs, and watch the generated video."
        )
        progress.progress(1.0)
        time.sleep(0.25)


def main() -> None:
    init_session()

    uploaded_file = None

    with st.sidebar:
        st.title("⚙️ Controls")

        sources = ["Synthetic demo", "Upload CSV/JSON/Excel", "Built-in SQLite demo"]
        current_index = (
            sources.index(st.session_state.active_source)
            if st.session_state.active_source in sources
            else 0
        )
        source = st.radio("Data source", sources, index=current_index)

        if source != st.session_state.active_source:
            st.session_state.active_source = source
            st.session_state.auto_start = True
            st.session_state.video_bytes = None
            st.session_state.video_key = ""
            st.session_state.audit = []
            st.session_state.proposals_decisions = {}

        st.session_state.seed = int(st.number_input("Random seed", value=int(st.session_state.seed), step=1))
        st.session_state.n_rows = int(
            st.slider(
                "Demo rows",
                30,
                500,
                int(st.session_state.n_rows),
                step=10,
                help="Number of synthetic rows generated for the demo.",
            )
        )
        st.session_state.outlier_z = float(
            st.slider(
                "Outlier z-score threshold",
                1.5,
                5.0,
                float(st.session_state.outlier_z),
                0.1,
                help="Numeric values with absolute z-score above this threshold are flagged as suspicious outliers.",
            )
        )
        st.session_state.duplicate_similarity_threshold = int(
            st.slider(
                "Near-duplicate similarity (%)",
                70,
                100,
                int(st.session_state.duplicate_similarity_threshold),
                help="Used for near-duplicate pair scanning. Exact normalized duplicates are always detected.",
            )
        )
        st.session_state.auto_apply_safe_fixes = st.checkbox(
            "Auto-apply safe fixes",
            value=bool(st.session_state.auto_apply_safe_fixes),
            help="Trim whitespace, normalize exact category casing/synonyms, and parse unambiguous dates only.",
        )
        st.session_state.require_approval_for_destructive = st.checkbox(
            "Require approval for proposed changes",
            value=bool(st.session_state.require_approval_for_destructive),
            help="Destructive changes always require approval. When unchecked, low-risk proposals may be auto-approved.",
        )
        st.session_state.max_preview_rows = int(st.selectbox("Preview rows", [5, 10, 20, 50], index=1))
        st.session_state.future_tolerance_days = int(
            st.number_input(
                "Future-date tolerance (days)",
                0,
                365,
                int(st.session_state.future_tolerance_days),
            )
        )

        if source == "Upload CSV/JSON/Excel":
            uploaded_file = st.file_uploader(
                "Upload a CSV, JSON, or Excel file",
                type=["csv", "json", "xlsx", "xls"],
                help="Files are processed locally in this Streamlit session.",
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ Run", use_container_width=True):
                st.session_state.auto_start = True
                st.rerun()
        with c2:
            if st.button("🔁 Reset", use_container_width=True):
                st.session_state.auto_start = True
                st.session_state.video_bytes = None
                st.session_state.video_key = ""
                st.session_state.audit = []
                st.session_state.proposals_decisions = {}
                st.rerun()

        if st.button("🗑️ Delete my data", use_container_width=True):
            for key in [
                "uploaded_df",
                "uploaded_meta",
                "last_result",
                "video_bytes",
                "video_key",
                "audit",
                "proposals_decisions",
            ]:
                st.session_state.pop(key, None)
            st.success("Cleared uploaded data and run artifacts from this browser session.")
            st.rerun()

    config = {
        "seed": int(st.session_state.seed),
        "n_rows": int(st.session_state.n_rows),
        "outlier_z": float(st.session_state.outlier_z),
        "duplicate_similarity_threshold": int(st.session_state.duplicate_similarity_threshold),
        "auto_apply_safe_fixes": bool(st.session_state.auto_apply_safe_fixes),
        "require_approval_for_destructive": bool(st.session_state.require_approval_for_destructive),
        "max_preview_rows": int(st.session_state.max_preview_rows),
        "future_tolerance_days": int(st.session_state.future_tolerance_days),
    }

    raw_df, manifest, reference_clean, source_meta = get_dataset(
        st.session_state.active_source,
        uploaded_file,
        config["seed"],
        config["n_rows"],
    )

    decisions = st.session_state.proposals_decisions
    result = process_dataframe(raw_df, manifest, config, decisions, source_meta)

    st.session_state.last_result = result
    st.session_state.audit = result["audit"]
    st.session_state.run_history.append(result["provenance"])

    st.title("🧹 Autonomous Data Cleaning Platform — Live Prototype")
    st.caption(
        "Runs locally on CPU. Imports CSV/JSON/Excel/SQLite, detects common quality defects, "
        "applies only safe reversible fixes by default, and routes destructive or ambiguous repairs to human review."
    )

    animation_area = st.container()
    if st.session_state.auto_start:
        animate_pipeline(result, animation_area)
        st.session_state.auto_start = False
    else:
        with animation_area:
            st.caption("Live demo initialized. Use ▶️ Run in the sidebar to replay the simulation.")

    drills = run_failure_drills(
        config["seed"],
        config["n_rows"],
        config["outlier_z"],
        config["future_tolerance_days"],
    )
    recovery_share = float(drills["passed"].mean()) if not drills.empty else 0.0
    result["metrics"]["recovery_success"] = {
        "share": recovery_share,
        "sample_size": int(len(drills)),
        "detail": "Automated failure drills executed in this app session.",
    }
    result["failure_drills"] = drills

    decision_hash = hashlib.sha1(
        json.dumps(decisions, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    video_key = f"{result['provenance']['input_checksum']}|{result['provenance']['config_hash']}|{decision_hash}"

    if st.session_state.video_key != video_key or st.session_state.video_bytes is None:
        with st.spinner("Generating live demo video locally..."):
            st.session_state.video_bytes = build_demo_video(result, config)
            st.session_state.video_key = video_key

    video_bytes = st.session_state.video_bytes

    tabs = st.tabs(
        [
            "Overview",
            "Raw & Manifest",
            "Detect",
            "Review & Repair",
            "Cleaned Output",
            "Metrics & Video",
            "Audit",
        ]
    )

    max_preview = config["max_preview_rows"]

    with tabs[0]:
        st.markdown(
            """
            ### What this prototype does
            - Loads synthetic or user-provided CSV/JSON/Excel/SQLite data locally.
            - Profiles columns for type, nulls, uniqueness, ranges, and candidate keys.
            - Detects duplicates, invalid values, inconsistent categories, malformed dates, outliers, and future timestamps.
            - Applies safe, reversible fixes by default.
            - Keeps destructive or ambiguous repairs in a preview-first review queue.
            - Records provenance, audit events, metrics, and exports a report.

            ### What it does not do
            - It is not production-ready.
            - It does not invent missing values.
            - It does not auto-apply destructive changes without approval.
            - It does not send uploaded data to any external API.
            """
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Run ID", result["provenance"]["run_id"])
        c2.metric("Rows in", result["provenance"]["rows_in"])
        c3.metric("Rows out", result["provenance"]["rows_out"])
        c4.metric("Peak memory", f"{result['metrics']['throughput']['peak_memory_mb']:.1f} MB")

        st.caption(
            f"Input checksum: `{result['provenance']['input_checksum'][:20]}...` · "
            f"Config hash: `{result['provenance']['config_hash']}` · "
            f"Seed: `{config['seed']}`"
        )

        if result["warnings"]:
            for w in result["warnings"]:
                st.warning(w)

        with st.expander("Step 1 — Ingestion & raw snapshot", expanded=False):
            st.json(source_meta)
            st.dataframe(result["raw"].head(max_preview), use_container_width=True)

        with st.expander("Step 2 — Column profiling", expanded=False):
            st.dataframe(result["profile"].head(max_preview * 2), use_container_width=True)

        with st.expander("Step 3 — Detection", expanded=False):
            st.dataframe(result["findings"].head(max_preview * 5), use_container_width=True)

        with st.expander("Step 4 — Preview-first cleaning", expanded=False):
            st.dataframe(pd.DataFrame(result["safe_audit"]), use_container_width=True)
            st.dataframe(proposals_display(result["proposals"]).head(max_preview * 3), use_container_width=True)

        with st.expander("Step 5 — Lineage & audit", expanded=False):
            st.json(result["provenance"])
            st.dataframe(pd.DataFrame(result["audit"]).tail(max_preview * 5), use_container_width=True)

        with st.expander("Step 6 — Evaluation", expanded=False):
            st.dataframe(build_metrics_table(result), use_container_width=True)

    with tabs[1]:
        st.subheader("Raw snapshot")
        st.dataframe(result["raw"].head(max_preview * 5), use_container_width=True)
        st.json(source_meta)

        st.subheader("Planted defect manifest")
        if manifest:
            manifest_df = pd.DataFrame(manifest)
            st.dataframe(manifest_df.head(max_preview * 10), use_container_width=True)
            st.download_button(
                "Download manifest JSON",
                json.dumps(manifest, indent=2, default=str).encode("utf-8"),
                file_name="defect_manifest.json",
                mime="application/json",
            )
        else:
            st.info(
                "No answer-key manifest is available for this input. Detectors still run, "
                "but precision/recall are not scored."
            )

    with tabs[2]:
        st.subheader("Detected findings")
        findings = result["findings"]
        if findings.empty:
            st.success("No defects detected by the current rule set.")
        else:
            counts = (
                findings["defect_type"]
                .value_counts()
                .rename_axis("defect_type")
                .reset_index(name="count")
            )
            st.bar_chart(counts, x="defect_type", y="count")
            st.caption(f"Source: current run · date: {result['provenance']['timestamp']}")
            st.dataframe(findings.head(max_preview * 20), use_container_width=True)

        st.subheader("Evaluation against manifest")
        ev = result["evaluation"]
        if ev["status"] == "scored":
            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{ev['precision']:.3f}")
            m2.metric("Recall", f"{ev['recall']:.3f}")
            m3.metric("F1", f"{ev['f1']:.3f}")
            st.dataframe(ev["per_type"], use_container_width=True)
            st.subheader("Ten worst cases")
            st.dataframe(ev["worst_cases"].head(10), use_container_width=True)
        else:
            st.warning(ev["reason"])

    with tabs[3]:
        st.subheader("Safe fixes already applied")
        if result["safe_audit"]:
            st.dataframe(pd.DataFrame(result["safe_audit"]), use_container_width=True)
        else:
            st.info("No safe fixes were applied in this configuration.")

        st.subheader("Repair proposals")
        if result["proposals"].empty:
            st.success("No pending repair proposals.")
        else:
            st.caption(
                "Approve, reject, or defer each proposed change. "
                "Destructive changes are never applied automatically."
            )
            options = ["pending", "approve", "reject", "defer"]

            for prop in result["proposals"].itertuples(index=False):
                st.markdown(
                    f"**{prop.defect_type}** — `{prop.column}` — {prop.row_count} rows — "
                    f"confidence {prop.confidence:.2f} — destructive: {prop.destructive}"
                )
                st.caption(f"Evidence: {prop.evidence}")
                current = st.session_state.proposals_decisions.get(prop.proposal_id, "pending")
                if current not in options:
                    current = "pending"
                choice = st.radio(
                    "Decision",
                    options,
                    index=options.index(current),
                    key=f"decision_{prop.proposal_id}",
                    horizontal=True,
                )
                if choice != current:
                    st.session_state.proposals_decisions[prop.proposal_id] = choice
                    st.rerun()

            if st.button("Clear all decisions"):
                st.session_state.proposals_decisions = {}
                st.rerun()

    with tabs[4]:
        st.subheader("Before / after")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw (defective)**")
            st.dataframe(result["raw"].head(max_preview), use_container_width=True)
        with col2:
            st.markdown("**Cleaned (after approved repairs)**")
            st.dataframe(result["cleaned"].head(max_preview), use_container_width=True)

        st.subheader("Downloads")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "Download cleaned CSV",
                result["cleaned"].to_csv(index=False).encode("utf-8"),
                file_name="cleaned_output.csv",
                mime="text/csv",
            )
        with d2:
            try:
                st.download_button(
                    "Download cleaned JSON",
                    result["cleaned"].to_json(orient="records", indent=2, date_format="iso").encode("utf-8"),
                    file_name="cleaned_output.json",
                    mime="application/json",
                )
            except Exception as exc:
                st.warning(f"JSON export unavailable: {exc}")
        with d3:
            try:
                parquet_buf = io.BytesIO()
                result["cleaned"].to_parquet(parquet_buf, index=False, engine="pyarrow")
                st.download_button(
                    "Download cleaned Parquet",
                    parquet_buf.getvalue(),
                    file_name="cleaned_output.parquet",
                    mime="application/octet-stream",
                )
            except Exception as exc:
                st.warning(f"Parquet export unavailable: {exc}")

    with tabs[5]:
        st.subheader("Quality metrics")
        st.dataframe(build_metrics_table(result), use_container_width=True)

        st.subheader("Validity rules")
        rules = result["metrics"]["validity"]["rules"]
        if rules:
            st.dataframe(pd.DataFrame(rules), use_container_width=True)
        else:
            st.info("No validity rules were applicable.")

        st.subheader("Failure drills / recovery")
        if "failure_drills" in result and not result["failure_drills"].empty:
            st.dataframe(result["failure_drills"], use_container_width=True)
        else:
            st.info("No failure drill results available.")

        st.subheader("Live demo video")
        if video_bytes:
            st.video(video_bytes, loop=True, autoplay=True, muted=True)
            st.download_button(
                "Download MP4 demo video",
                video_bytes,
                file_name="autonomous_data_cleaning_demo.mp4",
                mime="video/mp4",
            )
        else:
            st.warning(
                "Video encoder unavailable. Install the `av` package and restart Streamlit. "
                "The rest of the app still works without video generation."
            )

    with tabs[6]:
        st.subheader("Audit log")
        audit_df = pd.DataFrame(result["audit"])
        st.dataframe(audit_df.tail(max_preview * 20), use_container_width=True)

        st.download_button(
            "Download audit JSON",
            json.dumps(result["audit"], indent=2, default=str).encode("utf-8"),
            file_name="audit_log.json",
            mime="application/json",
        )

        report_md = build_markdown_report(result, config, result.get("failure_drills", pd.DataFrame()))
        st.download_button(
            "Download Markdown report",
            report_md.encode("utf-8"),
            file_name="run_report.md",
            mime="text/markdown",
        )

        st.subheader("Limitations")
        st.markdown(
            """
            - This prototype is not production-ready.
            - It uses transparent rules and simple statistics, not a universal learned cleaner.
            - It does not invent missing values; unknown cases are routed to human review.
            - It does not overwrite the raw snapshot in the session, but durable storage is not implemented.
            - Metrics are evidence for a human reviewer, not an automatic verdict.
            """
        )


if __name__ == "__main__":
    main()