from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

OUTPUT_DIR = DATA_ROOT / "runtime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = OUTPUT_DIR / "flow_metadata_audit.json"


# ---------------------------------------------------------
# Logical fields we care about
# ---------------------------------------------------------

FIELD_ALIASES = {
    "label": [
        "Label",
        "label",
    ],
    "flow_id": [
        "Flow ID",
        "FlowID",
        "flow_id",
    ],
    "source_ip": [
        "Source IP",
        "Src IP",
        "SourceIP",
        "SrcIP",
    ],
    "destination_ip": [
        "Destination IP",
        "Dst IP",
        "DestinationIP",
        "DstIP",
    ],
    "source_port": [
        "Source Port",
        "Src Port",
        "SourcePort",
        "SrcPort",
    ],
    "destination_port": [
        "Destination Port",
        "Dst Port",
        "DestinationPort",
        "DstPort",
    ],
    "protocol": [
        "Protocol",
        "protocol",
    ],
    "timestamp": [
        "Timestamp",
        "timestamp",
    ],
}


def find_column(columns, aliases):
    """Return the first matching real column name."""
    normalized = {
        str(col).strip().lower(): col
        for col in columns
    }

    for alias in aliases:
        key = alias.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def analyze_field(df, column):
    if column is None:
        return {
            "present": False,
            "column": None,
        }

    s = df[column]

    result = {
        "present": True,
        "column": str(column),
        "dtype": str(s.dtype),
        "rows": int(len(s)),
        "missing_count": int(s.isna().sum()),
        "missing_rate": float(s.isna().mean()),
        "unique_count": int(s.nunique(dropna=True)),
    }

    numeric = safe_numeric(s)

    numeric_valid = numeric.notna()

    if numeric_valid.any():
        result["numeric_valid_count"] = int(numeric_valid.sum())
        result["numeric_valid_rate"] = float(numeric_valid.mean())

        result["zero_count"] = int((numeric == 0).sum())
        result["zero_rate"] = float((numeric == 0).mean())

        finite = numeric.isna() | numeric.notna()

        # Explicit Inf detection
        try:
            inf_mask = numeric.isin([float("inf"), float("-inf")])
            result["inf_count"] = int(inf_mask.sum())
        except Exception:
            result["inf_count"] = None

        nonzero = numeric[numeric.notna() & (numeric != 0)]

        if len(nonzero):
            result["min_nonzero"] = float(nonzero.min())
            result["max_nonzero"] = float(nonzero.max())

    return result


def analyze_file(path: Path):
    print()
    print("=" * 90)
    print(f"FILE: {path.relative_to(PROJECT_ROOT)}")
    print("=" * 90)

    # Read only metadata candidates + label.
    # We first inspect schema to avoid loading unnecessary columns.
    try:
        schema_df = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        return {
            "file": str(path.relative_to(PROJECT_ROOT)),
            "error": repr(exc),
        }

    columns = list(schema_df.columns)

    resolved = {}

    for logical_name, aliases in FIELD_ALIASES.items():
        resolved[logical_name] = find_column(columns, aliases)

    print(f"Rows: {len(schema_df):,}")
    print(f"Columns: {len(columns)}")

    print("\nResolved fields:")

    for logical_name, column in resolved.items():
        status = "YES" if column is not None else "NO"
        print(f"  {logical_name:<20} {status:<4} {column}")

    result = {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "rows": int(len(schema_df)),
        "column_count": int(len(columns)),
        "columns": [str(c) for c in columns],
        "resolved_fields": resolved,
        "fields": {},
    }

    print("\nField statistics:")

    for logical_name, column in resolved.items():
        stats = analyze_field(schema_df, column)
        result["fields"][logical_name] = stats

        if not stats["present"]:
            print(f"  {logical_name:<20} MISSING")
            continue

        print(
            f"  {logical_name:<20} "
            f"missing={stats['missing_rate']:.2%} "
            f"unique={stats['unique_count']:,}",
            end=""
        )

        if "zero_rate" in stats:
            print(f" zero={stats['zero_rate']:.2%}")
        else:
            print()

    # Label distribution
    label_col = resolved["label"]

    if label_col is not None:
        labels = (
            schema_df[label_col]
            .astype(str)
            .value_counts(dropna=False)
        )

        result["label_distribution"] = {
            str(k): int(v)
            for k, v in labels.items()
        }

        print("\nTop labels:")

        for label, count in labels.head(15).items():
            print(f"  {label:<35} {count:,}")

    return result


def main():
    parquet_files = sorted(DATA_ROOT.rglob("*.parquet"))

    if not parquet_files:
        raise SystemExit(
            f"No parquet files found under: {DATA_ROOT}"
        )

    print("=" * 90)
    print("ADIS — M6.2.1 FLOW METADATA AVAILABILITY AUDIT")
    print("=" * 90)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data root    : {DATA_ROOT}")
    print(f"Parquet files: {len(parquet_files)}")

    reports = []

    for path in parquet_files:
        reports.append(analyze_file(path))

    # -----------------------------------------------------
    # Global summary
    # -----------------------------------------------------

    summary = {}

    logical_fields = list(FIELD_ALIASES.keys())

    for field in logical_fields:
        present = 0
        total = 0
        usable = 0

        for report in reports:
            if "fields" not in report:
                continue

            total += 1

            stats = report["fields"].get(field, {})

            if stats.get("present"):
                present += 1

                missing_rate = stats.get("missing_rate", 1.0)
                zero_rate = stats.get("zero_rate", 0.0)

                if missing_rate < 1.0 and zero_rate < 1.0:
                    usable += 1

        summary[field] = {
            "files_checked": total,
            "files_with_field": present,
            "files_with_usable_values": usable,
        }

    # -----------------------------------------------------
    # Important metadata conclusion
    # -----------------------------------------------------

    port_fields = ["source_port", "destination_port"]

    ports_available_files = 0
    ports_usable_files = 0

    for report in reports:
        fields = report.get("fields", {})

        src = fields.get("source_port", {})
        dst = fields.get("destination_port", {})

        if src.get("present") and dst.get("present"):
            ports_available_files += 1

        src_usable = (
            src.get("present")
            and src.get("missing_rate", 1.0) < 1.0
            and src.get("zero_rate", 1.0) < 1.0
        )

        dst_usable = (
            dst.get("present")
            and dst.get("missing_rate", 1.0) < 1.0
            and dst.get("zero_rate", 1.0) < 1.0
        )

        if src_usable and dst_usable:
            ports_usable_files += 1

    conclusion = {
        "ports_present_in_both_files": ports_available_files,
        "ports_usable_in_both_files": ports_usable_files,
    }

    final_report = {
        "audit": "M6.2.1",
        "description": "Flow metadata availability audit for ADIS",
        "files_checked": len(reports),
        "files": reports,
        "global_summary": summary,
        "port_summary": conclusion,
    }

    REPORT_JSON.write_text(
        json.dumps(
            final_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("GLOBAL SUMMARY")
    print("=" * 90)

    for field, stats in summary.items():
        print(
            f"{field:<20} "
            f"present={stats['files_with_field']}/{stats['files_checked']} "
            f"usable={stats['files_with_usable_values']}/{stats['files_checked']}"
        )

    print()
    print("=" * 90)
    print("PORT AVAILABILITY")
    print("=" * 90)

    print(
        f"Files with Source+Destination Port columns : "
        f"{ports_available_files}/{len(reports)}"
    )

    print(
        f"Files with usable Source+Destination Ports : "
        f"{ports_usable_files}/{len(reports)}"
    )

    print()
    print(f"Report saved to:")
    print(REPORT_JSON)


if __name__ == "__main__":
    main()