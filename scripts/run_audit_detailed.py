import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qa.detailed_pipeline import run_audit_detailed
from qa.input_loader import csv_row_to_audit_input, load_audit_input, scenario_to_audit_input

CATEGORIES = ["understandable", "preferred_tone_followed", "empathy", "personalization"]


def _print_usage() -> None:
    print(
        (
            "Usage: python scripts/run_audit_detailed.py <input.json|input.csv> "
            "[--row N] [--send-id ID] [--all] [--workers N] [--out-dir DIR]\n"
            "Defaults: audits all rows/scenarios and writes timestamped files to ./output"
        ),
        file=sys.stderr,
    )


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def _count_json_rows(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
        return len(data["scenarios"])
    return 1


def _count_input_rows(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _count_csv_rows(path)
    if suffix == ".json":
        return _count_json_rows(path)
    raise ValueError(f"Unsupported input file type: {path.suffix}. Use .json or .csv.")


def _load_all_audit_inputs(path: Path) -> tuple[list[tuple[int, object]], list[str]]:
    suffix = path.suffix.lower()
    loaded: list[tuple[int, object]] = []
    failures: list[str] = []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV has no header row.")

            for i, row in enumerate(reader, start=1):
                try:
                    loaded.append((i, csv_row_to_audit_input(row, i)))
                except Exception as exc:
                    failures.append(f"row {i}: {exc}")
        return loaded, failures

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
            for i, scenario in enumerate(data.get("scenarios", []), start=1):
                try:
                    loaded.append((i, scenario_to_audit_input(scenario, i)))
                except Exception as exc:
                    failures.append(f"row {i}: {exc}")
            return loaded, failures

        try:
            loaded.append((1, load_audit_input(path, row_num=1)))
        except Exception as exc:
            failures.append(f"row 1: {exc}")
        return loaded, failures

    raise ValueError(f"Unsupported input file type: {path.suffix}. Use .json or .csv.")


def _flatten_for_csv(obj: dict) -> dict:
    row = {
        "id": obj.get("id", ""),
        "message_text": obj.get("message_text", ""),
        "preferred_tone": obj.get("preferred_tone", ""),
    }

    category_results = obj.get("category_results", {})
    for category in CATEGORIES:
        cat = category_results.get(category, {})
        row[f"{category}_score"] = cat.get("score", "")
        row[f"{category}_justification"] = cat.get("justification", "")
        row[f"{category}_flagged_words_or_phrases"] = json.dumps(
            cat.get("flagged_words_or_phrases", []), ensure_ascii=False
        )

    return row


def _write_outputs(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"audited-detailed-{stamp}.json"
    csv_path = out_dir / f"audited-detailed-{stamp}.csv"

    json_path.write_text(f"{json.dumps(results, indent=2, ensure_ascii=False)}\n", encoding="utf-8")

    rows = [_flatten_for_csv(r) for r in results]
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(results)} audit result(s) to {json_path}")
    print(f"Wrote {len(results)} audit result(s) to {csv_path}")


def _format_hh_mm_ss(total_seconds: float) -> str:
    secs = max(0, int(round(total_seconds)))
    hours = secs // 3600
    minutes = (secs % 3600) // 60
    seconds = secs % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def _print_timing_summary(start_ts: float, audits_completed: int) -> None:
    elapsed_s = time.perf_counter() - start_ts
    avg_s = (elapsed_s / audits_completed) if audits_completed > 0 else 0.0
    print(f"Total runtime: {_format_hh_mm_ss(elapsed_s)}")
    print(f"Average seconds per audit: {avg_s:.2f}")


def main():
    try:
        run_start_ts = time.perf_counter()
        if len(sys.argv) < 2:
            _print_usage()
            sys.exit(1)

        p = Path(sys.argv[1])
        row_num = 1
        row_set = False
        send_id = None
        audit_all = True
        explicit_all = False
        workers = 1
        out_dir: Path | None = Path("output")

        i = 2
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg in ("-h", "--help"):
                _print_usage()
                sys.exit(0)
            if arg == "--all":
                explicit_all = True
                audit_all = True
                i += 1
                continue
            if arg == "--out-dir":
                if i + 1 >= len(sys.argv):
                    raise ValueError("--out-dir requires a directory path.")
                out_dir = Path(sys.argv[i + 1])
                i += 2
                continue
            if arg == "--workers":
                if i + 1 >= len(sys.argv):
                    raise ValueError("--workers requires a numeric value.")
                workers = int(sys.argv[i + 1])
                if workers < 1:
                    raise ValueError("--workers must be >= 1.")
                i += 2
                continue
            if arg == "--row":
                if i + 1 >= len(sys.argv):
                    raise ValueError("--row requires a numeric value.")
                row_num = int(sys.argv[i + 1])
                row_set = True
                i += 2
                continue
            if arg == "--send-id":
                if i + 1 >= len(sys.argv):
                    raise ValueError("--send-id requires a value.")
                send_id = sys.argv[i + 1]
                i += 2
                continue
            raise ValueError(f"Unknown argument: {arg}")

        if audit_all and send_id:
            raise ValueError("Use either --all or --send-id, not both.")

        if not explicit_all and (row_set or send_id is not None):
            audit_all = False

        if audit_all:
            loaded_rows, failures = _load_all_audit_inputs(p)
            if not loaded_rows:
                raise ValueError(
                    "All rows failed during input parsing. First failure: "
                    f"{failures[0] if failures else 'unknown error'}"
                )

            results_by_row: dict[int, dict] = {}

            if workers == 1:
                for idx, audit_in in tqdm(loaded_rows, desc="Auditing", unit="row"):
                    try:
                        out = run_audit_detailed(audit_in)
                        results_by_row[idx] = out
                    except Exception as row_exc:
                        failures.append(f"row {idx}: {row_exc}")
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_row = {
                        executor.submit(run_audit_detailed, audit_in): idx for idx, audit_in in loaded_rows
                    }
                    for future in tqdm(
                        as_completed(future_to_row),
                        total=len(future_to_row),
                        desc="Auditing",
                        unit="row",
                    ):
                        idx = future_to_row[future]
                        try:
                            out = future.result()
                            results_by_row[idx] = out
                        except Exception as row_exc:
                            failures.append(f"row {idx}: {row_exc}")

            results = [results_by_row[idx] for idx in sorted(results_by_row.keys())]

            if not results:
                raise ValueError(
                    "All rows failed during audit. First failure: "
                    f"{failures[0] if failures else 'unknown error'}"
                )

            _write_outputs(results, out_dir or Path("output"))
            _print_timing_summary(run_start_ts, len(results))

            if failures:
                print(
                    f"Warning: {len(failures)} row(s) failed. First failure: {failures[0]}",
                    file=sys.stderr,
                )
            return

        audit_in = load_audit_input(p, row_num=row_num, send_id=send_id)
        out = run_audit_detailed(audit_in)

        _write_outputs([out], out_dir or Path("output"))
        _print_timing_summary(run_start_ts, 1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
