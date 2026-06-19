#!/usr/bin/env python
import json
import os
import platform
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


def load_junit_report(path: Path):
    if not path.exists():
        return {
            "total": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "time": 0.0,
            "failed_tests": [],
        }

    tree = ET.parse(path)
    root = tree.getroot()
    suites = root.findall(".//testsuite")

    totals = {"total": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    failed_tests = []

    for suite in suites:
        totals["total"] += int(suite.attrib.get("tests", 0))
        totals["failures"] += int(suite.attrib.get("failures", 0))
        totals["errors"] += int(suite.attrib.get("errors", 0))
        totals["skipped"] += int(suite.attrib.get("skipped", 0))
        totals["time"] += float(suite.attrib.get("time", 0.0))

        for case in suite.findall("testcase"):
            failure_node = case.find("failure")
            error_node = case.find("error")
            if failure_node is not None or error_node is not None:
                node = failure_node or error_node
                failed_tests.append(
                    {
                        "name": case.attrib.get("name"),
                        "file": case.attrib.get("file", "unknown"),
                        "message": (node.attrib.get("message") or "").strip(),
                    }
                )

    return {**totals, "failed_tests": failed_tests}


def load_coverage(path: Path):
    if not path.exists():
        return {"percent": 0.0, "lowest": []}

    with path.open() as f:
        data = json.load(f)

    totals = data.get("totals", {})
    percent = totals.get("percent_covered", 0.0)

    files = data.get("files", {})
    coverage_entries = []
    for file_path, metrics in files.items():
        summary = metrics.get("summary", {})
        coverage_entries.append(
            {
                "file": file_path,
                "percent": summary.get("percent_covered", 0.0),
            }
        )

    lowest = sorted(coverage_entries, key=lambda x: x["percent"])[:3]
    return {"percent": percent, "lowest": lowest}


def load_benchmark(path: Path):
    """Load benchmark_results.json if it exists; return None otherwise."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _perf_summary_lines(bench: dict) -> list:
    """Return markdown lines summarising the key speedups from benchmark data."""
    results = bench.get("results", [])
    if not results:
        return ["- No benchmark data found."]

    # Reference label per benchmark name
    ref_labels = {
        "vip_scores": "Python loop",
        "pearson_columns": "Full-copy NumPy (baseline)",
        "column_variances": "NumPy (baseline)",
        "feature_preselection": "Raw NumPy var (baseline)",
        "stocsy": "Standard STOCSY",
        "opls_da_pipeline": "float64 pipeline",
    }

    # Build reference lookup
    refs: dict = {}
    for r in results:
        name, label = r["name"], r["label"]
        ref = ref_labels.get(name)
        if label == ref:
            key = (name, r["n_samples"], r["n_features"])
            refs[key] = r["min_ms"]

    lines = []
    for r in results:
        name = r["name"]
        ref = ref_labels.get(name)
        if not ref or r["label"] == ref:
            continue
        key = (name, r["n_samples"], r["n_features"])
        ref_ms = refs.get(key)
        if ref_ms and r["min_ms"] and r["min_ms"] == r["min_ms"]:
            ratio = ref_ms / max(r["min_ms"], 1e-9)
            ds = f"{r['n_samples']:,}x{r['n_features']:,}"
            lines.append(
                f"- `{name}` {r['label']} @ {ds}: "
                f"**{ratio:.1f}x** speedup ({r['min_ms']:.1f} ms vs {ref_ms:.1f} ms baseline)"
            )
    return lines or ["- No speedup data available."]


def main():
    root = Path(__file__).resolve().parent.parent
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    junit = load_junit_report(reports_dir / "junit.xml")
    coverage = load_coverage(reports_dir / "coverage.json")
    bench = load_benchmark(reports_dir / "benchmark_results.json")

    try:
        from metbit import __version__ as package_version
    except Exception:
        package_version = "unknown"

    timestamp = datetime.utcnow().isoformat() + "Z"
    python_version = platform.python_version()
    os_name = f"{platform.system()} {platform.release()}"

    passed = junit["total"] - (junit["failures"] + junit["errors"] + junit["skipped"])

    lines = [
        f"# Test Summary ({timestamp})",
        "",
        f"- Python: {python_version}",
        f"- OS: {os_name}",
        f"- Package version: {package_version}",
        f"- Total: {junit['total']}, Passed: {passed}, Failed: {junit['failures'] + junit['errors']}, Skipped: {junit['skipped']}",
        f"- Duration: {junit['time']:.2f}s",
        f"- Coverage: {coverage['percent']:.2f}%",
        "",
        "## Lowest coverage modules",
    ]

    if coverage["lowest"]:
        for entry in coverage["lowest"]:
            lines.append(f"- {entry['file']}: {entry['percent']:.2f}%")
    else:
        lines.append("- N/A")

    lines.append("")
    lines.append("## Performance")
    if bench:
        b_ts = bench.get("timestamp", "unknown")
        b_backend = bench.get("backend", {})
        native = "yes" if b_backend.get("native_c") else "no"
        omp    = b_backend.get("openmp_threads", 0)
        gpu    = "yes" if (b_backend.get("gpu_cupy") or b_backend.get("gpu_torch")) else "no"
        lines.append(f"- Benchmark run: {b_ts}")
        lines.append(f"- Backend: native_c={native}, openmp_threads={omp}, gpu={gpu}")
        lines += _perf_summary_lines(bench)
        lines.append(f"- Full report: reports/PERFORMANCE.md")
    else:
        lines.append("- No benchmark results found. Run: `python scripts/perf_report.py`")

    lines.append("")
    lines.append("## Failures")
    if junit["failed_tests"]:
        for fail in junit["failed_tests"]:
            lines.append(f"- {fail['name']} ({fail['file']}): {fail['message']}")
    else:
        lines.append("- None")

    summary_path = reports_dir / "test-summary.md"
    summary_path.write_text("\n".join(lines))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
