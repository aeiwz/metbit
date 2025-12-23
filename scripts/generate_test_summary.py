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


def main():
    root = Path(__file__).resolve().parent.parent
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    junit = load_junit_report(reports_dir / "junit.xml")
    coverage = load_coverage(reports_dir / "coverage.json")

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
