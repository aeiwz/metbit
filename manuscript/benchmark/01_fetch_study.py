#!/usr/bin/env python3
"""
Step 1 - Fetch a MetaboLights NMR study.

Downloads ISA-Tab metadata and raw Bruker FID files for a given MTBLS accession.
Default target: MTBLS1 (human urine 1H NMR, binary group labels).

Usage:
    python 01_fetch_study.py [ACCESSION]      # e.g. MTBLS1
    python 01_fetch_study.py MTBLS12785

Output layout:
    data/<ACCESSION>/
        s_*.txt            ISA-Tab sample file  (group labels live here)
        a_*.txt            ISA-Tab assay file   (links samples -> data files)
        FILES/             raw Bruker FID dirs  (one sub-dir per sample)
"""

import sys
import json
import os
import ftplib
import time
import requests
from pathlib import Path

ACCESSION = sys.argv[1] if len(sys.argv) > 1 else "MTBLS1"
BASE_API  = "https://www.ebi.ac.uk/metabolights/ws"
FTP_HOST  = "ftp.ebi.ac.uk"
FTP_BASE  = f"/pub/databases/metabolights/studies/public/{ACCESSION}"
OUT_DIR   = Path(__file__).parent / "data" / ACCESSION


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def api_get(url, retries=3, pause=2):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            print(f"  [warn] attempt {attempt+1} failed: {exc}")
            time.sleep(pause)
    raise RuntimeError(f"API call failed after {retries} retries: {url}")


def ftp_download_file(ftp, remote_path, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as fh:
        ftp.retrbinary(f"RETR {remote_path}", fh.write)


def ftp_list_dir(ftp, remote_dir):
    items = []
    ftp.retrlines(f"LIST {remote_dir}", items.append)
    return items


# ---------------------------------------------------------------------------
# 1. study metadata
# ---------------------------------------------------------------------------

print(f"\n=== Fetching {ACCESSION} metadata ===")
meta = api_get(f"{BASE_API}/studies/{ACCESSION}")
title = meta.get("title", "n/a")
organism = meta.get("organism", [{}])
print(f"  Title   : {title}")
print(f"  Organism: {organism}")

# Save metadata summary
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "metadata.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"  Saved   : {OUT_DIR}/metadata.json")


# ---------------------------------------------------------------------------
# 2. file listing
# ---------------------------------------------------------------------------

print(f"\n=== Listing files for {ACCESSION} ===")
files_resp = api_get(f"{BASE_API}/studies/{ACCESSION}/files")

# The API returns a nested structure - flatten to a list of relative paths
def flatten_files(node, prefix=""):
    results = []
    if isinstance(node, dict):
        name = node.get("file", node.get("name", ""))
        path = f"{prefix}/{name}".lstrip("/")
        if node.get("type") == "FILE" or "directory" not in node.get("type", "").lower():
            results.append(path)
        for child in node.get("files", []):
            results.extend(flatten_files(child, path if node.get("type","").lower() == "directory" else prefix))
    elif isinstance(node, list):
        for item in node:
            results.extend(flatten_files(item, prefix))
    return results

all_files = flatten_files(files_resp)

# Separate ISA-Tab from raw data
isa_files  = [f for f in all_files if f.startswith("s_") or f.startswith("a_") or f == "i_Investigation.txt"]
data_files = [f for f in all_files if not f.startswith(("s_", "a_", "i_"))]

print(f"  ISA-Tab files : {len(isa_files)}")
print(f"  Other files   : {len(data_files)}")

with open(OUT_DIR / "file_list.json", "w") as fh:
    json.dump({"isa": isa_files, "data": data_files}, fh, indent=2)


# ---------------------------------------------------------------------------
# 3. download ISA-Tab via FTP (small files - always grab these)
# ---------------------------------------------------------------------------

print(f"\n=== Downloading ISA-Tab files via FTP ===")
try:
    ftp = ftplib.FTP(FTP_HOST, timeout=30)
    ftp.login()

    for rel in ["i_Investigation.txt"] + [f for f in all_files if f.startswith(("s_", "a_"))]:
        remote = f"{FTP_BASE}/{rel}"
        local  = OUT_DIR / rel
        try:
            ftp_download_file(ftp, remote, local)
            print(f"  OK  {rel}")
        except Exception as exc:
            print(f"  ERR {rel}: {exc}")

    ftp.quit()
except Exception as exc:
    print(f"  [warn] FTP connection failed: {exc}")
    print("  Falling back to HTTP download for ISA-Tab files...")
    for rel in [f for f in all_files if f.startswith(("s_", "a_", "i_"))]:
        url   = f"https://www.ebi.ac.uk/metabolights/{ACCESSION}/files/{rel}"
        local = OUT_DIR / rel
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(r.content)
            print(f"  OK  {rel}")
        except Exception as e2:
            print(f"  ERR {rel}: {e2}")


# ---------------------------------------------------------------------------
# 4. identify NMR raw data directories
# ---------------------------------------------------------------------------

print(f"\n=== Identifying NMR raw data directories ===")
import pandas as pd

sample_files = list(OUT_DIR.glob("s_*.txt"))
if not sample_files:
    print("  [warn] No s_*.txt found - cannot parse group labels yet.")
    print("  Run this script again after ISA-Tab files are downloaded.")
    sys.exit(0)

s_file = sample_files[0]
print(f"  Parsing: {s_file.name}")
try:
    s_df = pd.read_csv(s_file, sep="\t")
    print(f"  Columns : {list(s_df.columns)}")
    factor_cols = [c for c in s_df.columns if "Factor Value" in c]
    print(f"  Factors : {factor_cols}")
    if factor_cols:
        for fc in factor_cols:
            counts = s_df[fc].value_counts()
            print(f"    {fc}: {counts.to_dict()}")
    s_df.to_csv(OUT_DIR / "samples_parsed.csv", index=False)
    print(f"  Saved   : samples_parsed.csv")
except Exception as exc:
    print(f"  [warn] Could not parse sample file: {exc}")


# ---------------------------------------------------------------------------
# 5. list NMR FID directories on FTP (do NOT download yet - may be large)
# ---------------------------------------------------------------------------

print(f"\n=== Scanning FTP for NMR FID directories ===")
try:
    ftp = ftplib.FTP(FTP_HOST, timeout=30)
    ftp.login()

    # Try common subdirectory patterns used by MetaboLights
    fid_dirs = []
    for candidate_dir in [FTP_BASE, f"{FTP_BASE}/FILES", f"{FTP_BASE}/RAW_FILES"]:
        try:
            listing = []
            ftp.retrlines(f"LIST {candidate_dir}", listing.append)
            fid_dirs = [line.split()[-1] for line in listing if line.startswith("d")]
            if fid_dirs:
                print(f"  Found {len(fid_dirs)} subdirs in {candidate_dir}")
                with open(OUT_DIR / "fid_dirs.txt", "w") as fh:
                    fh.write(f"# Remote FTP base: {candidate_dir}\n")
                    for d in fid_dirs:
                        fh.write(f"{candidate_dir}/{d}\n")
                print(f"  Saved : fid_dirs.txt  ({len(fid_dirs)} entries)")
                break
        except Exception:
            pass

    if not fid_dirs:
        print("  [warn] Could not enumerate FID directories.")

    ftp.quit()
except Exception as exc:
    print(f"  [warn] FTP scan failed: {exc}")

print(f"\nDone. Next step: python 02_download_fids.py {ACCESSION}")
