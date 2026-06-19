#!/usr/bin/env python3
"""
Step 2 - Download and extract raw Bruker FID ZIPs from MetaboLights.

For MTBLS1 the layout inside each ZIP is:
    <sample_name>/<exp_no>/fid
    <sample_name>/<exp_no>/acqus
    ...

This script downloads every per-sample ZIP from the FTP FILES/ directory,
extracts it, and re-arranges it so nmr_preprocessing can find the fid files:

    data/<ACCESSION>/FILES/<sample_name>/fid
    data/<ACCESSION>/FILES/<sample_name>/acqus
    ...

It also downloads the published processed spectral matrix (if present) which
is used in Step 3 for side-by-side comparison.

Usage:
    python 02_download_fids.py [ACCESSION] [--max N]

Options:
    --max N    Download at most N samples (default: all). Use --max 10 for pilot.
"""

import sys
import ftplib
import io
import time
import zipfile
import shutil
from pathlib import Path

ACCESSION   = "MTBLS1"
MAX_SAMPLES = None

for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--max" and i < len(sys.argv) - 1:
        MAX_SAMPLES = int(sys.argv[i + 1])
    elif not arg.startswith("--") and arg.upper().startswith("MTBLS"):
        ACCESSION = arg

FTP_HOST   = "ftp.ebi.ac.uk"
FTP_BASE   = f"/pub/databases/metabolights/studies/public/{ACCESSION}"
DATA_DIR   = Path(__file__).parent / "data" / ACCESSION
FID_ROOT   = DATA_DIR / "FILES"
FID_ROOT.mkdir(parents=True, exist_ok=True)

# Published processed matrix filename (MTBLS1 specific; may differ per study)
PUBLISHED_MATRIX = "ADG_transformed_data.xlsx"


def connect_ftp():
    ftp = ftplib.FTP(FTP_HOST, timeout=60)
    ftp.login()
    return ftp


def list_zip_files(ftp):
    items = []
    ftp.retrlines(f"LIST {FTP_BASE}/FILES", items.append)
    zips = [line.split()[-1] for line in items if line.split()[-1].endswith(".zip")]
    return zips


def download_and_extract(ftp, remote_filename, target_dir):
    """
    Download one ZIP from FTP into memory and extract it into target_dir,
    flattening the <exp_no>/ layer:
        <sample>/10/fid  ->  target_dir/fid
        <sample>/10/acqus -> target_dir/acqus
    """
    buf = io.BytesIO()
    ftp.retrbinary(f"RETR {FTP_BASE}/FILES/{remote_filename}", buf.write)
    buf.seek(0)

    with zipfile.ZipFile(buf) as zf:
        for member in zf.namelist():
            parts = Path(member).parts
            # parts[0] = sample_name, parts[1] = exp_no, parts[2:] = file path
            if len(parts) < 3:
                continue
            rel = Path(*parts[2:])  # strip sample_name/exp_no/
            if str(rel) == ".":
                continue
            dest = target_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(member)
            if data:   # skip directory entries
                dest.write_bytes(data)

    return True


# ---------------------------------------------------------------------------
# 1. list available ZIPs
# ---------------------------------------------------------------------------

print(f"\n=== Downloading FID data for {ACCESSION} ===")
print(f"  FTP: {FTP_HOST}{FTP_BASE}/FILES/")
print(f"  Local: {FID_ROOT}")

try:
    ftp = connect_ftp()
    zip_files = list_zip_files(ftp)
    print(f"  Found {len(zip_files)} ZIP files on FTP")
except Exception as exc:
    print(f"[error] FTP failed: {exc}")
    sys.exit(1)

if MAX_SAMPLES:
    zip_files = zip_files[:MAX_SAMPLES]
    print(f"  Limited to first {MAX_SAMPLES} samples (--max flag)")


# ---------------------------------------------------------------------------
# 2. download published processed matrix (for benchmark comparison)
# ---------------------------------------------------------------------------

pub_matrix_local = DATA_DIR / PUBLISHED_MATRIX
if not pub_matrix_local.exists():
    print(f"\n  Downloading published matrix: {PUBLISHED_MATRIX} ...")
    buf = io.BytesIO()
    try:
        ftp.retrbinary(f"RETR {FTP_BASE}/FILES/{PUBLISHED_MATRIX}", buf.write)
        pub_matrix_local.write_bytes(buf.getvalue())
        print(f"  Saved: {pub_matrix_local.name}  ({pub_matrix_local.stat().st_size // 1024} KB)")
    except Exception as exc:
        print(f"  [warn] Could not download {PUBLISHED_MATRIX}: {exc}")
else:
    print(f"\n  Published matrix already present: {PUBLISHED_MATRIX}")


# ---------------------------------------------------------------------------
# 3. download + extract sample ZIPs
# ---------------------------------------------------------------------------

print(f"\n  Downloading and extracting {len(zip_files)} samples...\n")

downloaded = 0
skipped    = 0
failed     = 0

for idx, zname in enumerate(zip_files, 1):
    sample_name = zname.replace(".zip", "")
    target_dir  = FID_ROOT / sample_name

    if (target_dir / "fid").exists():
        print(f"  [{idx:03d}/{len(zip_files)}] {sample_name}  [skip - already exists]")
        skipped += 1
        continue

    print(f"  [{idx:03d}/{len(zip_files)}] {sample_name} ... ", end="", flush=True)

    for attempt in range(3):
        try:
            download_and_extract(ftp, zname, target_dir)
            size_kb = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file()) // 1024
            print(f"OK  ({size_kb} KB)")
            downloaded += 1
            break
        except (ftplib.error_temp, EOFError, ConnectionResetError, OSError) as exc:
            print(f"\n  [warn] {exc} - reconnecting...", end="")
            time.sleep(3)
            try:
                ftp.quit()
            except Exception:
                pass
            ftp = connect_ftp()
    else:
        print("FAILED")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        failed += 1

try:
    ftp.quit()
except Exception:
    pass

# ---------------------------------------------------------------------------
# 4. verify fid file count
# ---------------------------------------------------------------------------

fid_count = len(list(FID_ROOT.glob("*/fid")))
print(f"\nSummary:")
print(f"  Downloaded : {downloaded}")
print(f"  Skipped    : {skipped}  (already present)")
print(f"  Failed     : {failed}")
print(f"  FID files found in {FID_ROOT}: {fid_count}")

if fid_count == 0:
    print("\n[error] No fid files found. Check the extraction above.")
    sys.exit(1)

print(f"\nNext step: python 03_run_pipeline.py {ACCESSION}")
