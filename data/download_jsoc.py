"""
download_jsoc.py
----------------
Downloads one SDO/HMI Ic_45s continuum FITS file per day from JSOC
for a specified date range, using the drms package directly.

No JSOC account or email required — this uses the public read-only
segment URL endpoint (http://jsoc.stanford.edu/<path>/continuum.fits).

Usage:
    python download_jsoc.py

Configuration:
    Edit START_DATE, END_DATE, and SAVE_DIR below.
    The script is safe to re-run — it skips dates already downloaded.
"""

import os
import time
import datetime
import requests
import drms

# ── Configuration ────────────────────────────────────────────────────────────
START_DATE = datetime.date(2023, 10, 19)   # First date to download
END_DATE   = datetime.date(2023, 12, 31)   # Last date to download (inclusive)
SAVE_DIR   = r"data\raw_fits"              # Relative to repo root
TARGET_UTC_HOUR = 12                       # Target ~12:00:00 TAI each day
QUALITY_MASK    = 0x00000000               # 0 = only perfect-quality images
RETRY_LIMIT     = 3                        # Retries per file on network error
RETRY_DELAY_S   = 10                       # Seconds between retries
REQUEST_TIMEOUT = 120                      # Seconds per HTTP request
INTER_FILE_DELAY = 2                       # Seconds between downloads (polite)
# ─────────────────────────────────────────────────────────────────────────────

JSOC_BASE = "https://jsoc1.stanford.edu"


def date_to_tai(d: datetime.date, hour: int = 12) -> str:
    """Convert a date to a JSOC TAI time string for the given UTC hour."""
    return f"{d.year}.{d.month:02d}.{d.day:02d}_{hour:02d}:00:00_TAI"


def already_downloaded(save_dir: str, date: datetime.date) -> bool:
    """Return True if a FITS file for this date already exists in save_dir."""
    prefix = f"hmi.ic_45s.{date.year}.{date.month:02d}.{date.day:02d}_"
    for f in os.listdir(save_dir):
        if f.startswith(prefix) and f.endswith(".fits"):
            return True
    return False


def fetch_file(url: str, dest_path: str, retries: int = RETRY_LIMIT) -> bool:
    """Download url to dest_path with retry logic. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True,
                             allow_redirects=True)
            if r.status_code == 200:
                tmp_path = dest_path + ".tmp"
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                os.rename(tmp_path, dest_path)
                return True
            else:
                print(f"    [attempt {attempt}] HTTP {r.status_code} — retrying...")
        except Exception as e:
            print(f"    [attempt {attempt}] Error: {e} — retrying...")
        time.sleep(RETRY_DELAY_S * attempt)
    return False


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    c = drms.Client()

    current = START_DATE
    total_days = (END_DATE - START_DATE).days + 1
    day_num = 0
    success_count = 0
    skip_count = 0
    fail_count = 0

    print(f"Downloading HMI Ic_45s: {START_DATE} → {END_DATE}  ({total_days} days)")
    print(f"Saving to: {os.path.abspath(SAVE_DIR)}")
    print("-" * 60)

    while current <= END_DATE:
        day_num += 1
        tai_str = date_to_tai(current)
        date_label = current.isoformat()

        # Skip if we already have this date
        if already_downloaded(SAVE_DIR, current):
            print(f"[{day_num:3d}/{total_days}] {date_label}  SKIP (already downloaded)")
            skip_count += 1
            current += datetime.timedelta(days=1)
            continue

        # Query JSOC for record metadata + segment path
        try:
            keys, segs = c.query(
                f"hmi.Ic_45s[{tai_str}]",
                key="T_REC,QUALITY",
                seg="continuum"
            )
        except Exception as e:
            print(f"[{day_num:3d}/{total_days}] {date_label}  QUERY ERROR: {e}")
            fail_count += 1
            current += datetime.timedelta(days=1)
            continue

        if len(keys) == 0:
            print(f"[{day_num:3d}/{total_days}] {date_label}  NO DATA")
            fail_count += 1
            current += datetime.timedelta(days=1)
            continue

        quality = int(keys["QUALITY"].iloc[0])
        t_rec   = keys["T_REC"].iloc[0]
        seg_path = segs["continuum"].iloc[0]

        if quality != QUALITY_MASK:
            print(f"[{day_num:3d}/{total_days}] {date_label}  QUALITY FLAG {quality:#010x} — skipping")
            fail_count += 1
            current += datetime.timedelta(days=1)
            continue

        # Build output filename matching existing convention
        # hmi.ic_45s.YYYY.MM.DD_HH_MM_SS_TAI.continuum.fits
        t_clean = t_rec.replace(":", "_").replace(".", ".", 2).replace(".", "_", 2)
        # t_rec format: "2023.10.19_12:00:00_TAI"
        # target:       "hmi.ic_45s.2023.10.19_12_00_00_TAI.continuum.fits"
        t_file = t_rec.replace(":", "_")         # "2023.10.19_12_00_00_TAI"
        filename = f"hmi.ic_45s.{t_file}.continuum.fits"
        dest_path = os.path.join(SAVE_DIR, filename)

        url = JSOC_BASE + seg_path
        size_mb_approx = 15  # typical Rice-compressed HMI Ic FITS

        print(f"[{day_num:3d}/{total_days}] {date_label}  {t_rec}  ~{size_mb_approx}MB ...", end="", flush=True)
        t0 = time.time()

        ok = fetch_file(url, dest_path)
        elapsed = time.time() - t0

        if ok:
            actual_mb = os.path.getsize(dest_path) / 1e6
            print(f"  OK  {actual_mb:.1f}MB  {elapsed:.1f}s")
            success_count += 1
        else:
            print(f"  FAILED after {RETRY_LIMIT} attempts")
            fail_count += 1

        current += datetime.timedelta(days=1)
        time.sleep(INTER_FILE_DELAY)

    print("-" * 60)
    print(f"Done.  Downloaded: {success_count}  Skipped: {skip_count}  Failed: {fail_count}")


if __name__ == "__main__":
    main()
