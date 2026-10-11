"""
select_daily.py
---------------
From the raw_fits directory (which may contain many 45-second cadence
images per day), selects the single FITS file per day closest to
12:00:00 TAI and copies it into data/daily_fits/.

This gives us a clean one-file-per-day dataset for the pipeline,
regardless of how many images were downloaded per day.

Usage:
    python select_daily.py

Run this after any batch download to keep daily_fits/ up to date.
"""

import os
import re
import shutil
import datetime

RAW_DIR   = r"data\raw_fits"
DAILY_DIR = r"data\daily_fits"

# Pattern: hmi.ic_45s.YYYY.MM.DD_HH_MM_SS_TAI.continuum.fits
FNAME_RE = re.compile(
    r"hmi\.ic_45s\.(\d{4})\.(\d{2})\.(\d{2})_(\d{2})_(\d{2})_(\d{2})_TAI\.continuum\.fits"
)

TARGET_SECONDS = 12 * 3600  # 12:00:00 in seconds from midnight


def parse_filename(fname):
    """Return (date, seconds_from_midnight) or None if pattern doesn't match."""
    m = FNAME_RE.match(fname)
    if not m:
        return None
    yyyy, mm, dd, hh, mi, ss = (int(x) for x in m.groups())
    d = datetime.date(yyyy, mm, dd)
    secs = hh * 3600 + mi * 60 + ss
    return d, secs


def main():
    os.makedirs(DAILY_DIR, exist_ok=True)

    # Group all FITS files by date
    by_date = {}
    for fname in os.listdir(RAW_DIR):
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        date, secs = parsed
        by_date.setdefault(date, []).append((secs, fname))

    copied = 0
    skipped = 0
    dates = sorted(by_date.keys())

    print(f"Found {len(dates)} unique dates in {RAW_DIR}")
    print(f"Selecting one image per day (closest to 12:00:00 TAI)")
    print("-" * 60)

    for date in dates:
        candidates = by_date[date]
        # Pick the file with time closest to 12:00:00
        candidates.sort(key=lambda x: abs(x[0] - TARGET_SECONDS))
        best_secs, best_fname = candidates[0]

        hh = best_secs // 3600
        mi = (best_secs % 3600) // 60
        ss = best_secs % 60
        time_str = f"{hh:02d}:{mi:02d}:{ss:02d}"

        dest_path = os.path.join(DAILY_DIR, best_fname)
        if os.path.exists(dest_path):
            print(f"  {date}  {time_str} TAI  ({len(candidates):2d} candidates)  ALREADY EXISTS")
            skipped += 1
            continue

        src_path = os.path.join(RAW_DIR, best_fname)
        shutil.copy2(src_path, dest_path)
        print(f"  {date}  {time_str} TAI  ({len(candidates):2d} candidates)  -> {best_fname}")
        copied += 1

    print("-" * 60)
    print(f"Done.  Copied: {copied}  Already present: {skipped}")
    print(f"Daily FITS directory: {os.path.abspath(DAILY_DIR)}")
    print(f"Total daily files now available: {len(os.listdir(DAILY_DIR))}")


if __name__ == "__main__":
    main()
