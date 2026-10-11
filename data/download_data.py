from sunpy.net import Fido, attrs
from astropy import units as u
from datetime import date, timedelta
import os

save_dir = "data/raw_fits"
os.makedirs(save_dir, exist_ok=True)

start = date(2023, 10, 1)
for i in range(90):
    day = start + timedelta(days=i)
    t_str = f"{day}T12:00:00"
    t_end = f"{day}T12:30:00"
    result = Fido.search(
        attrs.Time(t_str, t_end),
        attrs.Instrument("HMI"),
        attrs.Physobs("intensity")
    )
    if len(result) > 0:
        Fido.fetch(result[0:1], path=save_dir)
