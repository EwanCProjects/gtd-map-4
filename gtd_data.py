"""Loading, filtering and packing the Global Terrorism Database for the app.

Nothing here depends on Streamlit, so it can be tested and timed on its own.
"""

from __future__ import annotations

import ctypes
import gc
import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Columns ──────────────────────────────────────────────────────────────────
# Only REQUIRED must be in the file. Every other column switches features on when present,
# so a trimmed parquet (fewer columns) still works, with fewer filters.
REQUIRED = ["iyear", "imonth", "iday", "latitude", "longitude", "country_txt"]
CATEGORY_COLS = [  # repeated labels: stored as categoricals, filtered by integer code
    "country_txt", "region_txt", "provstate", "city", "attacktype1_txt", "targtype1_txt",
    "targsubtype1_txt", "natlty1_txt", "weaptype1_txt", "weapsubtype1_txt", "gname",
    "propextent_txt",
]
TEXT_COLS = ["summary", "target1", "corp1", "location", "motive", "weapdetail", "propcomment"]
NUMBER_COLS = ["nkill", "nwound", "nhostkid", "nperps", "propvalue"]  # the GTD uses -99 / -9 for unknown
FLAG_COLS = ["success", "suicide", "doubtterr", "multiple", "extended", "claimed", "property"]
ID_COL = "eventid"
ALL_COLS = list(dict.fromkeys(
    REQUIRED + [ID_COL] + CATEGORY_COLS + TEXT_COLS + NUMBER_COLS + FLAG_COLS))

# Columns searched by the free-text box (categoricals are searched through their labels only)
SEARCH_TEXT = ["summary", "target1", "corp1", "location", "motive"]
SEARCH_CATEGORIES = ["city", "provstate", "gname", "country_txt"]

# Filterable categories: sidebar key -> column
FACETS = {
    "regions": "region_txt",
    "countries": "country_txt",
    "attack": "attacktype1_txt",
    "target": "targtype1_txt",
    "weapon": "weaptype1_txt",
    "groups": "gname",
}

# Display names for a few very long GTD labels
SHORT_LABELS = {
    "Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)": "Vehicle (not vehicle bombs)",
    "Explosives/Bombs/Dynamite": "Explosives",
}
SUMMARY_DATE_PREFIX = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}:\s*")  # "03/12/2014: " repeats the date

# ── Map colour groups ────────────────────────────────────────────────────────
# A dot map can only keep about three colours apart (every dot can sit next to every other),
# so each "colour by" option shows three fixed groups plus "Other". The groups are fixed by name,
# not by rank, so a colour always means the same thing whatever the filters.
COLOR_GROUPS = {
    "attack": ("attacktype1_txt", "Attack type", [
        ("Bombing/explosion", lambda s: s.startswith("Bombing")),
        ("Armed assault", lambda s: s == "Armed Assault"),
        ("Assassination", lambda s: s == "Assassination"),
    ], "Other attack types"),
    "target": ("targtype1_txt", "Target", [
        ("Civilians & property", lambda s: s.startswith("Private Citizens")),
        ("Military & police", lambda s: s in ("Military", "Police")),
        ("Government", lambda s: s.startswith("Government")),
    ], "Other targets"),
    "weapon": ("weaptype1_txt", "Weapon", [
        ("Explosives", lambda s: s.startswith("Explosives")),
        ("Firearms", lambda s: s == "Firearms"),
        ("Incendiary", lambda s: s == "Incendiary"),
    ], "Other or unknown"),
}


@dataclass
class Facet:
    """A categorical column as integer codes (-1 = missing) plus its labels."""
    codes: np.ndarray
    labels: list[str]

    def lookup(self, chosen) -> np.ndarray:
        """Boolean array over all events: is the event's label one of `chosen`?"""
        table = np.zeros(len(self.labels) + 1, dtype=bool)  # last slot catches code -1
        index = {label: i for i, label in enumerate(self.labels)}
        for label in chosen:
            if label in index:
                table[index[label]] = True
        return table[self.codes]


@dataclass
class Events:
    df: pd.DataFrame                    # display columns; row i is event i everywhere
    lat: np.ndarray                     # float64
    lon: np.ndarray
    year: np.ndarray                    # int16
    datekey: np.ndarray                 # int32 yyyymmdd, 0 for unknown month/day
    killed: np.ndarray                  # float32, NaN = unknown
    wounded: np.ndarray
    facets: dict[str, Facet]            # column -> Facet
    flags: dict[str, np.ndarray]        # column -> int8 (1 yes, 0 no, -9 unknown)
    dropped: int                        # rows without usable coordinates or dates
    columns: set[str] = field(default_factory=set)
    data_id: str = ""

    @property
    def n(self) -> int:
        return len(self.lat)

    def has(self, col: str) -> bool:
        return col in self.columns


# ── Loading ──────────────────────────────────────────────────────────────────
def _to_text(s: pd.Series) -> pd.Series:
    """Arrow-backed strings: compact, and fast for substring search."""
    if s.isna().all():  # e.g. an all-empty column stored with the parquet "null" type
        return pd.Series(pd.array([None] * len(s), dtype="string[pyarrow]"), index=s.index)
    return s.astype("string[pyarrow]")


def load_events(path: str) -> Events:
    schema = pq.read_schema(path)
    missing = [c for c in REQUIRED if c not in schema.names]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
    cols = [c for c in ALL_COLS if c in schema.names]
    table = pq.read_table(path, columns=cols)
    release_memory()  # the file's decompression buffers

    # Which rows are usable, and their order, from a few small columns (so the rest is copied only once)
    key = table.select([c for c in ("iyear", "imonth", "iday", "latitude", "longitude", "country_txt", "nkill")
                        if c in cols]).to_pandas()
    for c in key.columns.drop("country_txt"):
        key[c] = pd.to_numeric(key[c], errors="coerce")
    month = key["imonth"].fillna(0).clip(0, 12)  # the GTD uses 0 for an unknown month or day
    day = key["iday"].fillna(0).clip(0, 31)
    # Unusable rows: no coordinates, impossible coordinates (a few GTD rows), or no date/country
    ok = key[["latitude", "longitude", "iyear", "country_txt"]].notna().all(axis=1)
    ok &= key["latitude"].between(-90, 90) & key["longitude"].between(-180, 180)
    dropped = int((~ok).sum())
    keep = np.flatnonzero(ok.to_numpy())
    # Draw order on the map = row order: deadliest first, so they sit underneath the small dots.
    order = keep[np.lexsort((
        (key["iyear"] * 10_000 + month * 100 + day).to_numpy()[keep],
        -key["nkill"].fillna(-1).to_numpy()[keep] if "nkill" in key else np.zeros(len(keep)),
    ))]
    del key, month, day, ok
    table = table.take(pa.array(order))
    release_memory()
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    del table

    for c in ["iyear", "imonth", "iday", "latitude", "longitude"] + [c for c in NUMBER_COLS + FLAG_COLS if c in df]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["imonth"] = df["imonth"].fillna(0).clip(0, 12)
    df["iday"] = df["iday"].fillna(0).clip(0, 31)

    # Each column is converted, then dropped from df, so only one copy of the data exists at a time
    out = pd.DataFrame(index=df.index)
    out["iyear"] = df.pop("iyear").astype("int16")
    out["imonth"] = df.pop("imonth").astype("int8")
    out["iday"] = df.pop("iday").astype("int8")
    out["latitude"] = df.pop("latitude").astype("float64")
    out["longitude"] = df.pop("longitude").astype("float64")
    if ID_COL in df:
        raw = df.pop(ID_COL)
        ids = pd.to_numeric(raw, errors="coerce")
        numeric = ids.notna().sum() == raw.notna().sum()
        out[ID_COL] = ids.astype("Int64") if numeric else raw.astype("string[pyarrow]")
        del raw, ids

    facets = {}
    for c in CATEGORY_COLS:
        if c in df:
            cat = df.pop(c).astype("string[pyarrow]").str.strip().replace("", pd.NA).astype("category")
            out[c] = cat
            facets[c] = Facet(cat.cat.codes.to_numpy().astype(np.int32), [str(x) for x in cat.cat.categories])
    for c in TEXT_COLS:
        if c in df:
            out[c] = _to_text(df.pop(c))
    if "summary" in out:
        out["summary"] = out["summary"].str.replace(SUMMARY_DATE_PREFIX, "", regex=True)
    for c in NUMBER_COLS:
        if c in df:
            out[c] = df.pop(c).astype("float32")
    flags = {}
    for c in FLAG_COLS:
        if c in df:
            flags[c] = df.pop(c).fillna(-9).astype("int8").to_numpy().copy()
            out[c] = flags[c]
    del df

    year = out["iyear"].to_numpy().copy()
    datekey = (year.astype(np.int32) * 10_000 + out["imonth"].to_numpy().astype(np.int32) * 100
               + out["iday"].to_numpy().astype(np.int32))
    nan = np.full(len(out), np.nan, dtype=np.float32)
    ev = Events(
        df=out,
        lat=out["latitude"].to_numpy().copy(),
        lon=out["longitude"].to_numpy().copy(),
        year=year,
        datekey=datekey,
        killed=out["nkill"].to_numpy().copy() if "nkill" in out else nan,
        wounded=out["nwound"].to_numpy().copy() if "nwound" in out else nan.copy(),
        facets=facets,
        flags=flags,
        dropped=dropped,
        columns=set(out.columns),
    )
    release_memory()
    return ev


def release_memory():
    """Hand memory freed while loading back to the system (Arrow's and C's allocators keep it otherwise)."""
    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except Exception:  # an older pyarrow
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux with glibc, e.g. Streamlit Community Cloud
    except (OSError, AttributeError):  # macOS, Windows
        pass


# ── Filtering ────────────────────────────────────────────────────────────────
def haversine_km(lat, lon, lats, lons):
    """Great-circle distance in km from one point to many."""
    lat, lon, lats, lons = map(np.radians, (lat, lon, lats, lons))
    a = np.sin((lats - lat) / 2) ** 2 + np.cos(lat) * np.cos(lats) * np.sin((lons - lon) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def search_terms(text: str) -> list[str]:
    """'embassy, consulate' -> ['embassy', 'consulate'] (commas separate alternatives)."""
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def text_mask(ev: Events, text: str) -> np.ndarray:
    """Events whose summary, place, target or group mentions any of the comma-separated terms."""
    keep = np.zeros(ev.n, dtype=bool)
    for term in search_terms(text):
        for c in SEARCH_TEXT:
            if c in ev.columns:
                hits = ev.df[c].str.contains(term, case=False, regex=False)
                keep = keep | hits.fillna(False).to_numpy(dtype=bool)
        for c in SEARCH_CATEGORIES:
            if c in ev.facets:
                labels = pd.Series(ev.facets[c].labels, dtype="string[pyarrow]")
                matched = labels[labels.str.contains(term, case=False, regex=False).fillna(False)]
                if len(matched):
                    keep = keep | ev.facets[c].lookup(matched.tolist())
    return keep


def filter_mask(ev: Events, f: dict, text_hits: np.ndarray | None = None) -> np.ndarray:
    """Boolean array: which events pass the filters. `f` is the app's filter dict."""
    keep = (ev.year >= f["years"][0]) & (ev.year <= f["years"][1])
    if f["where"] == "countries":
        # Chosen countries win; otherwise the regions filter (regions also narrow the country list)
        key = "countries" if f.get("countries") else "regions"
        if f.get(key) and FACETS[key] in ev.facets:
            keep = keep & ev.facets[FACETS[key]].lookup(f[key])
    elif f["where"] == "around":
        keep = keep & (haversine_km(f["lat"], f["lon"], ev.lat, ev.lon) <= f["radius"])
    elif f["where"] == "area" and f.get("area"):
        keep = keep & in_bounds(ev.lat, ev.lon, f["area"])
    for key in ("attack", "target", "weapon", "groups"):
        if f.get(key) and FACETS[key] in ev.facets:
            keep = keep & ev.facets[FACETS[key]].lookup(f[key])
    if f.get("min_killed"):
        keep = keep & (ev.killed >= f["min_killed"])  # unknown death tolls (NaN) drop out
    if f.get("suicide") and "suicide" in ev.flags:
        keep = keep & (ev.flags["suicide"] == 1)
    if f.get("success") and "success" in ev.flags:
        keep = keep & (ev.flags["success"] == 1)
    if f.get("no_doubt") and "doubtterr" in ev.flags:
        keep = keep & (ev.flags["doubtterr"] != 1)
    if text_hits is not None:
        keep = keep & text_hits
    return keep


def in_bounds(lat: np.ndarray, lon: np.ndarray, bounds) -> np.ndarray:
    """Which points are inside (west, south, east, north), as the map reports its view.

    The map's west/east can go past ±180° (it repeats the world), so this handles the date line.
    """
    west, south, east, north = bounds
    keep = (lat >= south) & (lat <= north)
    if east - west < 360:  # otherwise every longitude is inside
        west, east = (west + 180) % 360 - 180, (east + 180) % 360 - 180
        keep = keep & (((lon >= west) & (lon <= east)) if west <= east else ((lon >= west) | (lon <= east)))
    return keep


def data_bounds(ev: Events, idx: np.ndarray):
    """[west, south, east, north] around the bulk of the events, ignoring stray outliers."""
    if len(idx) == 0:
        return None
    lo, hi = (1, 99) if len(idx) > 100 else (0, 100)
    lat, lon = ev.lat[idx], ev.lon[idx]
    shifted = lon % 360  # 0–360°, for places that straddle the date line (e.g. Fiji)
    lon_lo, lon_hi = np.percentile(lon, [lo, hi])
    s_lo, s_hi = np.percentile(shifted, [lo, hi])
    if s_hi - s_lo < lon_hi - lon_lo:
        lon_lo, lon_hi = s_lo, s_hi
    lat_lo, lat_hi = np.percentile(lat, [lo, hi])
    return [float(lon_lo), float(lat_lo), float(lon_hi), float(lat_hi)]


# ── Display helpers ──────────────────────────────────────────────────────────
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def format_dates(df: pd.DataFrame) -> pd.Series:
    """ISO dates. An unknown day or month (0 in the GTD) gives '2014-03' or '2014'."""
    y = df["iyear"].astype(str)
    m = df["imonth"].astype(str).str.zfill(2)
    d = df["iday"].astype(str).str.zfill(2)
    return (y + "-" + m + "-" + d).where(df["iday"] > 0, y + "-" + m).where(df["imonth"] > 0, y)


def long_date(year: int, month: int, day: int) -> str:
    if month <= 0:
        return str(year)
    if day <= 0:
        return f"{MONTHS[month - 1]} {year}"
    return f"{day} {MONTHS[month - 1]} {year}"


def _clean(v):
    """A cell as a plain Python value, or None when missing."""
    if v is None or (np.ndim(v) == 0 and pd.isna(v)):
        return None
    return v.item() if isinstance(v, np.generic) else v


def event_details(ev: Events, i: int) -> dict:
    """Everything the map's event card shows for one event (JSON-friendly)."""
    row = ev.df.iloc[i]
    get = lambda c: _clean(row[c]) if c in ev.columns else None  # noqa: E731
    def flag(c):
        v = get(c)
        return None if v is None or v < 0 else bool(v)
    def num(c):  # a count or amount; None when unknown (missing, or the GTD's -99 / -9)
        v = get(c)
        return None if v is None or v < 0 else int(v)
    place = ", ".join(str(p) for p in (get("city"), get("provstate"), get("country_txt"))
                      if p and str(p).lower() != "unknown")
    eid = get(ID_COL)
    return {
        "index": int(i),
        "id": str(eid) if eid is not None else None,
        "date": long_date(int(row["iyear"]), int(row["imonth"]), int(row["iday"])),
        "place": place or str(get("country_txt")),
        "region": get("region_txt"),
        "lat": float(row["latitude"]), "lon": float(row["longitude"]),
        "attack": get("attacktype1_txt"),
        "target_type": get("targtype1_txt"),
        "target": get("target1"),
        "entity": get("corp1"),
        "weapon": get("weapsubtype1_txt") or get("weaptype1_txt"),
        "weapon_detail": get("weapdetail"),
        "group": get("gname"),
        "perpetrators": num("nperps") or None,
        "killed": num("nkill"), "wounded": num("nwound"), "hostages": num("nhostkid") or None,
        "success": flag("success"), "suicide": flag("suicide"), "doubt": flag("doubtterr"),
        "claimed": flag("claimed"), "extended": flag("extended"), "multiple": flag("multiple"),
        # Property damage, when the file has it: any (yes/no/unknown), extent, value in US$, description
        "damage": {
            "any": flag("property"), "extent": get("propextent_txt"),
            "value": num("propvalue") or None, "comment": get("propcomment"),
        } if ev.has("property") or ev.has("propextent_txt") else None,
        "summary": get("summary"), "motive": get("motive"), "location": get("location"),
    }


# ── Packing the events for the browser ───────────────────────────────────────
def color_group_maps(ev: Events) -> dict:
    """For each 'colour by' option: labels, and the group index of every category code."""
    out = {}
    for key, (col, title, groups, other) in COLOR_GROUPS.items():
        if col not in ev.facets:
            continue
        labels = ev.facets[col].labels
        by_code = []
        for label in labels:
            g = next((gi for gi, (_, test) in enumerate(groups) if test(label)), len(groups))
            by_code.append(g)
        out[key] = {"title": title, "labels": [g[0] for g in groups] + [other],
                    "byCode": [len(groups)] + by_code}  # code 0 = missing -> "Other"
    return out


def _codes(ev: Events, col: str, dtype) -> np.ndarray:
    if col not in ev.facets:
        return np.zeros(ev.n, dtype=dtype)
    return (ev.facets[col].codes + 1).astype(dtype)  # 0 = missing


def _counts(a: np.ndarray) -> np.ndarray:
    """Casualty counts as uint16, 65535 = unknown."""
    return np.where(np.isnan(a), 65535, np.clip(a, 0, 65534)).astype(np.uint16)


def pack_for_browser(ev: Events) -> tuple[bytes, str]:
    """All events as one gzipped binary file the map loads once (about 2–3 MB for the full GTD).

    Layout: b"GTD1", uint32 header length, JSON header, then one little-endian array per column
    (each starting on a 4-byte boundary). The header lists the columns and the label lists.
    """
    def labels(col):
        return ev.facets[col].labels if col in ev.facets else []
    city_dtype = np.uint16 if len(labels("city")) < 65535 else np.uint32
    group_dtype = np.uint16 if len(labels("gname")) < 65535 else np.uint32
    arrays = [
        ("lon", ev.lon.astype("<f4")),
        ("lat", ev.lat.astype("<f4")),
        ("year", ev.year.astype("<u2")),
        ("month", ev.df["imonth"].to_numpy().astype(np.uint8)),
        ("day", ev.df["iday"].to_numpy().astype(np.uint8)),
        ("attack", _codes(ev, "attacktype1_txt", np.uint8)),
        ("target", _codes(ev, "targtype1_txt", np.uint8)),
        ("weapon", _codes(ev, "weaptype1_txt", np.uint8)),
        ("country", _codes(ev, "country_txt", "<u2")),
        ("city", _codes(ev, "city", city_dtype).astype(np.dtype(city_dtype).newbyteorder("<"))),
        ("group", _codes(ev, "gname", group_dtype).astype(np.dtype(group_dtype).newbyteorder("<"))),
        ("killed", _counts(ev.killed).astype("<u2")),
        ("wounded", _counts(ev.wounded).astype("<u2")),
    ]
    header = {
        "n": ev.n,
        "casualties": ev.has("nkill") or ev.has("nwound"),
        "dicts": {
            "attack": labels("attacktype1_txt"), "target": labels("targtype1_txt"),
            "weapon": labels("weaptype1_txt"), "country": labels("country_txt"),
            "city": labels("city"), "group": labels("gname"),
        },
        "columns": [],
    }
    # Offsets are relative to the start of the data section
    offset, chunks = 0, []
    for name, arr in arrays:
        pad = (-offset) % 4
        chunks.append(b"\0" * pad)
        offset += pad
        header["columns"].append([name, arr.dtype.kind + str(arr.dtype.itemsize), offset])
        raw = arr.tobytes()
        chunks.append(raw)
        offset += len(raw)
    head = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    head += b" " * ((-(8 + len(head))) % 4)  # data section starts 4-byte aligned
    raw = b"GTD1" + len(head).to_bytes(4, "little") + head + b"".join(chunks)
    data_id = hashlib.sha1(raw).hexdigest()[:12]
    return gzip.compress(raw, compresslevel=6, mtime=0), data_id
