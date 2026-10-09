"""
Terrorist attacks since 1970: an interactive map of the Global Terrorism Database.

Every event is drawn on the map (no sampling); see gtd_map/ for how. The basemaps work without any
API key. To use CARTO's Positron / Dark Matter styles instead, add  CARTO_API_KEY = "your-key"  to
.streamlit/secrets.toml (or to the app's Secrets on Streamlit Community Cloud).
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import uuid
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

import gtd_data as gd
from gtd_map import event_map

st.set_page_config(page_title="Global Terrorism Map", page_icon="🌍", layout="wide")

# ── Settings ─────────────────────────────────────────────────────────────────
DATA_FILE = "globalterrorism.parquet"
CONTACT_EMAIL = "casandjianewan@gmail.com"
MAX_TABLE_ROWS = 1_000      # the table shows the first 1,000 matching events; the CSV has all of them
CSV_DOWNLOAD = True         # False hides the CSV download (for a public app: the GTD's terms forbid redistributing the data)
INSTANT_CSV_ROWS = 10_000   # bigger CSV files are only built when someone asks for one
# Shown under the page, as START's terms of use ask
CITATION = ("START (National Consortium for the Study of Terrorism and Responses to Terrorism). (2022). "
            "Global Terrorism Database 1970 - 2020 [data file]. https://www.start.umd.edu/data-tools/GTD. "
            "Copyright University of Maryland 2022.")
MAP_HEIGHT = 620
MAP_KEY = "event_map"

DOT_COLOR = "#c8384b"       # events (single colour), and single-series charts
DOT_COLOR_DARK = "#e0566a"  # the same on dark basemaps
AREA_COLOR = "#1f5fbf"      # the "around a point" search area
# Three group colours + "Other" for "colour by" (the same on the map, its legend and the charts)
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a"], "dark": ["#3987e5", "#d95926", "#199e70"]}
OTHER_COLOR = "#898781"
RAMP = {"light": ["#cde2fb", "#104281"], "dark": ["#184f95", "#cde2fb"]}  # sequential (heat grid)

WHERE = {"anywhere": "Anywhere", "countries": "Countries", "around": "Around a point", "area": "Map area"}
DEATH_STEPS = [0, 1, 2, 5, 10, 20, 50, 100, 250, 500]
COLOR_BY = {"none": "One colour", "attack": "Attack type", "target": "Target", "weapon": "Weapon"}
MEASURES = {"Events": None, "Deaths": "killed", "Wounded": "wounded"}

ss = st.session_state


# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading events…")
def load():
    """Load and pack the events once per server process; every session shares this read-only copy."""
    ev = gd.load_events(DATA_FILE)
    blob, data_id = gd.pack_for_browser(ev)
    ev.data_id = data_id
    meta = {"options": {}, "group_counts": {}, "country_region": {}}
    for key, col in gd.FACETS.items():
        if col in ev.facets:
            meta["options"][key] = ev.facets[col].labels
    if "gname" in ev.facets:
        fac = ev.facets["gname"]
        counts = np.bincount(fac.codes[fac.codes >= 0], minlength=len(fac.labels))
        order = np.argsort(-counts, kind="stable")
        meta["options"]["groups"] = [fac.labels[i] for i in order if counts[i]]  # most active first
        meta["group_counts"] = {fac.labels[i]: int(counts[i]) for i in order}
    if "region_txt" in ev.facets:
        c, r = ev.facets["country_txt"], ev.facets["region_txt"]
        pairs = pd.DataFrame({"c": c.codes, "r": r.codes})
        pairs = pairs[(pairs.c >= 0) & (pairs.r >= 0)].value_counts().reset_index()
        pairs = pairs.drop_duplicates("c")  # each country's most common region
        meta["country_region"] = {c.labels[i]: r.labels[j] for i, j in zip(pairs.c, pairs.r)}
    return ev, blob, gd.color_group_maps(ev), meta


@st.cache_data(max_entries=32, show_spinner=False)
def text_hits(data_id: str, text: str) -> np.ndarray:
    return gd.text_mask(load()[0], text)


@st.cache_data(max_entries=64, show_spinner=False)
def matching_mask(data_id: str, fkey: tuple) -> np.ndarray:
    f = dict(fkey)
    hits = text_hits(data_id, f["text"]) if gd.search_terms(f["text"]) else None
    return gd.filter_mask(load()[0], f, hits)


@st.cache_data(max_entries=64, show_spinner=False)
def results_bounds(data_id: str, mask_id: str, _idx: np.ndarray):
    return gd.data_bounds(load()[0], _idx)


try:
    EV, BLOB, GROUPS, META = load()
except FileNotFoundError:
    st.error(f"Data file not found: `{DATA_FILE}`. Put the GTD parquet file next to app.py.")
    st.stop()
except ValueError as err:
    st.error(str(err))
    st.stop()

MIN_YEAR, MAX_YEAR = int(EV.year.min()), int(EV.year.max())
OPTIONS = META["options"]


# ── Basemaps ─────────────────────────────────────────────────────────────────
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
# Attributions as each Esri service states them (identical strings are shown once)
ESRI_ATTR = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors, and the GIS user community"
ESRI_LABELS_ATTR = ESRI_ATTR
ESRI_SAT_ATTR = "Imagery &copy; Esri &mdash; Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community"
OSM_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors'
CARTO_ATTR = (
    '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors '
    '&copy; <a href="https://carto.com/attributions" target="_blank">CARTO</a>'
)


def carto_key():
    """Optional CARTO key from an environment variable or Streamlit secrets."""
    key = os.environ.get("CARTO_API_KEY")
    if not key:
        try:
            key = st.secrets.get("CARTO_API_KEY")
        except Exception:  # no secrets configured
            key = None
    return key or None


def esri(service, maxzoom, attribution=ESRI_ATTR, labels=False):
    return {"tiles": [ESRI_URL.format(service)], "maxzoom": maxzoom, "attribution": attribution, "labels": labels}


def basemap_catalog(key):
    """Raster basemaps for the map. 'dark' switches the dots and overlays to their dark-map colours."""
    maps = {
        "Light": {"dark": False, "background": "#e4e4e1", "layers": [
            esri("Canvas/World_Light_Gray_Base", 16),
            esri("Canvas/World_Light_Gray_Reference", 16, ESRI_LABELS_ATTR, labels=True)]},
        "Dark": {"dark": True, "background": "#272727", "layers": [
            esri("Canvas/World_Dark_Gray_Base", 16),
            esri("Canvas/World_Dark_Gray_Reference", 16, ESRI_LABELS_ATTR, labels=True)]},
        "Streets": {"dark": False, "background": "#f2efe9", "layers": [
            {"tiles": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], "maxzoom": 19, "attribution": OSM_ATTR}]},
        "Satellite": {"dark": True, "background": "#1c2430", "layers": [
            esri("World_Imagery", 18, ESRI_SAT_ATTR),
            esri("Reference/World_Boundaries_and_Places", 18, ESRI_LABELS_ATTR, labels=True)]},
    }
    if key:
        for name, variant in (("Light", "light_all"), ("Dark", "dark_all")):
            urls = [f"https://{s}.basemaps.cartocdn.com/{variant}/{{z}}/{{x}}/{{y}}.png?key={quote(key, safe='')}"
                    for s in "abcd"]
            maps[name]["layers"] = [{"tiles": urls, "maxzoom": 18, "attribution": CARTO_ATTR}]
    return [{"id": name, **spec} for name, spec in maps.items()]


BASEMAPS = basemap_catalog(carto_key())
BASEMAP_IDS = [b["id"] for b in BASEMAPS]


# Newer Streamlit sizes elements with width="stretch"; older versions use use_container_width
STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.vega_lite_chart).parameters
           else {"use_container_width": True})


def md_escape(text: str) -> str:
    """Show user-typed or data text literally inside Markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|~<>$])", r"\\\1", str(text))


def short(label: str) -> str:
    """Display name for a category label (a few GTD labels are very long)."""
    return gd.SHORT_LABELS.get(label, label)


def theme_type() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:  # older Streamlit
        return "light"


# ── Filter state, defaults and shareable links ───────────────────────────────
DEFAULTS = {
    "text": "", "years": (MIN_YEAR, MAX_YEAR), "where": "anywhere", "regions": [], "countries": [],
    "lat": 30.0, "lon": 70.0, "radius": 50, "attack": [], "target": [], "weapon": [], "groups": [],
    "min_killed": 0, "suicide": False, "success": False, "no_doubt": False,
    "area": None,  # (west, south, east, north) when Location is "Map area"
}
DISPLAY = {"show": "Dots", "color_by": "none", "scale": False, "basemap": "Light",
           "measure": "Events", "sort": "Most recent", "as_tables": False}
# session-state key -> URL parameter
URL_KEYS = {
    "text": "q", "years": "years", "where": "where", "regions": "region", "countries": "country",
    "lat": "lat", "lon": "lon", "radius": "km", "attack": "attack", "target": "target", "weapon": "weapon",
    "groups": "group", "min_killed": "deaths", "suicide": "suicide", "success": "success",
    "no_doubt": "nodoubt", "area": "area", "show": "show", "color_by": "color", "scale": "scale",
    "basemap": "map",
}
LIST_KEYS = ["regions", "countries", "attack", "target", "weapon", "groups"]


def read_url():
    """Start the session from the link's parameters (ignoring anything invalid)."""
    qp = st.query_params

    def num(name, cast, lo, hi):
        try:
            return min(max(cast(qp[name]), lo), hi)
        except (KeyError, ValueError, TypeError):
            return None

    state = {}
    if "q" in qp:
        state["text"] = qp["q"][:200]
    if "years" in qp:
        try:
            y0, y1 = sorted(int(v) for v in qp["years"].split("-"))
            state["years"] = (max(y0, MIN_YEAR), min(y1, MAX_YEAR))
        except ValueError:
            pass
    if qp.get("where") in WHERE:
        state["where"] = qp["where"]
    for key in LIST_KEYS:
        valid = set(OPTIONS.get(key, []))
        chosen = [v for v in qp.get_all(URL_KEYS[key]) if v in valid]
        if chosen:
            state[key] = list(dict.fromkeys(chosen))
    for key, cast, lo, hi in (("lat", float, -90.0, 90.0), ("lon", float, -180.0, 180.0), ("radius", int, 5, 1000)):
        v = num(URL_KEYS[key], cast, lo, hi)
        if v is not None:
            state[key] = v
    if "radius" in state:
        state["radius"] = int(round(state["radius"] / 5) * 5)
    deaths = num("deaths", int, 0, 10_000)
    if deaths is not None:
        state["min_killed"] = max([s for s in DEATH_STEPS if s <= deaths])
    for key in ("suicide", "success", "no_doubt"):
        if qp.get(URL_KEYS[key]) == "1":
            state[key] = True
    if qp.get("show") in ("Dots", "Heatmap"):
        state["show"] = qp["show"]
    if qp.get("color") in COLOR_BY:
        state["color_by"] = qp["color"]
    if qp.get("scale") == "1":
        state["scale"] = True
    if qp.get("map") in BASEMAP_IDS:
        state["basemap"] = qp["map"]
    try:
        w, so, e, n = (float(v) for v in qp["area"].split(","))
        if -90 <= so < n <= 90 and w < e and e - w <= 720:
            state["area"] = (w, so, e, n)
    except (KeyError, ValueError):
        pass
    return state


def url_params() -> dict:
    """The current filters as URL parameters, defaults left out so links stay short."""
    where = ss["where"]
    params = {}
    for key, name in URL_KEYS.items():
        if key in ("regions", "countries") and where != "countries":
            continue
        if key in ("lat", "lon", "radius") and where != "around":
            continue
        if key == "area" and where != "area":
            continue
        value = ss.get(key)
        default = DEFAULTS.get(key, DISPLAY.get(key))
        if value is None or value == default or (key in LIST_KEYS and not value):
            continue
        if key == "years":
            value = f"{value[0]}-{value[1]}"
        elif key == "area":
            value = ",".join(f"{v:.4f}" for v in value)
        elif isinstance(value, bool):
            value = "1"
        elif key in LIST_KEYS:
            value = list(value)
        else:
            value = str(value)
        params[name] = value
    return params


def sync_url():
    """Keep the address bar a link to the current filters, so it can be shared as it is."""
    params = url_params()
    if params != ss.get("_url"):
        ss["_url"] = params
        st.query_params.from_dict(params)


def reset_filters():
    for key, value in DEFAULTS.items():
        ss[key] = value
    ss["_picking"] = False
    ss["_capture"] = None
    ss["selected"] = -1


def set_picking(on: bool):
    ss["_picking"] = on


def request_capture():
    """Ask the map for the area it shows; it answers with a "Map area" action (handle_map_value)."""
    ss["_capture"] = uuid.uuid4().hex[:8]


def on_where_change():
    if ss["where"] == "area":
        request_capture()  # "Map area" starts from what the map shows right now


def area_text(area) -> str:
    """'29.3°N to 37.7°N, 38.9°E to 48.6°E'"""
    w, so, e, n = area
    if e - w >= 360:
        return f"{abs(so):.1f}°{'N' if so >= 0 else 'S'} to {abs(n):.1f}°{'N' if n >= 0 else 'S'}, all longitudes"
    def lat(v):
        return f"{abs(v):.1f}°{'N' if v >= 0 else 'S'}"
    def lon(v):
        v = (v + 180) % 360 - 180
        return f"{abs(v):.1f}°{'E' if v >= 0 else 'W'}"
    return f"{lat(so)} to {lat(n)}, {lon(w)} to {lon(e)}"


# First run of a session: take the filters from the link, then fill in defaults.
if "_started" not in ss:
    ss["_started"] = True
    if theme_type() == "dark":
        ss["basemap"] = "Dark"
    for k, v in read_url().items():
        ss[k] = v
# Re-assigning every value keeps it while its widget is hidden (e.g. after switching location mode)
for k, v in {**DEFAULTS, **DISPLAY}.items():
    ss[k] = ss.get(k, v)
ss.setdefault("_picking", False)
ss.setdefault("_capture", None)
ss.setdefault("selected", -1)
if ss["basemap"] not in BASEMAP_IDS:
    ss["basemap"] = BASEMAP_IDS[0]
# Changes requested from the map (a picked centre, or "Search around here" in the event card)
if "_new_centre" in ss:
    ss["lat"], ss["lon"] = ss.pop("_new_centre")
if "_new_area" in ss:
    ss["area"] = ss.pop("_new_area")
if "_new_where" in ss:
    ss["where"] = ss.pop("_new_where")
if ss["where"] == "area" and not ss["area"] and not ss["_capture"]:
    request_capture()


# ── Sidebar ──────────────────────────────────────────────────────────────────
def group_label(name: str) -> str:
    n = META["group_counts"].get(name)
    return f"{name} ({n:,})" if n else name


with st.sidebar:
    st.header("Filters")
    st.text_input(
        "Search", key="text", placeholder="e.g. embassy, Bhutto",
        help="Finds events whose summary, place, target or perpetrator group mentions the text. "
             "Separate alternatives with commas.",
    )
    st.slider("Years", MIN_YEAR, MAX_YEAR, key="years")
    st.radio("Location", list(WHERE), format_func=WHERE.get, key="where", on_change=on_where_change)
    if ss["where"] == "countries":
        if "regions" in OPTIONS:
            st.multiselect("Regions", OPTIONS["regions"], key="regions", placeholder="Any region")
        countries = OPTIONS["countries"]
        if ss["regions"] and META["country_region"]:
            in_regions = [c for c in countries if META["country_region"].get(c) in ss["regions"]]
            countries = sorted(set(in_regions) | set(ss["countries"]))  # never drop a chosen country
        st.multiselect("Countries", countries, key="countries", placeholder="Any country",
                       help="Leave empty to include every country in the chosen regions.")
    elif ss["where"] == "around":
        col_lat, col_lon = st.columns(2)
        col_lat.number_input("Latitude", -90.0, 90.0, step=0.5, format="%.4f", key="lat")
        col_lon.number_input("Longitude", -180.0, 180.0, step=0.5, format="%.4f", key="lon")
        st.slider("Radius (km)", 5, 1000, step=5, key="radius")
        if ss["_picking"]:
            st.button("✖ Cancel picking", on_click=set_picking, args=(False,))
        else:
            st.button("📍 Pick the centre on the map", on_click=set_picking, args=(True,))
    elif ss["where"] == "area":
        st.caption(f"Events inside {area_text(ss['area'])}." if ss["area"] else "Reading the map view…")
        st.button("⛶ Use the current map view", on_click=request_capture,
                  help="Pan and zoom the map to the area you want first. The button on the map, "
                       "“Filter to this view”, does the same.")
    if ss["where"] != "around":
        ss["_picking"] = False

    st.divider()
    if "attack" in OPTIONS:
        st.multiselect("Attack type", OPTIONS["attack"], key="attack", placeholder="Any attack type")
    if "target" in OPTIONS:
        st.multiselect("Target", OPTIONS["target"], key="target", placeholder="Any target")
    if "weapon" in OPTIONS:
        st.multiselect("Weapon", OPTIONS["weapon"], key="weapon", placeholder="Any weapon", format_func=short)
    if "groups" in OPTIONS:
        st.multiselect(
            "Perpetrator group", OPTIONS["groups"], key="groups", format_func=group_label,
            placeholder="Any group", help="Sorted by number of events. Type to search.",
        )
    if EV.has("nkill"):
        st.select_slider("Deaths per event", DEATH_STEPS, key="min_killed",
                         format_func=lambda v: "Any" if v == 0 else f"At least {v:,}")
    if "suicide" in EV.flags:
        st.checkbox("Suicide attacks only", key="suicide")
    if "success" in EV.flags:
        st.checkbox("Successful attacks only", key="success")
    if "doubtterr" in EV.flags:
        st.checkbox(
            "Leave out doubtful cases", key="no_doubt",
            help="The GTD flags events where it is unclear whether the violence was terrorism "
                 "(for example insurgency, crime or internal conflict).",
        )
    st.button("↺ Reset filters", on_click=reset_filters)

filters = {k: (tuple(ss[k]) if isinstance(ss[k], (list, tuple)) else ss[k]) for k in DEFAULTS}
filters["text"] = filters["text"].strip()
fkey = tuple(sorted(filters.items()))
mask = matching_mask(EV.data_id, fkey)
mask_id = hashlib.md5(f"{EV.data_id}{fkey}".encode()).hexdigest()[:12]
if ss.get("_csv") and ss["_csv"]["mask_id"] != mask_id:
    del ss["_csv"]  # a CSV prepared for other filters: free the memory


# ── Header ───────────────────────────────────────────────────────────────────
def describe_filters(f) -> str:
    def few(items, n=3):
        items = list(items)
        return ", ".join(items) if len(items) <= n else ", ".join(items[:n]) + f" and {len(items) - n} more"
    parts = []
    if f["text"]:
        parts.append(f"mentioning “{f['text']}”")
    if f["years"] != (MIN_YEAR, MAX_YEAR):
        parts.append(f"{f['years'][0]}–{f['years'][1]}")
    if f["where"] == "countries" and (f["regions"] or f["countries"]):
        parts.append(few(f["countries"] or f["regions"]))
    elif f["where"] == "around":
        parts.append(f"within {f['radius']:,} km of {f['lat']:.2f}, {f['lon']:.2f}")
    elif f["where"] == "area" and f["area"]:
        parts.append(f"inside {area_text(f['area'])}")
    for key in ("attack", "target", "weapon", "groups"):
        if f[key]:
            parts.append(few(f[key], 2))
    if f["min_killed"]:
        parts.append(f"at least {f['min_killed']:,} deaths")
    for key, text in (("suicide", "suicide attacks"), ("success", "successful attacks"),
                      ("no_doubt", "no doubtful cases")):
        if f[key]:
            parts.append(text)
    return "; ".join(parts)


with st.container():  # one fixed slot, so the map below never shifts and reloads
    st.title(f"Terrorist attacks since {MIN_YEAR}")
    st.caption(
        f"Explore all {EV.n:,} mapped events from the Global Terrorism Database. The filters in the sidebar "
        "decide what the map, numbers, tables and charts show. To focus on one area, zoom the map to it and "
        "press “Filter to this view”. Hover over a dot for a summary, click it for details, or press play "
        "on the timeline to watch the years go by."
    )
    active = describe_filters(filters)
    if active:
        st.markdown(f"**Filters:** {md_escape(active)}")


# ── Charts ───────────────────────────────────────────────────────────────────
def ink() -> str:
    return "#fafafa" if theme_type() == "dark" else "#31333f"


def weights(idx, measure):
    col = MEASURES[measure]
    if col is None:
        return None
    values = EV.killed if col == "killed" else EV.wounded
    return np.nan_to_num(values[idx], nan=0.0)


def tally(idx, col, measure, top=None):
    """Events (or deaths, wounded) per label of a categorical column, largest first."""
    fac = EV.facets[col]
    totals = np.bincount(fac.codes[idx] + 1, weights=weights(idx, measure), minlength=len(fac.labels) + 1)
    df = pd.DataFrame({"label": ["Unknown"] + [short(x) for x in fac.labels], "value": totals})
    df = df.groupby("label", as_index=False)["value"].sum()  # merges a recorded "Unknown" with missing values
    df = df[df["value"] > 0].sort_values("value", ascending=False)
    return df.head(top) if top else df


def year_ticks(y0: int, y1: int) -> list[int]:
    span = y1 - y0 + 1
    step = 10 if span > 30 else 5 if span > 12 else 1
    return [y for y in range(y0, y1 + 1) if y % step == 0] or [y0]


def tip(field, title=None, kind="nominal", fmt=None) -> dict:
    t = {"field": field, "type": kind, "title": title or field}
    if fmt:
        t["format"] = fmt
    return t


# Charts are plain Vega-Lite specs: the same charts as Altair would make, without its per-rerun
# schema validation (about half the time of a rerun) and independent of the installed Altair version.
#
# Streamlit's chart theme sets labelFlush and labelOverlap on every axis. On a category axis, flush
# nudges the first label off its bar, which then counts as overlapping the next one, and Vega hides
# every other label. Category axes switch both off (CATEGORY_AXIS), and their charts give each row a
# fixed height ({"step": ...}) instead of a total height that the axes and legend eat into.
CATEGORY_AXIS = {"labelOverlap": False, "labelFlush": False}
ROW_STEP = 24  # pixels per bar or row


def hbar(df, measure, title, label_limit=260) -> dict:
    """Horizontal bars, one colour, value at the tip."""
    vmax = float(df["value"].max()) if len(df) else 1.0
    return {
        "height": {"step": ROW_STEP},
        "encoding": {
            "y": {"field": "label", "type": "nominal", "sort": df["label"].tolist(), "title": None,  # largest first
                  "axis": {**CATEGORY_AXIS, "labelLimit": label_limit, "ticks": False, "domain": False}},
            "x": {"field": "value", "type": "quantitative", "title": None,
                  "scale": {"domain": [0, vmax * 1.2], "nice": False}, "axis": {"format": ",.0f", "tickCount": 4}},
            "tooltip": [tip("label", title), tip("value", measure, "quantitative", ",.0f")],
        },
        "layer": [
            {"mark": {"type": "bar", "color": DOT_COLOR, "cornerRadiusEnd": 4, "size": 15}},
            {"mark": {"type": "text", "align": "left", "dx": 5, "color": ink()},
             "encoding": {"text": {"field": "value", "type": "quantitative", "format": ",.0f"}}},
        ],
    }


def show(df, spec, as_table, table=None, columns=None):
    """A chart, or the same numbers as a table."""
    if as_table:
        st.dataframe((df if table is None else table).rename(columns=columns or {}), hide_index=True)
    else:
        st.vega_lite_chart(df, spec, **STRETCH)


def over_time(idx, measure, color_by, years):
    """Events (or deaths, wounded) per year, stacked by the map's colour groups when they are on."""
    y0, y1 = years
    w = weights(idx, measure)
    span = y1 - y0 + 1
    rel = EV.year[idx].astype(np.int64) - y0
    groups = GROUPS.get(color_by) if color_by != "none" else None
    enc = {
        "x": {"field": "Year", "type": "ordinal", "title": None,
              "axis": {"values": year_ticks(y0, y1), "labelAngle": 0, "labelFlush": False}},
        "y": {"field": "value", "type": "quantitative", "title": measure, "stack": "zero",
              "axis": {"format": ",.0f"}},
    }
    if groups:
        col = gd.COLOR_GROUPS[color_by][0]
        group = np.asarray(groups["byCode"])[EV.facets[col].codes[idx] + 1]
        totals = np.bincount(rel * 4 + group, weights=w, minlength=span * 4).reshape(span, 4)
        df = pd.DataFrame({
            "Year": np.repeat(np.arange(y0, y1 + 1), 4),
            "Group": np.tile(groups["labels"], span),
            "order": np.tile(np.arange(4), span),
            "value": totals.ravel(),
        })
        palette = SERIES["dark" if theme_type() == "dark" else "light"] + [OTHER_COLOR]
        enc["color"] = {"field": "Group", "type": "nominal", "title": None,
                        "scale": {"domain": groups["labels"], "range": palette},
                        "legend": {"orient": "top", "direction": "horizontal"}}
        enc["order"] = {"field": "order", "type": "quantitative"}  # first group sits on the baseline
        enc["tooltip"] = [tip("Year", kind="ordinal"), tip("Group"), tip("value", measure, "quantitative", ",.0f")]
        surface = "#0e1117" if theme_type() == "dark" else "#ffffff"
        mark = {"type": "bar", "stroke": surface, "strokeWidth": 1}  # a thin gap between stacked segments
        table = df.pivot(index="Year", columns="Group", values="value")[groups["labels"]].reset_index()
    else:
        df = pd.DataFrame({"Year": np.arange(y0, y1 + 1), "value": np.bincount(rel, weights=w, minlength=span)})
        enc["tooltip"] = [tip("Year", kind="ordinal"), tip("value", measure, "quantitative", ",.0f")]
        mark = {"type": "bar", "color": DOT_COLOR, "cornerRadiusEnd": 3}
        table = df.rename(columns={"value": measure})
    return df, {"height": 320, "mark": mark, "encoding": enc}, table


def group_activity(idx, measure, years, top_groups):
    """Heat grid: the most active groups (rows) by year (columns)."""
    fac = EV.facets["gname"]
    lookup = np.full(len(fac.labels) + 1, -1)  # code -> row in the grid
    for row, name in enumerate(top_groups):
        lookup[fac.labels.index(name)] = row
    rows = lookup[fac.codes[idx]]
    keep = rows >= 0
    y0, y1 = years
    span = y1 - y0 + 1
    w = weights(idx, measure)
    totals = np.bincount(rows[keep] * span + (EV.year[idx][keep] - y0),
                         weights=None if w is None else w[keep], minlength=len(top_groups) * span)
    df = pd.DataFrame({
        "Group": np.repeat(top_groups, span), "Year": np.tile(np.arange(y0, y1 + 1), len(top_groups)),
        "value": totals,
    })
    df = df[df["value"] > 0]
    spec = {
        "height": {"step": ROW_STEP},
        "mark": {"type": "rect", "cornerRadius": 2},
        "encoding": {
            "x": {"field": "Year", "type": "ordinal", "title": None, "scale": {"domain": list(range(y0, y1 + 1))},
                  "axis": {"values": year_ticks(y0, y1), "labelAngle": 0, "labelFlush": False}},
            "y": {"field": "Group", "type": "nominal", "title": None,
                  "scale": {"domain": list(top_groups)},  # most active first; every group gets its row
                  "axis": {**CATEGORY_AXIS, "labelLimit": 340}},
            "color": {"field": "value", "type": "quantitative", "title": measure,
                      "scale": {"range": RAMP["dark" if theme_type() == "dark" else "light"]},
                      "legend": {"orient": "top", "format": ",.0f"}},
            "tooltip": [tip("Group"), tip("Year", kind="ordinal"), tip("value", measure, "quantitative", ",.0f")],
        },
    }
    return df, spec


# ── Table and download ───────────────────────────────────────────────────────
TABLE_CONFIG = {
    "Date": st.column_config.TextColumn(width="small"),
    "Country": st.column_config.TextColumn(width="medium"),
    "City": st.column_config.TextColumn(width="medium"),
    "Attack": st.column_config.TextColumn(width="medium"),
    "Target": st.column_config.TextColumn(width="medium"),
    "Perpetrator": st.column_config.TextColumn(width="medium"),
    "Killed": st.column_config.NumberColumn(format="%d", width="small"),
    "Wounded": st.column_config.NumberColumn(format="%d", width="small"),
    "Summary": st.column_config.TextColumn(width="large"),
}
CSV_COLUMNS = ["eventid", "iyear", "imonth", "iday", "country_txt", "region_txt", "provstate", "city",
               "latitude", "longitude", "attacktype1_txt", "targtype1_txt", "target1", "weaptype1_txt",
               "gname", "nkill", "nwound", "suicide", "success", "summary"]


def events_table(idx) -> pd.DataFrame:
    df = EV.df.iloc[idx]
    out = {"Date": gd.format_dates(df).to_numpy()}
    for name, col in (("Country", "country_txt"), ("City", "city"), ("Attack", "attacktype1_txt"),
                      ("Target", "targtype1_txt"), ("Perpetrator", "gname")):
        if col in df:
            out[name] = df[col].astype("string").to_numpy()
    for name, col in (("Killed", "nkill"), ("Wounded", "nwound")):
        if col in df:
            out[name] = df[col].astype("Int64").to_numpy()
    if "summary" in df:
        out["Summary"] = df["summary"].fillna("").to_numpy()
    return pd.DataFrame(out)


def to_csv(idx) -> bytes:
    cols = [c for c in CSV_COLUMNS if c in EV.df.columns]
    order = idx[np.argsort(-EV.datekey[idx], kind="stable")]
    return EV.df.iloc[order][cols].to_csv(index=False).encode("utf-8")


def csv_download(idx, mask_id):
    """A CSV of all the matching events, behind an ordinary download button.

    The file is built once per set of filters (and only on request when it's big), then kept for this
    session until the filters change. (Streamlit's "build on click" downloads finish after the click,
    which some browsers and hosts block.)
    """
    ready = ss.get("_csv")  # (dropped when the filters change, see mask_id)
    big = len(idx) > INSTANT_CSV_ROWS
    if ready is None and (not big or st.button(f"Prepare a CSV of the {len(idx):,} matching events", key="csv_prepare")):
        with st.spinner("Preparing the file…"):
            ready = ss["_csv"] = {"mask_id": mask_id, "data": to_csv(idx)}
    if ready:
        size = len(ready["data"]) / 1e6
        st.download_button(
            f"⬇️ Download {len(idx):,} events (CSV, {size:,.1f} MB)" if size >= 0.1 else f"⬇️ Download {len(idx):,} events (CSV)",
            ready["data"], "gtd_events.csv", "text/csv", key="csv_download", on_click="ignore",
            type="primary" if big else "secondary",
        )


# ── Map events (clicks, picked centres, card buttons, table rows) ────────────
def handle_map_value(value: dict):
    seq = value.get("pickSeq")
    if seq and seq != ss.get("_pick_seq"):
        ss["_pick_seq"] = seq
        if ss["_picking"] and value.get("pick"):
            ss["_new_centre"] = tuple(value["pick"])
            ss["_picking"] = False
            st.rerun(scope="app")
    seq = value.get("actionSeq")
    if seq and seq != ss.get("_action_seq"):
        ss["_action_seq"] = seq
        action = value.get("action") or {}
        kind = action.get("type")
        if kind == "around":
            ss["_new_centre"] = (action["lat"], action["lon"])
            ss["_new_where"] = "around"
            st.rerun(scope="app")
        elif kind == "area" and action.get("bounds"):
            ss["_new_area"] = tuple(round(float(v), 4) for v in action["bounds"])
            ss["_new_where"] = "area"
            ss["_capture"] = None
            st.rerun(scope="app")
        elif kind == "area_clear":
            ss["_new_where"] = "anywhere"
            st.rerun(scope="app")
    seq = value.get("selectSeq")
    if seq and seq != ss.get("_select_seq"):
        ss["_select_seq"] = seq
        ss["selected"] = int(value.get("selected", -1))


def handle_table_selection():
    """A row clicked in the events table selects that event on the map."""
    shown = ss.get("_table")
    if not shown:
        return
    state = ss.get(shown["key"])
    rows = list(state["selection"]["rows"]) if state else []
    if rows != shown["rows"]:
        shown["rows"] = rows
        ss["selected"] = int(shown["events"][rows[0]]) if rows else -1


def circle_bounds(lat, lon, radius_km):
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(np.cos(np.radians(lat)), 0.05))
    return [lon - dlon, max(lat - dlat, -85.0), lon + dlon, min(lat + dlat, 85.0)]


def fit_request(f, idx):
    """Where the map should fly when the location-like filters change (and for the 'fit' button)."""
    if f["where"] == "area" and f["area"]:
        # The area came from the map's own view, so don't move the map, except to open a shared link
        return {"id": "area", "bounds": list(f["area"]), "max_zoom": 18, "initial": True, "fly": False}
    if f["where"] == "countries":
        loc = ("countries", f["regions"], f["countries"])
    elif f["where"] == "around":
        loc = ("around", f["lat"], f["lon"], f["radius"])
    else:
        loc = ("anywhere",)
    loc += (f["groups"], f["text"])
    fit_id = hashlib.md5(repr(loc).encode()).hexdigest()[:10]
    if f["where"] == "around":
        bounds, max_zoom = circle_bounds(f["lat"], f["lon"], f["radius"]), 13
    else:
        bounds, max_zoom = results_bounds(EV.data_id, mask_id, idx), 8
    initial = f["where"] != "anywhere" or bool(f["groups"]) or bool(f["text"])
    return {"id": fit_id, "bounds": bounds, "max_zoom": max_zoom, "initial": initial, "fly": True}


# ── Page: numbers, map, tables and charts ────────────────────────────────────
# Everything below shows the same events: the ones that match the filters.
idx = np.flatnonzero(mask)
handle_table_selection()  # before the map, so a clicked row shows on the map in this run

stats = [("Matching events", f"{len(idx):,}", "Events that pass all the filters in the sidebar.")]
if EV.has("nkill"):
    stats.append(("Deaths", f"{int(np.nansum(EV.killed[idx])):,}",
                  "Recorded deaths, including perpetrators. Unknown tolls count as zero."))
if EV.has("nwound"):
    stats.append(("Wounded", f"{int(np.nansum(EV.wounded[idx])):,}", None))
stats.append(("Countries", f"{len(np.unique(EV.facets['country_txt'].codes[idx])):,}", None))
for col, (label, number, help_text) in zip(st.columns(len(stats)), stats):
    col.metric(label, number, help=help_text)

# Map display options, in one row above the map
c_show, c_color, c_scale, c_base = st.columns([1.2, 1.2, 1, 1], vertical_alignment="bottom")
c_show.radio("Show events as", ["Dots", "Heatmap"], key="show", horizontal=True)
c_color.selectbox("Colour dots by", list(COLOR_BY), format_func=COLOR_BY.get, key="color_by",
                  disabled=ss["show"] == "Heatmap" or not GROUPS)
c_scale.toggle("Scale by deaths", key="scale", disabled=not EV.has("nkill"),
               help="Dots: size shows the death toll. Heatmap: density of deaths instead of events.")
c_base.selectbox("Basemap", BASEMAP_IDS, key="basemap")
color_by = ss["color_by"] if ss["color_by"] in GROUPS else "none"


@st.fragment
def map_section(mask: np.ndarray, mask_id: str, f: dict, idx: np.ndarray):
    """The map. Clicking a dot reruns only this part; panning and zooming rerun nothing."""
    handle_map_value(ss.get(MAP_KEY) or {})
    selected = ss["selected"]
    if selected >= EV.n or (selected >= 0 and not mask[selected]):
        selected = ss["selected"] = -1  # the filters no longer include it

    # Messages go in a fixed slot above the map, so the map never moves and reloads
    with st.container():
        if ss["_picking"]:
            st.info("Click anywhere on the map to set the centre of the search area.", icon="📍")
        if not len(idx):
            st.warning("No events match these filters."
                       + (" Try a bigger radius or another centre." if f["where"] == "around" else ""))

    area_on = f["where"] == "area"
    event_map(
        data_id=EV.data_id, blob_gz=BLOB,
        mask=np.packbits(mask, bitorder="little").tobytes(), mask_id=mask_id,
        year_range=f["years"], basemaps=BASEMAPS, basemap=ss["basemap"],
        mode="heatmap" if ss["show"] == "Heatmap" else "points",
        color_by=color_by, groups=GROUPS, scale=bool(ss["scale"]) and EV.has("nkill"),
        dot_color=DOT_COLOR, dot_color_dark=DOT_COLOR_DARK, area_color=AREA_COLOR,
        circle={"lat": f["lat"], "lon": f["lon"], "radius_km": f["radius"]} if f["where"] == "around" else None,
        area=list(f["area"]) if area_on and f["area"] else None, area_on=area_on,
        capture_id=ss["_capture"], picking=ss["_picking"],
        selected=selected, select_seq=ss.get("_select_seq"),
        detail=gd.event_details(EV, selected) if selected >= 0 else None,
        fit=fit_request(f, idx), height=MAP_HEIGHT, key=MAP_KEY,
    )


map_section(mask, mask_id, filters, idx)

# Tables and charts for the same matching events
c_measure, c_tables = st.columns([3, 1], vertical_alignment="bottom")
measures = ["Events"] + [m for m in ("Deaths", "Wounded") if EV.has("nkill" if m == "Deaths" else "nwound")]
if ss["measure"] not in measures:
    ss["measure"] = "Events"
measure = c_measure.radio("Charts count", measures, key="measure", horizontal=True,
                          help="What the charts add up, for the events matching the filters.")
as_tables = c_tables.toggle("Show charts as tables", key="as_tables")

tab_events, tab_time, tab_what, tab_who = st.tabs(["📋 Events", "📈 Over time", "🧭 Breakdown", "👥 Perpetrators"])
with tab_events:
    if not len(idx):
        st.info("No events match the filters.")
    else:
        order = st.radio("Order", ["Most recent", "Deadliest"], key="sort", horizontal=True,
                         label_visibility="collapsed")
        if order == "Deadliest":
            rank = np.lexsort((-EV.datekey[idx], -np.nan_to_num(EV.killed[idx], nan=-1)))
        else:
            rank = np.argsort(-EV.datekey[idx], kind="stable")
        top = idx[rank[:MAX_TABLE_ROWS]]
        st.caption(
            (f"{len(idx):,} matching events; showing the {MAX_TABLE_ROWS:,} {order.lower()}. "
             + ("The download includes all of them. " if CSV_DOWNLOAD else "") if len(idx) > MAX_TABLE_ROWS else "")
            + "Click a row to show the event on the map."
        )
        key = "events_" + hashlib.md5(top.tobytes()).hexdigest()[:10]
        state = ss.get(key)
        ss["_table"] = {"key": key, "events": top, "rows": list(state["selection"]["rows"]) if state else []}
        st.dataframe(events_table(top), key=key, on_select="rerun", selection_mode="single-row",
                     hide_index=True, height=420, column_config=TABLE_CONFIG)
        if CSV_DOWNLOAD:
            csv_download(idx, mask_id)

with tab_time:
    if not len(idx):
        st.info("Nothing to chart: no events match the filters.")
    else:
        df, spec, table = over_time(idx, measure, color_by, filters["years"])
        title = f"{measure} per year"
        if color_by != "none":
            title += f", by {COLOR_BY[color_by].lower()}"
        st.markdown(f"**{title}**")
        show(df, spec, as_tables, table)
        if filters["years"][0] <= 1993 <= filters["years"][1] and not (EV.year == 1993).any():
            st.caption("The GTD has no records for 1993: that year's original files were lost.")

with tab_what:
    if not len(idx):
        st.info("Nothing to chart: no events match the filters.")
    else:
        left, right = st.columns(2, gap="large")
        for box, col, title, top_n in ((left, "attacktype1_txt", "Attack type", None),
                                       (right, "weaptype1_txt", "Weapon", 10),
                                       (left, "targtype1_txt", "Target", 10),
                                       (right, "country_txt", "Country", 12)):
            if col not in EV.facets:
                continue
            with box:
                df = tally(idx, col, measure, top_n)
                st.markdown(f"**{measure} by {title.lower()}**" + (f" (top {top_n})" if top_n else ""))
                show(df, hbar(df, measure, title), as_tables, columns={"label": title, "value": measure})

with tab_who:
    if "gname" not in EV.facets or not len(idx):
        st.info("Nothing to chart: no events match the filters.")
    else:
        df = tally(idx, "gname", measure)
        unknown = float(df.loc[df["label"] == "Unknown", "value"].sum())
        total = float(df["value"].sum())
        named = df[df["label"] != "Unknown"]
        if total:
            st.caption(f"{unknown / total:.0%} of the {measure.lower()} have no attributed perpetrator; "
                       "they are left out below.")
        if named.empty:
            st.info("None of the matching events has a named perpetrator group.")
        else:
            st.markdown(f"**{measure} by group** (top 15)")
            top15 = named.head(15)
            show(top15, hbar(top15, measure, "Group", label_limit=340), as_tables,
                 columns={"label": "Group", "value": measure})
            st.markdown(f"**{measure} per year for the 10 most active groups**")
            df_g, spec = group_activity(idx, measure, filters["years"], named["label"].head(10).tolist())
            show(df_g, spec, as_tables, columns={"value": measure})

sync_url()

st.divider()
st.caption(
    f"Data: {CITATION} "
    + (f"{EV.dropped:,} events without usable coordinates are not shown. " if EV.dropped else "")
    + f"For questions or the full dataset, contact {CONTACT_EMAIL}."
)
