"""The event map: a Streamlit component that draws every event with WebGL (MapLibre + deck.gl).

Why a custom component: folium/st_folium and st.pydeck_chart re-send every point to the browser on
each rerun, which is what limited the old map to a 10,000-point sample. Here the browser downloads
all events once, as a compact binary file (about 2.6 MB for the whole GTD), and each rerun only
sends one bit per event saying whether it matches the filters (about 26 KB). The browser filters
on the GPU, so the full 200k+ events pan, zoom and filter smoothly. Panning and zooming stay in
the browser: they never rerun the app.
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

FRONTEND = Path(__file__).parent / "frontend"


def _write_data(root: Path, data_id: str, blob_gz: bytes) -> str:
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    name = f"events-{data_id}.bin"
    for path, payload in ((data_dir / f"{name}.gz", blob_gz), (data_dir / name, None)):
        if not path.exists():
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(payload if payload is not None else gzip.decompress(blob_gz))
            tmp.replace(path)  # atomic, so a browser never reads a half-written file
    for old in data_dir.glob("events-*"):  # files from earlier versions of the data
        if data_id not in old.name:
            old.unlink(missing_ok=True)
    return f"data/{name}"


@st.cache_resource(show_spinner=False)
def _declare(data_id: str, _blob_gz: bytes):
    """Put the events file next to the map's HTML (served by Streamlit), then register the component.

    Falls back to a temporary copy of the frontend when the app folder is read-only.
    """
    for root in (FRONTEND, Path(tempfile.gettempdir()) / "gtd_event_map"):
        try:
            if root != FRONTEND:
                shutil.copytree(FRONTEND, root, dirs_exist_ok=True, ignore=shutil.ignore_patterns("data"))
            url = _write_data(root, data_id, _blob_gz)
            return components.declare_component("event_map", path=str(root)), url
        except OSError:
            continue
    raise RuntimeError("Could not write the map's data file to the app folder or to a temporary folder.")


def event_map(
    *,
    data_id: str,
    blob_gz: bytes,
    mask: bytes,
    mask_id: str,
    year_range: tuple[int, int],
    basemaps: list[dict],
    basemap: str,
    mode: str = "points",
    color_by: str = "none",
    groups: dict | None = None,
    scale: bool = False,
    dot_color: str = "#c8384b",
    dot_color_dark: str = "#e0566a",
    area_color: str = "#1f5fbf",
    circle: dict | None = None,
    area: list | None = None,
    area_on: bool = False,
    capture_id: str | None = None,
    picking: bool = False,
    selected: int = -1,
    select_seq: float | None = None,
    detail: dict | None = None,
    fit: dict | None = None,
    height: int = 600,
    key: str = "event_map",
):
    """Render the map and return its latest value (clicked event, picked centre, button actions).

    The map doesn't report panning or zooming: only explicit actions change the rest of the page.
    `capture_id`: when it changes, the map answers with an "area" action holding its current bounds.
    """
    component, url = _declare(data_id, blob_gz)
    return component(
        data_url=f"{url}.gz", data_url_raw=url, mask=mask, mask_id=mask_id,
        year_range=list(year_range), basemaps=basemaps, basemap=basemap, mode=mode,
        color_by=color_by, groups=groups or {}, scale=scale, dot_color=dot_color,
        dot_color_dark=dot_color_dark, area_color=area_color, circle=circle, area=area, area_on=area_on,
        capture_id=capture_id, picking=picking, selected=selected, select_seq=select_seq, detail=detail,
        fit=fit, height=height, key=key, default=None,
    )
