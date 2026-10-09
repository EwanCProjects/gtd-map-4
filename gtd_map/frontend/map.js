/* Event map: a Streamlit custom component (components v1 protocol, no build step).
 *
 * Python sends on every rerun (see gtd_map/__init__.py):
 *   data_url  the events file. It is downloaded once and then kept in memory and in the browser cache.
 *   mask      one bit per event: does it match the filters? (~26 KB for 200k events), plus mask_id.
 *   display settings, the selected event and its details, the search circle, a "fit" request...
 * The map sends back one value object, only when the user does something (panning and zooming
 * stay in the browser and never rerun the app):
 *   selected  the event clicked on the map (selectSeq tells a new click from an old one)
 *   pick      a clicked search centre, while picking
 *   action    a button pressed: "Search around here", "Filter to this view", "Clear"
 */
(function () {
  "use strict";

  // ── Streamlit protocol ─────────────────────────────────────────────────────
  function send(type, data) {
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, data), "*");
  }
  var Streamlit = {
    ready: function () { send("streamlit:componentReady", { apiVersion: 1 }); },
    setValue: function (value) { send("streamlit:setComponentValue", { value: value, dataType: "json" }); },
    setHeight: function (height) { send("streamlit:setFrameHeight", { height: height }); },
  };

  // ── Constants ──────────────────────────────────────────────────────────────
  // Categorical colours for "colour by": three groups + "Other" (validated for colour-vision
  // deficiencies on light and dark basemaps; a dot map cannot keep more than three hues apart).
  var PALETTE = { light: ["#2a78d6", "#eb6834", "#1baf7a"], dark: ["#3987e5", "#d95926", "#199e70"] };
  var OTHER = "#898781";
  // Heatmap: one hue, low to high. On dark maps the ramp runs dark to light.
  var HEAT = {
    light: ["#f6d3d7", "#eba3ac", "#de6f7e", "#c8384b", "#9b2234", "#661220"],
    dark: ["#4f1620", "#7d2232", "#ac2f43", "#d6495f", "#ef8794", "#fbcfd4"],
  };
  var CODES = { attack: "attack", target: "target", weapon: "weapon" };
  var UNKNOWN = 65535;  // casualty count not recorded
  var STEP_MS = 650;    // time-lapse speed per year
  // Events recorded at exactly the same point (the GTD often uses a city's centre) are fanned out
  // around it in a sunflower pattern, so each one gets its own dot. Bigger stacks are squeezed into
  // the size of a cluster of stackCap() dots: small when zoomed out (so cities don't swell over
  // their neighbours), up to STACK_MAX when zoomed in on a city.
  var STACK_MIN = 30, STACK_MAX = 200;
  var GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
  var NF = new Intl.NumberFormat("en-US");
  var MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"];

  var $ = function (id) { return document.getElementById(id); };
  var ui = {
    map: $("map"), legend: $("legend"), banner: $("banner"), card: $("card"), tooltip: $("tooltip"),
    status: $("status"), timeline: $("timeline"), play: $("tl-play"), label: $("tl-label"),
    chart: $("tl-chart"), clear: $("tl-clear"),
  };

  // ── State ──────────────────────────────────────────────────────────────────
  var S = {
    args: null, height: 0,
    map: null, overlay: null, ready: false, labelBefore: undefined, basemap: null,
    data: null, dataUrl: null, failed: false,
    attr: {}, filterValues: null, eventsData: null, heatData: null, heatKey: null,
    constColor: [200, 56, 75, 230], colorKey: null, radiusKey: null,
    maskId: null, matching: 0, yearCounts: null, groupCountKey: null, groupCounts: null,
    layoutKey: null, stackSize: null, stackRank: null, offsets: null,
    selected: -1, fitId: null, pointScale: 2, areaKey: null, area: null,
    year: null, playing: false, timer: null,
    out: { selected: -1, selectSeq: 0, pick: null, pickSeq: 0, action: null, actionSeq: 0 },
    captureId: null, veilKey: null, veil: null,
    lastSent: "", frame: 0,
  };
  var FILTER = null;  // created once: a new extension instance would recompile the shaders

  // ── Small helpers ──────────────────────────────────────────────────────────
  function el(tag, props) {
    var node = document.createElement(tag);
    if (props) Object.keys(props).forEach(function (k) {
      if (k === "text") node.textContent = props[k];
      else if (k === "className") node.className = props[k];
      else if (k.slice(0, 2) === "on") node.addEventListener(k.slice(2), props[k]);
      else node.setAttribute(k, props[k]);
    });
    for (var i = 2; i < arguments.length; i++) {
      var child = arguments[i];
      if (child == null || child === false) continue;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
  }
  function rgb(hex) {
    var h = hex.replace("#", "");
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  // Any CSS colour (hex, rgb(), named) as [r, g, b], using the canvas colour parser
  var colorCtx = document.createElement("canvas").getContext("2d");
  function parseColor(color, fallback) {
    if (!color) return fallback;
    colorCtx.fillStyle = "#000001";
    colorCtx.fillStyle = color;
    var v = colorCtx.fillStyle;
    if (v === "#000001") return fallback;  // not a colour the browser understands
    if (v[0] === "#") return rgb(v);
    var m = v.match(/[\d.]+/g);
    return m && m.length >= 3 ? [+m[0], +m[1], +m[2]] : fallback;
  }
  function round(x, d) { var f = Math.pow(10, d); return Math.round(x * f) / f; }
  function wrapLon(lon) { return ((lon + 180) % 360 + 360) % 360 - 180; }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function label(dict, code) { return code > 0 && dict ? dict[code - 1] : null; }
  function known(s) { return s && s !== "Unknown" ? s : null; }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function nowSeq() { return Date.now() + Math.random(); }  // differs from anything sent before a reload

  function dateText(y, m, d) {
    if (!m) return String(y);
    return (d ? d + " " : "") + MONTHS[m - 1] + " " + y;
  }
  function tollText(k, w) {
    if (k === UNKNOWN && w === UNKNOWN) return "Casualties not recorded";
    if (k === 0 && (w === 0 || w === UNKNOWN)) return w === 0 ? "No one killed or wounded" : "No deaths";
    var parts = [];
    if (k !== UNKNOWN) parts.push(NF.format(k) + " killed");
    if (w !== UNKNOWN) parts.push(NF.format(w) + " wounded");
    return parts.join(", ");
  }

  function showStatus(text, isError) {
    ui.status.textContent = text;
    ui.status.className = isError ? "error" : "";
    ui.status.hidden = !text;
  }

  // ── Messages from Streamlit ────────────────────────────────────────────────
  window.addEventListener("message", function (event) {
    var msg = event.data;
    if (!msg || msg.type !== "streamlit:render") return;
    S.args = msg.args || {};
    applyTheme(msg.theme);
    try { update(); } catch (err) { fail(err); }
  });

  function fail(err) {
    console.error(err);
    showStatus("The map could not be drawn: " + (err && err.message ? err.message : err), true);
  }

  function applyTheme(theme) {
    if (!theme) return;
    var root = document.documentElement.style;
    if (theme.backgroundColor) root.setProperty("--bg", theme.backgroundColor);
    if (theme.textColor) root.setProperty("--ink", theme.textColor);
    if (theme.font) root.setProperty("--font", theme.font + ", system-ui, -apple-system, 'Segoe UI', sans-serif");
    // Derived from the theme's own colours, so custom Streamlit themes match too
    var dark = theme.base === "dark";
    var ink = parseColor(theme.textColor, dark ? [250, 250, 250] : [49, 51, 63]);
    var bg = parseColor(theme.backgroundColor, dark ? [14, 17, 23] : [255, 255, 255]);
    var muted = ink.map(function (v, i) { return Math.round(v * 0.64 + bg[i] * 0.36); });
    root.setProperty("--muted", "rgb(" + muted.join(",") + ")");
    root.setProperty("--hairline", "rgba(" + ink.join(",") + ",0.15)");
    root.setProperty("--panel", "rgba(" + bg.join(",") + ",0.93)");
  }

  function update() {
    var a = S.args;
    setHeight(a.height);
    if (S.args.dot_color) root("--accent", a.dot_color);
    if (typeof maplibregl === "undefined" || typeof deck === "undefined") {
      showStatus("The map libraries could not be loaded from cdn.jsdelivr.net. " +
        "Check the internet connection, or whether a firewall or ad blocker blocks that site.", true);
      return;
    }
    if (!S.map && !S.failed) createMap(a);
    if (S.map && a.capture_id && a.capture_id !== S.captureId) {
      S.captureId = a.capture_id;
      sendAction({ type: "area", bounds: currentBounds(), capture: a.capture_id });
    }
    if (a.data_url && a.data_url !== S.dataUrl) loadData(a.data_url, a.data_url_raw);
    refresh();
  }
  function root(name, value) { document.documentElement.style.setProperty(name, value); }

  function setHeight(h) {
    h = Math.max(320, Number(h) || 600);
    if (S.height === h) return;
    S.height = h;
    document.body.style.height = h + "px";
    Streamlit.setHeight(h);
    if (S.map) S.map.resize();
  }

  // ── Map and basemaps ───────────────────────────────────────────────────────
  function basemapDef(id) {
    var list = (S.args && S.args.basemaps) || [];
    for (var i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
    return list[0];
  }
  function isDark() { var b = basemapDef(S.args.basemap); return !!(b && b.dark); }

  function buildStyle(basemaps, current) {
    var sources = {}, base = [], labels = [];
    basemaps.forEach(function (bm) {
      bm.layers.forEach(function (layer, i) {
        var id = "bm-" + bm.id + "-" + i;
        sources[id] = {
          type: "raster", tiles: layer.tiles, tileSize: 256, attribution: layer.attribution || "",
          maxzoom: layer.maxzoom || 18,
        };
        (layer.labels ? labels : base).push({
          id: id, type: "raster", source: id,
          layout: { visibility: bm.id === current ? "visible" : "none" },
          paint: { "raster-fade-duration": 120 },
        });
      });
    });
    var bg = basemapDef(current);
    // deck.gl draws the events just below the first place-name layer, so labels stay readable
    S.labelBefore = labels.length ? labels[0].id : undefined;
    return {
      version: 8, sources: sources,
      layers: [{ id: "background", type: "background", paint: { "background-color": (bg && bg.background) || "#e6e6e3" } }]
        .concat(base, labels),
    };
  }

  function applyBasemap(id) {
    if (!S.ready || id === S.basemap) return;
    S.basemap = id;
    var map = S.map;
    map.getStyle().layers.forEach(function (layer) {
      if (layer.id.indexOf("bm-") !== 0) return;
      var on = layer.id.indexOf("bm-" + id + "-") === 0;
      map.setLayoutProperty(layer.id, "visibility", on ? "visible" : "none");
    });
    var def = basemapDef(id);
    map.setPaintProperty("background", "background-color", (def && def.background) || "#e6e6e3");
  }

  function FitControl() {}
  FitControl.prototype.onAdd = function () {
    var box = el("div", { className: "maplibregl-ctrl maplibregl-ctrl-group" });
    var btn = el("button", {
      type: "button", className: "fit-btn", title: "Zoom to the matching events",
      "aria-label": "Zoom to the matching events",
      onclick: function () { var f = S.args.fit; if (f) fitTo(f.bounds, f.max_zoom, 700); },
    });
    // A frame around a few dots: "zoom to the events"
    btn.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="1.5" y="1.5" width="13" height="13" rx="2.5" ' +
      'fill="none" stroke="currentColor" stroke-width="1.4"/><circle cx="5.5" cy="9.5" r="1.6"/><circle cx="9.8" cy="5.8" r="1.6"/>' +
      '<circle cx="10.5" cy="10.5" r="1.6"/></svg>';
    box.appendChild(btn);
    return box;
  };
  FitControl.prototype.onRemove = function () {};

  function createMap(a) {
    var map;
    try {
      map = new maplibregl.Map({
        container: ui.map,
        style: buildStyle(a.basemaps || [], a.basemap),
        center: [12, 22],
        zoom: worldZoom(),
        minZoom: 0.5, maxZoom: 18, maxPitch: 0,
        dragRotate: false, pitchWithRotate: false, touchPitch: false,
        attributionControl: { compact: false },
        fadeDuration: 0,
      });
    } catch (err) {
      showStatus("This browser could not start the map (it needs WebGL 2). " + (err.message || ""), true);
      S.failed = true;
      return;
    }
    S.map = map;
    S.basemap = a.basemap;
    map.touchZoomRotate.disableRotation();
    map.keyboard.disableRotation();
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 110 }), "bottom-left");
    map.addControl(new FitControl(), "bottom-right");
    map.addControl(new maplibregl.FullscreenControl({ container: document.body }), "bottom-right");
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");

    // Start drawing events as soon as the style is parsed. (MapLibre's "load" event also waits for
    // every basemap tile, so a slow tile server would hold back the events.)
    function start() {
      if (S.overlay) return;
      FILTER = new deck.DataFilterExtension({ filterSize: 2 });
      S.overlay = new deck.MapboxOverlay({ interleaved: true, layers: [], onHover: onHover, pickingRadius: 5 });
      map.addControl(S.overlay);
      S.ready = true;
      S.pointScale = pointScaleFor(map.getZoom());
      refresh();
    }
    map.once("style.load", start);
    map.once("load", start);
    map.on("zoom", function () {
      var s = pointScaleFor(map.getZoom());
      if (Math.abs(s - S.pointScale) > 0.08 * S.pointScale) { S.pointScale = s; requestLayers(); }
    });
    map.on("zoomend", function () {  // settle on the exact size once the zoom stops
      var s = pointScaleFor(map.getZoom());
      if (s !== S.pointScale) { S.pointScale = s; requestLayers(); }
    });
    map.on("movestart", hideTooltip);
    map.on("click", onMapClick);
    map.getCanvasContainer().addEventListener("mouseleave", hideTooltip);
  }

  // The zoom at which the whole world just fits the map's width
  function worldZoom() { return clamp(Math.log2(Math.max(ui.map.clientWidth, 300) / 512), 0.5, 2.2); }

  function pointScaleFor(zoom) { return clamp(1.8 + 0.55 * (zoom - 1.5), 1.8, 6.5); }
  // Largest fanned-out cluster (in dots) at the current zoom: doubles every 2 zoom levels
  function stackCap() { return clamp(STACK_MIN * Math.pow(2, (S.pointScale - 1.8) / 1.1), STACK_MIN, STACK_MAX); }

  // ── Events file ────────────────────────────────────────────────────────────
  function loadData(url, rawUrl) {
    S.dataUrl = url;
    S.data = null;
    S.eventsData = S.heatData = null;
    S.maskId = S.colorKey = S.radiusKey = S.heatKey = S.groupCountKey = null;
    var gz = typeof DecompressionStream !== "undefined";
    var target = gz ? url : rawUrl || url;
    showStatus("Loading events…");
    fetch(target).then(function (res) {
      if (!res.ok) throw new Error("HTTP " + res.status + " for " + target);
      var total = Number(res.headers.get("Content-Length")) || 0, seen = 0;
      var stream = res.body;
      if (total && typeof TransformStream !== "undefined") {
        stream = stream.pipeThrough(new TransformStream({
          transform: function (chunk, ctl) {
            seen += chunk.byteLength;
            showStatus("Loading events… " + Math.min(99, Math.round(100 * seen / total)) + "%");
            ctl.enqueue(chunk);
          },
        }));
      }
      if (gz && target === url && /\.gz$/.test(url)) stream = stream.pipeThrough(new DecompressionStream("gzip"));
      return new Response(stream).arrayBuffer();
    }).then(function (buf) {
      if (S.dataUrl !== url) return;  // a newer file was requested meanwhile
      S.data = decode(buf);
      prepareStatic();
      showStatus("");
      refresh();
    }).catch(function (err) {
      if (S.dataUrl !== url) return;
      showStatus("The events could not be loaded (" + err.message + "). Reload the page to try again.", true);
    });
  }

  function decode(buf) {
    var bytes = new Uint8Array(buf, 0, 8);
    if (String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !== "GTD1") throw new Error("unexpected file format");
    var headLen = new DataView(buf).getUint32(4, true);
    var head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, headLen)));
    var start = 8 + headLen;
    var types = { f4: Float32Array, u1: Uint8Array, u2: Uint16Array, u4: Uint32Array };
    var d = { n: head.n, dicts: head.dicts, casualties: head.casualties !== false };
    head.columns.forEach(function (c) { d[c[0]] = new types[c[1]](buf, start + c[2], head.n); });
    var lo = 9999, hi = 0;
    for (var i = 0; i < d.n; i++) { var y = d.year[i]; if (y < lo) lo = y; if (y > hi) hi = y; }
    d.minYear = lo; d.maxYear = hi;
    return d;
  }

  function prepareStatic() {
    var d = S.data, n = d.n;
    var pos = new Float32Array(n * 2);
    var filt = new Float32Array(n * 2);  // [matches filters (0/1), year] per event
    for (var i = 0; i < n; i++) {
      pos[2 * i] = d.lon[i];
      pos[2 * i + 1] = d.lat[i];
      filt[2 * i + 1] = d.year[i];
    }
    S.attr.position = { value: pos, size: 2 };
    S.filterValues = filt;
    S.attr.filter = { value: filt, size: 2 };
    // Which events share exactly the same point (compared bit for bit)
    var lonBits = new Uint32Array(d.lon.buffer, d.lon.byteOffset, n);
    var latBits = new Uint32Array(d.lat.buffer, d.lat.byteOffset, n);
    var ids = new Map(), stack = new Uint32Array(n), count = 0;
    for (i = 0; i < n; i++) {
      var key = lonBits[i] + ":" + latBits[i], id = ids.get(key);
      if (id === undefined) { id = count++; ids.set(key, id); }
      stack[i] = id;
    }
    d.stack = stack;
    S.stackSize = new Uint32Array(count);  // events of each stack currently shown
    S.stackRank = new Uint32Array(n);
    S.offsets = new Float32Array(n * 2);
    S.layoutKey = null;
  }

  // Pixel offsets that fan out events sharing a point. Recomputed when the shown events change
  // (filters, time-lapse year) or the dot size changes with the zoom.
  function layoutStacks() {
    var key = S.maskId + "|" + S.year + "|" + S.pointScale.toFixed(2);
    if (key === S.layoutKey) return;
    S.layoutKey = key;
    var d = S.data, n = d.n, f = S.filterValues, stack = d.stack, size = S.stackSize, rank = S.stackRank;
    var off = S.offsets, lo = S.year === null ? -1 : S.year - 1, hi = S.year === null ? 1e5 : S.year;
    size.fill(0);
    for (var i = 0; i < n; i++) {
      if (f[2 * i] === 1 && d.year[i] >= lo && d.year[i] <= hi) rank[i] = size[stack[i]]++;
    }
    var c = (1.9 * S.pointScale) / 1.77;  // neighbouring dots almost touch
    var cap = stackCap();
    for (i = 0; i < n; i++) {
      var m = size[stack[i]];
      if (m < 2 || f[2 * i] !== 1 || d.year[i] < lo || d.year[i] > hi) { off[2 * i] = 0; off[2 * i + 1] = 0; continue; }
      var k = rank[i], r = c * Math.sqrt(m > cap ? k * cap / m : k), t = k * GOLDEN_ANGLE;
      off[2 * i] = r * Math.cos(t);
      off[2 * i + 1] = r * Math.sin(t);
    }
    S.attr.offset = { value: off, size: 2 };  // a new descriptor makes deck.gl upload the new values
    S.eventsData = null;
  }

  // ── Applying Python's state ────────────────────────────────────────────────
  function refresh() {
    var a = S.args;
    if (!a || !S.ready || !S.data || S.failed) {
      renderBanner();
      return;
    }
    applyBasemap(a.basemap);
    if (a.mask && a.mask_id !== S.maskId) applyMask(a.mask, a.mask_id);
    if (S.year !== null && (S.year < a.year_range[0] || S.year > a.year_range[1])) clearYear();
    updateColors();
    updateRadii();
    syncSelection();
    buildLayers();
    renderLegend();
    renderTimeline();
    renderCard();
    renderBanner();
    applyFit();
  }

  function applyMask(bits, id) {
    var d = S.data, n = d.n, f = S.filterValues;
    if (!bits || bits.length * 8 < n) return;  // stale mask for another data version
    var yc = new Uint32Array(d.maxYear - d.minYear + 1), count = 0;
    for (var i = 0; i < n; i++) {
      var on = (bits[i >> 3] >> (i & 7)) & 1;
      f[2 * i] = on;
      if (on) { count++; yc[d.year[i] - d.minYear]++; }
    }
    S.attr.filter = { value: f, size: 2 };  // a new descriptor makes deck.gl upload the new values
    S.maskId = id;
    S.matching = count;
    S.yearCounts = yc;
    S.eventsData = null;
    S.heatKey = S.groupCountKey = null;
  }

  function updateColors() {
    var a = S.args, dark = isDark();
    var key = a.color_by + "|" + dark + "|" + a.dot_color;
    if (key === S.colorKey) return;
    S.colorKey = key;
    var dot = rgb((dark ? a.dot_color_dark : a.dot_color) || "#c8384b");
    S.constColor = [dot[0], dot[1], dot[2], 225];
    var groups = a.color_by && a.groups ? a.groups[a.color_by] : null;
    if (!groups) {
      S.attr.color = null;
    } else {
      var d = S.data, n = d.n, codes = d[CODES[a.color_by]];
      var pal = (dark ? PALETTE.dark : PALETTE.light).map(rgb).concat([rgb(OTHER)]);
      var lut = groups.byCode.map(function (g) { return pal[g]; });
      var other = pal[pal.length - 1];
      var out = new Uint8Array(n * 4);
      for (var i = 0; i < n; i++) {
        var c = lut[codes[i]] || other;
        out[4 * i] = c[0]; out[4 * i + 1] = c[1]; out[4 * i + 2] = c[2]; out[4 * i + 3] = 225;
      }
      S.attr.color = { value: out, size: 4 };
    }
    S.eventsData = null;
    S.groupCountKey = null;
  }

  // Dot radius as a multiple of the base size: area grows with the death toll
  function deathRadius(k) { return k === UNKNOWN || k === 0 ? 1 : Math.min(1 + 0.45 * Math.sqrt(k), 18); }

  function updateRadii() {
    var key = S.args.scale ? "deaths" : "uniform";
    if (key === S.radiusKey) return;
    S.radiusKey = key;
    if (!S.args.scale) {
      S.attr.radius = null;
    } else {
      var k = S.data.killed, n = S.data.n, r = new Float32Array(n);
      for (var i = 0; i < n; i++) r[i] = deathRadius(k[i]);
      S.attr.radius = { value: r, size: 1 };
    }
    S.eventsData = null;
    S.heatKey = null;
  }

  function syncSelection() {
    var a = S.args, sel = typeof a.selected === "number" ? a.selected : -1;
    // After a click here, wait until Python has seen it before following Python's selection again
    if (S.out.selectSeq && a.select_seq !== S.out.selectSeq) return;
    if (sel >= S.data.n) sel = -1;
    S.selected = sel;
  }

  // ── Layers ─────────────────────────────────────────────────────────────────
  function eventsData() {
    if (!S.eventsData) {
      var attributes = { getPosition: S.attr.position, getFilterValue: S.attr.filter, getPixelOffset: S.attr.offset };
      if (S.attr.color) attributes.getFillColor = S.attr.color;
      if (S.attr.radius) attributes.getRadius = S.attr.radius;
      S.eventsData = { length: S.data.n, attributes: attributes };
    }
    return S.eventsData;
  }

  function heatData() {
    var key = S.maskId + "|" + S.year + "|" + (S.args.scale ? 1 : 0);
    if (key !== S.heatKey) {
      S.heatKey = key;
      var d = S.data, f = S.filterValues, n = d.n, m = 0;
      var weighted = !!S.args.scale;
      var pos = new Float32Array(S.matching * 2), w = weighted ? new Float32Array(S.matching) : null;
      for (var i = 0; i < n; i++) {
        if (f[2 * i] !== 1 || (S.year !== null && d.year[i] !== S.year)) continue;
        pos[2 * m] = d.lon[i];
        pos[2 * m + 1] = d.lat[i];
        if (w) { var k = d.killed[i]; w[m] = k === UNKNOWN ? 0 : k; }
        m++;
      }
      var attributes = { getPosition: { value: pos.subarray(0, 2 * m), size: 2 } };
      if (w) attributes.getWeight = { value: w.subarray(0, m), size: 1 };
      S.heatData = { length: m, attributes: attributes };
    }
    return S.heatData;
  }

  function circlePolygon(lat, lon, km) {
    var R = 6371.0088, dist = km / R, p1 = lat * Math.PI / 180, l1 = lon * Math.PI / 180, pts = [];
    for (var i = 0; i <= 96; i++) {
      var b = 2 * Math.PI * i / 96;
      var p2 = Math.asin(Math.sin(p1) * Math.cos(dist) + Math.cos(p1) * Math.sin(dist) * Math.cos(b));
      var l2 = l1 + Math.atan2(Math.sin(b) * Math.sin(dist) * Math.cos(p1), Math.cos(dist) - Math.sin(p1) * Math.sin(p2));
      var x = l2 * 180 / Math.PI;
      if (pts.length) {  // keep longitudes continuous across the date line
        var prev = pts[pts.length - 1][0];
        while (x - prev > 180) x -= 360;
        while (x - prev < -180) x += 360;
      }
      pts.push([x, p2 * 180 / Math.PI]);
    }
    return pts;
  }

  function requestLayers() {
    if (S.frame) return;
    S.frame = requestAnimationFrame(function () { S.frame = 0; buildLayers(); });
  }

  function buildLayers() {
    if (!S.overlay || !S.data) return;
    var a = S.args, dark = isDark(), layers = [];
    var years = S.year === null ? [-1, 1e5] : [S.year - 1.5, S.year + 0.5];
    if (a.mode === "heatmap") {
      layers.push(new deck.HeatmapLayer({
        id: "heat", data: heatData(), beforeId: S.labelBefore,
        radiusPixels: 22, intensity: 1, threshold: 0.03, aggregation: "SUM", debounceTimeout: 100,
        colorRange: (dark ? HEAT.dark : HEAT.light).map(rgb),
      }));
    } else {
      layoutStacks();
      layers.push(new deck.ScatterplotLayer({
        id: "events", data: eventsData(), beforeId: S.labelBefore,
        getFillColor: S.constColor, getRadius: 1,
        radiusUnits: "pixels", radiusScale: S.pointScale, radiusMinPixels: 1.2, radiusMaxPixels: 80,
        stroked: true, lineWidthUnits: "pixels", getLineWidth: 0.6,
        getLineColor: dark ? [12, 12, 12, 150] : [255, 255, 255, 170],
        opacity: a.scale ? 0.7 : 1,  // overlapping bubbles stay readable
        pickable: !a.picking, autoHighlight: true,
        highlightColor: dark ? [255, 255, 255, 230] : [20, 20, 20, 210],
        extensions: [FILTER],
        filterRange: [[0.5, 1.5], years],
        filterSoftRange: S.year === null ? null : [[0.5, 1.5], [S.year - 0.5, S.year + 0.5]],
        updateTriggers: { getFillColor: S.constColor.join(","), getLineColor: dark },
      }));
    }
    if (a.circle) {
      var key = a.circle.lat + "," + a.circle.lon + "," + a.circle.radius_km;
      if (key !== S.areaKey) {
        S.areaKey = key;
        S.area = [{ polygon: circlePolygon(a.circle.lat, a.circle.lon, a.circle.radius_km), centre: [a.circle.lon, a.circle.lat] }];
      }
      var blue = rgb(a.area_color || "#1f5fbf");
      layers.push(new deck.PolygonLayer({
        id: "area", data: S.area, getPolygon: function (o) { return o.polygon; },
        filled: true, stroked: true, getFillColor: blue.concat([18]), getLineColor: blue.concat([235]),
        lineWidthUnits: "pixels", getLineWidth: 2, pickable: false,
      }));
      layers.push(new deck.ScatterplotLayer({
        id: "centre", data: S.area, getPosition: function (o) { return o.centre; },
        radiusUnits: "pixels", getRadius: 4.5, getFillColor: blue.concat([255]),
        stroked: true, lineWidthUnits: "pixels", getLineWidth: 1.5, getLineColor: [255, 255, 255, 255],
      }));
    }
    if (a.area && a.area[2] - a.area[0] < 360) {
      var key2 = a.area.join(",");
      if (key2 !== S.veilKey) {
        S.veilKey = key2;
        var w = a.area[0], so = a.area[1], e = a.area[2], no = a.area[3];
        var hole = [[w, so], [e, so], [e, no], [w, no], [w, so]];
        S.veil = [{ ring: hole, polygon: [[[-540, -86], [540, -86], [540, 86], [-540, 86], [-540, -86]], hole] }];
      }
      var blueA = rgb(a.area_color || "#1f5fbf");
      layers.push(new deck.PolygonLayer({
        id: "area-veil", data: S.veil, getPolygon: function (o) { return o.polygon; }, beforeId: S.labelBefore,
        filled: true, stroked: false, getFillColor: dark ? [0, 0, 0, 115] : [255, 255, 255, 140], pickable: false,
        updateTriggers: { getFillColor: dark },
      }));
      layers.push(new deck.PathLayer({
        id: "area-edge", data: S.veil, getPath: function (o) { return o.ring; },
        getColor: blueA.concat([230]), widthUnits: "pixels", getWidth: 2, pickable: false,
      }));
    }
    if (S.selected >= 0 && a.mode !== "heatmap") {
      var i = S.selected, d = S.data;
      var r = (a.scale ? deathRadius(d.killed[i]) : 1) * S.pointScale + 5;
      var shift = [S.offsets[2 * i], S.offsets[2 * i + 1]];
      layers.push(new deck.ScatterplotLayer({
        id: "selected", data: [[d.lon[i], d.lat[i]]], getPosition: function (p) { return p; },
        getPixelOffset: shift,
        radiusUnits: "pixels", getRadius: r, filled: false, stroked: true,
        lineWidthUnits: "pixels", getLineWidth: 2.5,
        getLineColor: dark ? [255, 255, 255, 255] : [17, 17, 17, 255],
        updateTriggers: { getRadius: r, getLineColor: dark, getPixelOffset: shift.join(",") },
      }));
    }
    S.overlay.setProps({ layers: layers });
  }

  // ── Interaction ────────────────────────────────────────────────────────────
  function pushValue() {
    var text = JSON.stringify(S.out);
    if (text === S.lastSent) return;
    S.lastSent = text;
    Streamlit.setValue(JSON.parse(text));
  }

  // [west, south, east, north] of what the map shows (west/east can pass ±180 when zoomed out)
  function currentBounds() {
    var b = S.map.getBounds();
    return [round(b.getWest(), 4), round(clamp(b.getSouth(), -90, 90), 4),
      round(b.getEast(), 4), round(clamp(b.getNorth(), -90, 90), 4)];
  }

  function selectEvent(i) {
    S.selected = i;
    S.out.selected = i;
    S.out.selectSeq = nowSeq();
    buildLayers();
    renderCard();
    pushValue();
  }

  function sendAction(action) {
    S.out.action = action;
    S.out.actionSeq = nowSeq();
    pushValue();
  }

  function onMapClick(e) {
    var a = S.args;
    if (!a || !S.data) return;
    if (a.picking) {
      S.out.pick = [round(clamp(e.lngLat.lat, -90, 90), 4), round(wrapLon(e.lngLat.lng), 4)];
      S.out.pickSeq = nowSeq();
      pushValue();
      return;
    }
    if (a.mode === "heatmap" || !S.overlay) return;
    var info = S.overlay.pickObject({ x: e.point.x, y: e.point.y, radius: 5, layerIds: ["events"] });
    if (info && info.index >= 0) selectEvent(info.index);
    else if (S.selected >= 0) selectEvent(-1);
  }

  function onHover(info) {
    if (!info || !info.layer || info.layer.id !== "events" || info.index < 0 || !S.data || S.args.picking) {
      hideTooltip();
      return;
    }
    var d = S.data, i = info.index;
    var place = [known(label(d.dicts.city, d.city[i])), label(d.dicts.country, d.country[i])].filter(Boolean).join(", ");
    var group = known(label(d.dicts.group, d.group[i]));
    var tip = ui.tooltip, rows = [
      el("div", { className: "place", text: place }),
      el("div", { className: "muted", text: dateText(d.year[i], d.month[i], d.day[i]) }),
    ];
    // Only the fields this dataset has (a trimmed parquet may lack some)
    if (d.dicts.attack.length) rows.push(el("div", { text: label(d.dicts.attack, d.attack[i]) || "Unknown attack type" }));
    if (d.dicts.group.length) rows.push(el("div", { className: "muted", text: group ? "By " + group : "Perpetrator unknown" }));
    if (d.casualties) rows.push(el("div", { className: "toll", text: tollText(d.killed[i], d.wounded[i]) }));
    var shared = S.stackSize ? S.stackSize[d.stack[i]] : 1;
    if (shared > 1) rows.push(el("div", { className: "muted stack", text: "One of " + NF.format(shared) + " events recorded at this point" }));
    tip.replaceChildren.apply(tip, rows);
    tip.hidden = false;
    placeTooltip(info.x, info.y);
  }

  function placeTooltip(x, y) {
    var tip = ui.tooltip, w = tip.offsetWidth, h = tip.offsetHeight;
    var W = document.body.clientWidth, H = document.body.clientHeight;
    var left = x + 14 + w > W - 6 ? x - 14 - w : x + 14;
    var top = y + 14 + h > H - 6 ? y - 14 - h : y + 14;
    tip.style.left = clamp(left, 4, W - w - 4) + "px";
    tip.style.top = clamp(top, 4, H - h - 4) + "px";
  }
  function hideTooltip() { ui.tooltip.hidden = true; }

  function fitTo(b, maxZoom, duration) {
    if (!b || !S.map) return;
    var cardOpen = !ui.card.hidden && document.body.clientWidth > 640;
    S.map.fitBounds([[b[0], b[1]], [b[2], b[3]]], {
      padding: { top: 40, bottom: 70, left: 40, right: cardOpen ? 360 : 50 },
      maxZoom: maxZoom || 8, duration: duration || 0,
    });
  }

  function applyFit() {
    var fit = S.args.fit;
    if (!fit || fit.id === S.fitId) return;
    var first = S.fitId === null;
    S.fitId = fit.id;
    // On first load keep the starting view, unless the filters (e.g. from a shared link) ask otherwise.
    // Later, fly to new results, except for "Map area", which was just taken from the map's own view.
    if (first ? !fit.initial : fit.fly === false) return;
    fitTo(fit.bounds, fit.max_zoom, first ? 0 : 800);
  }

  // ── Legend ─────────────────────────────────────────────────────────────────
  function groupCounts() {
    var a = S.args, key = S.maskId + "|" + a.color_by + "|" + S.year;
    if (key !== S.groupCountKey) {
      S.groupCountKey = key;
      var d = S.data, f = S.filterValues, codes = d[CODES[a.color_by]], lut = a.groups[a.color_by].byCode;
      var counts = [0, 0, 0, 0];
      for (var i = 0; i < d.n; i++) {
        if (f[2 * i] !== 1 || (S.year !== null && d.year[i] !== S.year)) continue;
        var g = lut[codes[i]];
        counts[g === undefined ? 3 : g]++;
      }
      S.groupCounts = counts;
    }
    return S.groupCounts;
  }

  function shownCount() {
    if (S.year === null) return S.matching;
    return S.yearCounts ? S.yearCounts[S.year - S.data.minYear] || 0 : 0;
  }

  function renderLegend() {
    var a = S.args, dark = isDark(), box = ui.legend, n = shownCount();
    var head = S.year === null ? NF.format(n) + (n === 1 ? " event" : " events")
      : NF.format(n) + (n === 1 ? " event in " : " events in ") + S.year;
    var parts = [el("div", { className: "count", text: n ? head : (S.year === null ? "No events match the filters" : "No events in " + S.year) })];
    if (!n) {
      // nothing else to explain
    } else if (a.mode === "heatmap") {
      parts.push(el("div", { className: "title", text: a.scale ? "Density of deaths" : "Density of events" }));
      var ramp = (dark ? HEAT.dark : HEAT.light);
      parts.push(el("div", { className: "ramp", style: "background: linear-gradient(90deg," + ramp.join(",") + ")" }));
      parts.push(el("div", { className: "ramp-labels" }, el("span", { text: "Fewer" }), el("span", { text: "More" })));
    } else if (a.color_by && a.color_by !== "none" && a.groups && a.groups[a.color_by]) {
      var g = a.groups[a.color_by], counts = groupCounts();
      var pal = (dark ? PALETTE.dark : PALETTE.light).concat([OTHER]);
      parts.push(el("div", { className: "title", text: g.title }));
      var list = el("ul");
      g.labels.forEach(function (name, k) {
        list.appendChild(el("li", null, el("span", { className: "sw", style: "background:" + pal[k] }),
          el("span", { text: name }), el("span", { className: "n", text: NF.format(counts[k]) })));
      });
      parts.push(list);
    }
    if (n && a.mode !== "heatmap" && a.scale) {
      parts.push(el("div", { className: "title", text: "Deaths" }));
      var sizes = el("div", { className: "sizes" });
      [1, 10, 100, 1000].forEach(function (k) {
        var dpx = Math.max(3, 2 * deathRadius(k) * S.pointScale);
        sizes.appendChild(el("span", null, el("i", { style: "width:" + dpx + "px;height:" + dpx + "px" }), NF.format(k)));
      });
      parts.push(sizes);
    }
    // "Filter to this view": turn what the map shows into the "Map area" filter (and back)
    var tools = el("div", { className: "area-tools" });
    if (a.area_on) {
      tools.appendChild(el("span", { className: "note", text: a.area ? "Filtered to the outlined area" : "Reading the map view…" }));
      tools.appendChild(el("button", {
        type: "button", className: "text-btn", text: "Update to this view",
        title: "Use the area the map shows now", onclick: function () { sendAction({ type: "area", bounds: currentBounds() }); },
      }));
      tools.appendChild(el("button", {
        type: "button", className: "text-btn", text: "Clear",
        title: "Remove the area filter", onclick: function () { sendAction({ type: "area_clear" }); },
      }));
    } else {
      tools.appendChild(el("button", {
        type: "button", className: "text-btn", text: "Filter to this view",
        title: "Show only the events inside the visible map area. The numbers, tables and charts follow.",
        onclick: function () { sendAction({ type: "area", bounds: currentBounds() }); },
      }));
    }
    parts.push(tools);
    box.replaceChildren.apply(box, parts);
    box.hidden = false;
  }

  function renderBanner() {
    var picking = !!(S.args && S.args.picking);
    ui.banner.hidden = !picking;
    ui.banner.textContent = picking ? "Click the map to set the centre of the search area" : "";
    ui.map.classList.toggle("picking", picking);
    if (picking) hideTooltip();
  }

  // ── Event card ─────────────────────────────────────────────────────────────
  function renderCard() {
    var i = S.selected, card = ui.card;
    if (i < 0 || !S.data || S.args.mode === "heatmap") { card.hidden = true; return; }
    var d = S.data, a = S.args;
    var det = a.detail && a.detail.index === i ? a.detail : null;
    var place = det ? det.place : [known(label(d.dicts.city, d.city[i])), label(d.dicts.country, d.country[i])].filter(Boolean).join(", ");
    var facts = el("dl");
    function fact(name, value, note, className) {
      if (!value) return;
      facts.appendChild(el("dt", { text: name }));
      facts.appendChild(el("dd", className ? { className: className } : null, value,
        note ? el("div", { className: "note", text: note }) : null));
    }
    if (det && det.id) fact("GTD ID", det.id, null, "id");
    fact("Attack", (det && det.attack) || label(d.dicts.attack, d.attack[i]));
    var target = det ? [det.target_type, known(det.target)].filter(Boolean).join(": ") : label(d.dicts.target, d.target[i]);
    fact("Target", target);
    fact("Weapon", (det && det.weapon) || label(d.dicts.weapon, d.weapon[i]));
    if (d.dicts.group.length) {
      var n = det && det.perpetrators;
      fact("Perpetrator", ((det && det.group) || label(d.dicts.group, d.group[i]) || "Unknown") +
        (n ? " (" + NF.format(n) + (n === 1 ? " perpetrator)" : " perpetrators)") : ""));
    }
    if (det && det.hostages) fact("Hostages", NF.format(det.hostages));
    if (det && det.damage) fact("Damage", damageText(det.damage), det.damage.any === false ? null : det.damage.comment);

    var tags = el("div", { className: "tags" });
    if (det) {
      if (det.suicide) tags.appendChild(el("span", { className: "tag", text: "Suicide attack" }));
      if (det.success === false) tags.appendChild(el("span", { className: "tag", text: "Unsuccessful" }));
      if (det.claimed) tags.appendChild(el("span", { className: "tag", text: "Responsibility claimed" }));
      if (det.multiple) tags.appendChild(el("span", { className: "tag", text: "Part of a series" }));
      if (det.doubt) tags.appendChild(el("span", { className: "tag", text: "Doubt it was terrorism" }));
    }

    var text;
    if (!det) text = el("p", { className: "text pending", text: "Loading the summary…" });
    else if (det.summary) text = el("p", { className: "text", text: det.summary });
    else text = el("p", { className: "text pending", text: "The database has no written summary for this event." });

    var motive = det && det.motive && !/^(unknown|the specific motive for the attack is unknown\.?)$/i.test(det.motive.trim())
      ? el("div", null, el("div", { className: "label", text: "Motive" }), el("p", { className: "text", text: det.motive }))
      : null;

    var lat = d.lat[i], lon = d.lon[i];
    var actions = el("div", { className: "actions" },
      el("button", {
        type: "button", className: "text-btn", text: "Zoom in",
        onclick: function () { S.map.flyTo({ center: [lon, lat], zoom: Math.max(S.map.getZoom() + 2, 9), duration: 900 }); },
      }),
      el("button", {
        type: "button", className: "text-btn", text: "Search around here",
        title: "Show only events within the search radius of this one",
        onclick: function () { sendAction({ type: "around", lat: round(lat, 4), lon: round(lon, 4) }); },
      })
    );

    var parts = [
      el("div", { className: "head" },
        el("div", null, el("h2", { text: place || "Unknown place" }), el("div", { className: "date", text: dateText(d.year[i], d.month[i], d.day[i]) })),
        el("button", { type: "button", className: "icon-btn close", "aria-label": "Close", title: "Close", text: "×", onclick: function () { selectEvent(-1); } })
      ),
      d.casualties ? el("div", { className: "toll" }, el("strong", { text: tollText(d.killed[i], d.wounded[i]) })) : null,
      facts,
      tags.childNodes.length ? tags : null,
      text,
      motive,
      actions
    ].filter(Boolean);  // replaceChildren would print "null" for missing parts
    card.replaceChildren.apply(card, parts);
    card.hidden = false;
  }

  // Property damage: "Minor (likely < $1 million), about $20,000", "None", "Yes, extent unknown" or "Unknown"
  function damageText(dm) {
    if (dm.any === false) return "None";
    var extent = dm.extent && dm.extent !== "Unknown" ? dm.extent : null;
    var text = extent || (dm.any ? "Yes, extent unknown" : "Unknown");
    return dm.value ? text + ", about $" + NF.format(dm.value) : text;
  }

  // ── Timeline and time-lapse ────────────────────────────────────────────────
  function renderTimeline() {
    var a = S.args, d = S.data, yc = S.yearCounts;
    if (!yc) { ui.timeline.hidden = true; return; }
    ui.timeline.hidden = false;
    var y0 = a.year_range[0], y1 = a.year_range[1];
    ui.play.disabled = y0 === y1;
    ui.timeline.classList.toggle("playing", S.playing);
    ui.clear.hidden = S.year === null;
    if (S.year === null) {
      ui.label.replaceChildren(el("b", { text: y0 === y1 ? String(y0) : y0 + "–" + y1 }), el("span", { text: "All years" }));
    } else {
      ui.label.replaceChildren(el("b", { text: String(S.year) }), el("span", { text: NF.format(shownCount()) + " events" }));
    }
    // Bars
    var canvas = ui.chart, dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    var max = 1, nYears = y1 - y0 + 1;
    for (var y = y0; y <= y1; y++) max = Math.max(max, yc[y - d.minYear] || 0);
    var slot = w / nYears, bar = Math.max(1, Math.min(slot - 1.5, 14));
    var muted = cssVar("--muted") || "#888", accent = cssVar("--accent") || "#c8384b";
    for (y = y0; y <= y1; y++) {
      var c = (y >= d.minYear && y <= d.maxYear) ? yc[y - d.minYear] : 0;
      var x = (y - y0) * slot + (slot - bar) / 2;
      var bh = c ? Math.max(2, (h - 12) * c / max) : 0;
      ctx.globalAlpha = S.year === null || y === S.year ? 1 : (y < S.year ? 0.55 : 0.3);
      ctx.fillStyle = y === S.year ? accent : muted;
      if (bh) ctx.fillRect(x, h - 10 - bh, bar, bh);
    }
    ctx.globalAlpha = 1;
    // Axis labels: first, last and round decades
    ctx.fillStyle = muted;
    ctx.font = "10px " + (cssVar("--font") || "sans-serif");
    ctx.textBaseline = "bottom";
    var step = nYears > 30 ? 10 : nYears > 12 ? 5 : nYears > 5 ? 2 : 1;
    for (y = y0; y <= y1; y++) {
      if (y !== y0 && y !== y1 && y % step) continue;
      var tx = (y - y0) * slot + slot / 2;
      if (y !== y0 && y !== y1 && (tx < 26 || tx > w - 26)) continue;
      ctx.textAlign = y === y0 ? "left" : y === y1 ? "right" : "center";
      ctx.fillText(String(y), y === y0 ? 0 : y === y1 ? w : tx, h);
    }
  }

  function yearAt(clientX) {
    var a = S.args, r = ui.chart.getBoundingClientRect();
    var n = a.year_range[1] - a.year_range[0] + 1;
    return clamp(a.year_range[0] + Math.floor((clientX - r.left) / (r.width / n)), a.year_range[0], a.year_range[1]);
  }

  function setYear(y) {
    S.year = y;
    S.heatKey = S.groupCountKey = null;
    buildLayers();
    renderTimeline();
    renderLegend();
  }
  function clearYear() { stopPlay(); S.year = null; S.heatKey = S.groupCountKey = null; }

  function stopPlay() {
    S.playing = false;
    clearInterval(S.timer);
    S.timer = null;
    ui.timeline.classList.remove("playing");
    ui.play.setAttribute("aria-label", "Play through the years");
  }

  function step() {
    var y1 = S.args.year_range[1];
    if (S.year === null || S.year >= y1) { stopPlay(); renderTimeline(); return; }
    setYear(S.year + 1);
    if (S.year >= y1) { stopPlay(); renderTimeline(); }
  }

  ui.play.addEventListener("click", function () {
    if (!S.data || !S.args) return;
    if (S.playing) { stopPlay(); renderTimeline(); return; }
    var y0 = S.args.year_range[0], y1 = S.args.year_range[1];
    S.playing = true;
    ui.play.setAttribute("aria-label", "Pause");
    if (S.year === null || S.year >= y1) setYear(y0); else renderTimeline();
    S.timer = setInterval(step, STEP_MS);
  });
  ui.clear.addEventListener("click", function () { clearYear(); buildLayers(); renderTimeline(); renderLegend(); });
  ui.chart.addEventListener("click", function (e) {
    if (!S.data) return;
    stopPlay();
    var y = yearAt(e.clientX);
    if (y === S.year) clearYear();
    else S.year = y;
    S.heatKey = S.groupCountKey = null;
    buildLayers(); renderTimeline(); renderLegend();
  });
  ui.chart.addEventListener("mousemove", function (e) {
    if (!S.data || !S.yearCounts) return;
    var y = yearAt(e.clientX), d = S.data;
    var c = y >= d.minYear && y <= d.maxYear ? S.yearCounts[y - d.minYear] : 0;
    ui.tooltip.replaceChildren(el("div", { className: "place", text: String(y) }),
      el("div", { text: c ? NF.format(c) + (c === 1 ? " event" : " events") : "No events" }),
      el("div", { className: "muted", text: y === S.year ? "Click to show all years" : "Click to show only this year" }));
    var tip = ui.tooltip, r = ui.chart.getBoundingClientRect();
    tip.hidden = false;
    tip.style.left = clamp(e.clientX - tip.offsetWidth / 2, 4, document.body.clientWidth - tip.offsetWidth - 4) + "px";
    tip.style.top = Math.max(4, r.top - tip.offsetHeight - 12) + "px";
  });
  ui.chart.addEventListener("mouseleave", hideTooltip);
  window.addEventListener("resize", function () { if (S.data && S.args) renderTimeline(); });

  // For debugging from the browser console: eventMap.map is the MapLibre map, eventMap.state the rest
  window.eventMap = { get map() { return S.map; }, get state() { return S; } };

  Streamlit.ready();
})();
