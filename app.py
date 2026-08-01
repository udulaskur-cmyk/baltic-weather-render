"""
Render.com version — Flask app that renders the 7 weather maps in a
background thread and serves them (+ status.json) over HTTP.

Deploy: see SETUP.md. Free Render web service = 512MB RAM / 0.1 CPU (shared),
so GRID_STEP/DPI here are tuned down slightly from the desktop version to
keep renders fast and light. Bump them back up if you outgrow the free tier.

Env vars (all optional, set in Render dashboard):
    MAPS_REGION       EE | LV | LT | EE_LV_LT   (default EE_LV_LT)
    UPDATE_SECONDS    seconds between renders    (default 600 = 10 min)
    PORT              set automatically by Render
"""

import os, json, threading, time, datetime, warnings
import requests, xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from flask import Flask, send_from_directory, jsonify, Response

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Config — tuned down from the desktop version for the free instance
# ---------------------------------------------------------------------

UPDATE_INTERVAL = int(os.environ.get("UPDATE_SECONDS", 600))   # 10 min default
DPI             = 150                                          # was 200 on desktop
GRID_STEP       = 0.007                                        # was 0.004 on desktop (coarser = lighter)
SIGMA           = 2.0

_EXTENT_EE   = [21.3, 28.7, 57.3, 60.05]
_EXTENT_LV   = [20.8, 28.3, 55.4, 58.2]
_EXTENT_LT   = [20.8, 26.9, 53.8, 56.5]
_EXTENT_BOTH = [20.8, 28.7, 53.8, 60.05]
_REGION_MAP = {"EE": _EXTENT_EE, "LV": _EXTENT_LV, "LT": _EXTENT_LT, "EE_LV_LT": _EXTENT_BOTH}
EXTENT = _REGION_MAP.get(os.environ.get("MAPS_REGION", "EE_LV_LT"), _EXTENT_BOTH)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "maps")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Colour tables (from your QGIS QML exports)
# ---------------------------------------------------------------------

def parse_qml(filename, vmin, vmax):
    path = os.path.join(SCRIPT_DIR, filename)
    root = ET.parse(path).getroot()
    vals, cols = [], []
    for item in root.findall(".//colorrampshader/item"):
        v = float(item.get("value"))
        h = item.get("color").lstrip("#")
        r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
        vals.append(v); cols.append((r, g, b))
    vals = np.array(vals, dtype=float)
    cols = np.array(cols, dtype=float)
    idx  = np.argsort(vals)
    vals, cols = vals[idx], cols[idx]
    pos  = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list("qml", list(zip(pos, cols)), N=2048)
    return cmap, Normalize(vmin=vmin, vmax=vmax)


def _bc(stops, v0, v1):
    pos = [max(0., min(1., (v - v0) / (v1 - v0))) for v, _ in stops]
    cm  = LinearSegmentedColormap.from_list("fb", list(zip(pos, [c for _, c in stops])), N=2048)
    return cm, Normalize(vmin=v0, vmax=v1)


def load_cmaps():
    c = {}
    try:    c["temp"] = parse_qml("temperature_color_table_high.qml", -40, 50)
    except: c["temp"] = _bc([(-40,"#ff6eff"),(-20,"#32007f"),(-10,"#259aff"),
                               (0,"#d9ecff"),(10,"#52ca0b"),(20,"#f4bd0b"),
                               (30,"#af0f14"),(45,"#c5c5c5")], -40, 45)
    try:    c["wind"] = parse_qml("wind_gust_color_table.qml", 0, 50)
    except: c["wind"] = _bc([(0,"#ffffff"),(10,"#3c96f5"),(20,"#ffa000"),
                               (30,"#e11400"),(50,"#8c645a")], 0, 50)
    c["gust"] = c["wind"]
    try:    c["pres"] = parse_qml("pressure_color_table.qml", 890, 1064)
    except: c["pres"] = _bc([(965,"#32007f"),(990,"#91ccff"),(1000,"#07a127"),
                               (1013,"#f3fb01"),(1030,"#f4520b"),(1050,"#f0a0a0")], 960, 1055)
    try:    c["prec"] = parse_qml("precipitation_color_table.qml", 0, 125)
    except: c["prec"] = _bc([(0,"#f0f0f0"),(1,"#0482ff"),(5,"#1acf05"),
                               (10,"#ff7f27"),(20,"#bf0000"),(50,"#64007f")], 0, 50)
    c["hum"] = _bc([(0,"#ffffff"),(20,"#d0eaff"),(50,"#55a3e0"),
                    (80,"#084a90"),(100,"#021860")], 0, 100)
    try:    c["dewp"] = parse_qml("temperature_color_table_high.qml", -40, 50)
    except: c["dewp"] = c["temp"]
    return c


def _dewpoint(t, rh):
    if np.isnan(t) or np.isnan(rh):
        return np.nan
    a, b = 17.625, 243.04
    g = np.log(rh / 100.0) + a * t / (b + t)
    return round(b * g / (a - g), 1)


def _parse_emhi(content):
    root = ET.fromstring(content)
    ts = root.get("timestamp", "")
    try:
        dt = datetime.datetime.fromtimestamp(int(ts), tz=datetime.timezone.utc)
        time_str = dt.strftime("%Y-%m-%d  %H:%M UTC")
    except:
        time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d  %H:%M UTC")

    def fv(s, tag):
        e = s.find(tag)
        if e is not None and e.text and e.text.strip() not in ("", "null", "None"):
            try: return float(e.text.strip())
            except: pass
        return np.nan

    stations = []
    for s in root.findall("station"):
        lat = fv(s, "latitude"); lon = fv(s, "longitude")
        if np.isnan(lat) or np.isnan(lon): continue
        ne = s.find("name")
        t  = fv(s, "airtemperature")
        rh = fv(s, "relativehumidity")
        stations.append({
            "name": ne.text.strip() if ne is not None and ne.text else "?",
            "country": "EE", "lat": lat, "lon": lon,
            "temp": t, "wind": fv(s, "windspeed"), "wind_dir": fv(s, "winddirection"),
            "gust": fv(s, "windspeedmax"), "hum": rh, "prec": fv(s, "precipitations"),
            "pres": fv(s, "airpressure"), "dewp": _dewpoint(t, rh),
        })
    return stations, time_str


_LV_STATIONS = [
    ("Rīga",56.946,24.106),("Liepāja",56.505,21.011),("Ventspils",57.401,21.544),
    ("Jelgava",56.652,23.721),("Jūrmala",56.968,23.771),("Jēkabpils",56.499,25.877),
    ("Valmiera",57.541,25.426),("Rēzekne",56.510,27.331),("Daugavpils",55.875,26.536),
    ("Kolka",57.748,22.594),("Mersrags",57.335,23.119),("Ainažu",57.865,24.355),
    ("Sigulda",57.153,24.852),("Alūksne",57.432,27.046),("Zilupe",56.386,28.128),
    ("Pāvilosta",56.887,21.193),("Saldus",56.666,22.493),("Bauska",56.410,24.193),
    ("Stende",57.165,22.532),("Priekuļi",57.321,25.352),
]

_LT_STATIONS = [
    ("Vilnius",54.687,25.279),("Kaunas",54.900,23.933),("Klaipėda",55.703,21.144),
    ("Šiauliai",55.934,23.316),("Panevėžys",55.735,24.337),("Alytus",54.396,24.046),
    ("Marijampolė",54.559,23.352),("Mažeikiai",56.309,22.341),("Jonava",55.073,24.279),
    ("Utena",55.499,25.601),("Kėdainiai",55.286,23.970),("Telšiai",55.985,22.254),
    ("Tauragė",55.250,22.289),("Ukmergė",55.245,24.771),("Visaginas",55.593,26.430),
    ("Plungė",55.911,21.846),("Kretinga",55.889,21.237),("Skuodas",56.270,21.529),
    ("Jurbarkas",55.074,22.767),("Lazdijai",54.234,23.517),("Biržai",56.200,24.754),
    ("Ignalina",55.342,26.162),("Druskininkai",54.017,23.967),("Šilutė",55.352,21.468),
    ("Rokiškis",55.960,25.585),
]


def _fetch_openmeteo(station_list, country_code):
    lats = ",".join(str(s[1]) for s in station_list)
    lons = ",".join(str(s[2]) for s in station_list)
    params = ("temperature_2m,wind_speed_10m,wind_direction_10m,"
              "wind_gusts_10m,relative_humidity_2m,precipitation,pressure_msl")
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={lats}&longitude={lons}"
           f"&current={params}&wind_speed_unit=ms&timeformat=unixtime")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        data = [data]
    stations = []
    for i, (name, lat, lon) in enumerate(station_list):
        if i >= len(data): break
        cur = data[i].get("current", {})
        t  = cur.get("temperature_2m")
        rh = cur.get("relative_humidity_2m")
        t  = np.nan if t  is None else t
        rh = np.nan if rh is None else rh
        stations.append({
            "name": name, "country": country_code, "lat": lat, "lon": lon,
            "temp": t,
            "wind": np.nan if cur.get("wind_speed_10m")     is None else cur["wind_speed_10m"],
            "wind_dir": np.nan if cur.get("wind_direction_10m") is None else cur["wind_direction_10m"],
            "gust": np.nan if cur.get("wind_gusts_10m")     is None else cur["wind_gusts_10m"],
            "hum": rh,
            "prec": np.nan if cur.get("precipitation")      is None else cur["precipitation"],
            "pres": np.nan if cur.get("pressure_msl")       is None else cur["pressure_msl"],
            "dewp": _dewpoint(t, rh),
        })
    return stations


def fetch():
    hdr = {"User-Agent": "Mozilla/5.0"}
    all_stations = []

    if EXTENT in (_EXTENT_EE, _EXTENT_BOTH):
        r = requests.get("https://www.ilmateenistus.ee/ilma_andmed/xml/observations.php",
                         headers=hdr, timeout=20)
        r.raise_for_status()
        ee_stations, time_str = _parse_emhi(r.content)
        print(f"  EE: {len(ee_stations)} stations  |  {time_str}")
        all_stations.extend(ee_stations)
    else:
        time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d  %H:%M UTC")

    if EXTENT in (_EXTENT_LV, _EXTENT_BOTH):
        try:
            lv = _fetch_openmeteo(_LV_STATIONS, "LV")
            print(f"  LV: {len(lv)} stations from Open-Meteo")
            all_stations.extend(lv)
        except Exception as e:
            print(f"  LV: fetch failed: {e}")

    if EXTENT in (_EXTENT_LT, _EXTENT_BOTH):
        try:
            lt = _fetch_openmeteo(_LT_STATIONS, "LT")
            print(f"  LT: {len(lt)} stations from Open-Meteo")
            all_stations.extend(lt)
        except Exception as e:
            print(f"  LT: fetch failed: {e}")

    print(f"  Total: {len(all_stations)} stations")
    return all_stations, time_str


def make_grid():
    lons = np.arange(EXTENT[0], EXTENT[1] + GRID_STEP, GRID_STEP)
    lats = np.arange(EXTENT[2], EXTENT[3] + GRID_STEP, GRID_STEP)
    return np.meshgrid(lons, lats)


def interpolate(stations, key, gx, gy):
    ok  = [s for s in stations if not np.isnan(s.get(key, np.nan))]
    if len(ok) < 4: return None
    pts = np.array([(s["lon"], s["lat"]) for s in ok])
    vs  = np.array([s[key] for s in ok])
    zi  = griddata(pts, vs, (gx, gy), method="linear")
    znn = griddata(pts, vs, (gx, gy), method="nearest")
    zi  = np.where(np.isnan(zi), znn, zi)
    return gaussian_filter(zi, sigma=SIGMA)


_SHPCACHE = {}

def _get_country_geoms(iso_keep_tuple):
    if iso_keep_tuple in _SHPCACHE:
        return _SHPCACHE[iso_keep_tuple]
    iso_keep = set(iso_keep_tuple)
    shp_path = shpreader.natural_earth(resolution="10m", category="cultural",
                                       name="admin_0_countries")
    reader = shpreader.Reader(shp_path)
    keep_geoms, other_geoms = [], []
    for rec in reader.records():
        iso  = rec.attributes.get("ISO_A2", "")
        iso2 = rec.attributes.get("ADM0_A3", "")
        geom = rec.geometry
        if iso in iso_keep or iso2 in iso_keep:
            keep_geoms.append(geom)
        else:
            other_geoms.append(geom)
    keep_union  = unary_union(keep_geoms)  if keep_geoms  else None
    other_union = unary_union(other_geoms) if other_geoms else None
    _SHPCACHE[iso_keep_tuple] = (keep_union, other_union)
    return keep_union, other_union


_PE  = [mpe.withStroke(linewidth=3, foreground="white")]
_PE2 = [mpe.withStroke(linewidth=4, foreground="white")]


def render_one(stations, key, cmap, norm, title, unit, fmt,
               time_str, gx, gy, outfile, wind_arrows=False, isobars=False):

    grid = interpolate(stations, key, gx, gy)
    ok   = [s for s in stations if not np.isnan(s.get(key, np.nan))]
    if ok:
        vmin_obs = min(s[key] for s in ok)
        vmax_obs = max(s[key] for s in ok)
        s_min    = min(ok, key=lambda s: s[key])
        s_max    = max(ok, key=lambda s: s[key])
    else:
        vmin_obs = vmax_obs = 0
        s_min = s_max = None

    if EXTENT == _EXTENT_EE:
        iso_keep = ("EE",); region_label = "Eesti"
        credits = "Andmeallikas: EMHI vaatlusjaamad  •  ilmateenistus.ee"
    elif EXTENT == _EXTENT_LV:
        iso_keep = ("LV",); region_label = "Läti"
        credits = "Andmeallikas: LVĢMC / Open-Meteo  •  open-meteo.com"
    elif EXTENT == _EXTENT_LT:
        iso_keep = ("LT",); region_label = "Leedu"
        credits = "Andmeallikas: LHMT / Open-Meteo  •  open-meteo.com"
    else:
        iso_keep = ("EE", "LV", "LT"); region_label = "Eesti + Läti + Leedu"
        credits = ("Andmeallikas: EMHI (ilmateenistus.ee)  •  "
                   "LVĢMC / LHMT / Open-Meteo (open-meteo.com)")

    keep_union, other_union = _get_country_geoms(iso_keep)

    fig = plt.figure(figsize=(16, 11), facecolor="white", dpi=DPI)
    ax  = fig.add_axes([0.04, 0.06, 0.82, 0.88], projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor("white")

    if grid is not None:
        masked = np.ma.masked_invalid(grid)
        ax.pcolormesh(gx, gy, masked, cmap=cmap, norm=norm,
                      shading="auto", transform=ccrs.PlateCarree(),
                      rasterized=True, zorder=2)

    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="white", zorder=3)
    if other_union is not None:
        ax.add_geometries([other_union], ccrs.PlateCarree(),
                          facecolor="white", edgecolor="none", zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("10m"),
                   facecolor="white", edgecolor="#666666", linewidth=0.5, zorder=4)

    if isobars and grid is not None:
        pmin = np.floor(np.nanmin(grid) / 2) * 2
        pmax = np.ceil(np.nanmax(grid)  / 2) * 2
        levels = np.arange(pmin, pmax + 2, 2)
        cs = ax.contour(gx, gy, grid, levels=levels,
                        colors="#222222", linewidths=0.7,
                        transform=ccrs.PlateCarree(), zorder=6)
        ax.clabel(cs, inline=True, fontsize=6.5, fmt="%d", inline_spacing=4)

    ax.coastlines(resolution="10m", linewidth=1.1, color="#222222", zorder=7)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                   linestyle="-", edgecolor="#333333", linewidth=0.9, zorder=7)
    admin1 = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines",
                                          "10m", facecolor="none",
                                          edgecolor="#555555", linewidth=0.5)
    ax.add_feature(admin1, zorder=7)

    lon0, lon1, lat0, lat1 = EXTENT
    if wind_arrows:
        for s in ok:
            slon, slat = s["lon"], s["lat"]
            if not (lon0-0.1 <= slon <= lon1+0.1 and lat0-0.1 <= slat <= lat1+0.1): continue
            wd = s.get("wind_dir", np.nan)
            if np.isnan(wd) or s[key] < 0.5: continue
            wr = np.radians(wd)
            u  = -s[key] * np.sin(wr) * 0.05
            v  = -s[key] * np.cos(wr) * 0.035
            ax.annotate("", xy=(slon+u, slat+v), xytext=(slon, slat),
                        arrowprops=dict(arrowstyle="-|>", color="#111111",
                                        lw=0.7, mutation_scale=6),
                        transform=ccrs.PlateCarree(), zorder=9)

    margin = 0.05
    for s in ok:
        slon, slat = s["lon"], s["lat"]
        if not (lon0-margin <= slon <= lon1+margin and
                lat0-margin <= slat <= lat1+margin): continue
        ax.plot(slon, slat, "s", color="black", ms=3.2, mec="black", mew=0.0,
                transform=ccrs.PlateCarree(), zorder=10)
        ax.text(slon+0.025, slat+0.02, fmt.format(s[key]),
                fontsize=6.5, fontweight="bold", color="black",
                path_effects=_PE, transform=ccrs.PlateCarree(), zorder=11)

    if s_min and s_max:
        ax.text(0.012, 0.17, f"{fmt.format(vmax_obs)}{unit}",
                transform=ax.transAxes, fontsize=14, fontweight="bold",
                color="#cc0000", path_effects=_PE2, zorder=20)
        ax.text(0.012, 0.12, s_max["name"],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color="#cc0000", path_effects=_PE2, zorder=20)
        ax.text(0.012, 0.07, f"{fmt.format(vmin_obs)}{unit}",
                transform=ax.transAxes, fontsize=14, fontweight="bold",
                color="#0055cc", path_effects=_PE2, zorder=20)
        ax.text(0.012, 0.02, s_min["name"],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color="#0055cc", path_effects=_PE2, zorder=20)

    cax = fig.add_axes([0.875, 0.06, 0.022, 0.88])
    cb  = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                       cax=cax, orientation="vertical", extend="both")
    cb.set_label(unit, fontsize=9, rotation=270, labelpad=14)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)

    ax.set_title(f"{region_label}  •  {title}  •  1 km\n{time_str}",
                 fontsize=13, fontweight="bold", pad=10, loc="center", color="#111111")
    fig.text(0.5, 0.005, credits, ha="center", fontsize=8, color="#888888")

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved  {os.path.basename(outfile)}")


PANELS = [
    ("temp", "Õhutemperatuur 2 m (°C)",          "°C",  "{:.1f}", False, False, "map_1_temperature.png"),
    ("wind", "Tuule kiirus (m/s)",                "m/s", "{:.1f}", True,  False, "map_2_wind_speed.png"),
    ("gust", "Tuule puhang – max (m/s)",          "m/s", "{:.1f}", True,  False, "map_3_wind_gust.png"),
    ("pres", "Meretasemele taandatud rõhk (hPa)", "hPa", "{:.1f}", False, True,  "map_4_pressure.png"),
    ("hum",  "Suhteline niiskus (%)",             "%",   "{:.0f}", False, False, "map_5_humidity.png"),
    ("prec", "Sademete hulk (mm/h)",              "mm",  "{:.1f}", False, False, "map_6_precipitation.png"),
    ("dewp", "Kastepunkt 2 m (°C)",               "°C",  "{:.1f}", False, False, "map_7_dewpoint.png"),
]

_cmaps = load_cmaps()
_gx, _gy = make_grid()
_render_lock = threading.Lock()


def run_once():
    with _render_lock:
        try:
            stations, time_str = fetch()
        except Exception as e:
            print(f"  Fetch error: {e}")
            return False
        ok_count = 0
        for (key, title, unit, fmt, arrows, isobars, fname) in PANELS:
            outfile = os.path.join(OUTPUT_DIR, fname)
            try:
                render_one(stations, key, *_cmaps[key],
                           title, unit, fmt, time_str, _gx, _gy, outfile,
                           wind_arrows=arrows, isobars=isobars)
                ok_count += 1
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  Error {fname}: {e}")

        status = {
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "observation_time": time_str,
            "stations": len(stations),
            "panels_ok": ok_count,
            "panels_total": len(PANELS),
        }
        with open(os.path.join(OUTPUT_DIR, "status.json"), "w") as f:
            json.dump(status, f)
        return ok_count == len(PANELS)


def _background_loop():
    while True:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now}] Rendering …")
        try:
            run_once()
        except Exception as e:
            print(f"  Loop error: {e}")
        time.sleep(UPDATE_INTERVAL)


# ---------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------

app = Flask(__name__)


@app.after_request
def add_cors(resp):
    # Public read-only weather data — safe to allow cross-origin fetch()
    # from your WordPress site's JS (needed for status.json; images via
    # <img> don't strictly need it but this keeps things simple).
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/")
def index():
    # Root endpoint doubles as the health-check / keep-alive ping target.
    return jsonify({"ok": True, "service": "baltic-weather-render"})


@app.route("/status.json")
def status_json():
    path = os.path.join(OUTPUT_DIR, "status.json")
    if not os.path.exists(path):
        return jsonify({"ok": False, "message": "First render not finished yet"}), 503
    return send_from_directory(OUTPUT_DIR, "status.json")


@app.route("/<path:filename>")
def serve_map(filename):
    if filename not in [p[6] for p in PANELS]:
        return Response("Not found", status=404)
    return send_from_directory(OUTPUT_DIR, filename)


# Start the background renderer once, at import time (works both under
# `python app.py` and under a WSGI server like gunicorn with 1 worker).
_started = False
_start_lock = threading.Lock()

def _ensure_started():
    global _started
    with _start_lock:
        if not _started:
            _started = True
            threading.Thread(target=_background_loop, daemon=True).start()

_ensure_started()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
