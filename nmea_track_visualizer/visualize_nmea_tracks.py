from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


RMC_TYPES = {"GPRMC", "GNRMC", "GLRMC", "GARMC", "BDRMC"}
GGA_TYPES = {"GPGGA", "GNGGA", "GLGGA", "GAGGA", "BDGGA"}
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "NMEA-Track-Visualizer/0.1"


def strip_checksum(sentence: str) -> str:
    return sentence.split("*", 1)[0].strip()


def parse_nmea_coord(raw: str, hemisphere: str) -> float | None:
    if not raw or not hemisphere:
        return None

    try:
        value = float(raw)
    except ValueError:
        return None

    degrees = math.floor(value / 100)
    minutes = value - degrees * 100
    decimal = degrees + minutes / 60
    if hemisphere.upper() in {"S", "W"}:
        decimal *= -1
    return decimal


def parse_rmc_datetime(time_raw: str, date_raw: str) -> str | None:
    if len(time_raw) < 6 or len(date_raw) != 6:
        return None

    try:
        hour = int(time_raw[0:2])
        minute = int(time_raw[2:4])
        second = int(time_raw[4:6])
        microsecond = 0
        if "." in time_raw:
            fraction = time_raw.split(".", 1)[1]
            microsecond = int((fraction + "000000")[:6])

        day = int(date_raw[0:2])
        month = int(date_raw[2:4])
        year_two_digits = int(date_raw[4:6])
        year = 2000 + year_two_digits if year_two_digits < 80 else 1900 + year_two_digits
        stamp = dt.datetime(year, month, day, hour, minute, second, microsecond, tzinfo=dt.timezone.utc)
    except ValueError:
        return None

    return stamp.isoformat().replace("+00:00", "Z")


def parse_iso_datetime(value: str | None) -> dt.datetime | None:
    if not value or not value.endswith("Z"):
        return None

    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def annotate_elapsed_seconds(points: list[dict[str, Any]]) -> None:
    start_timestamp = None
    for point in points:
        start_timestamp = parse_iso_datetime(point.get("timestamp"))
        if start_timestamp:
            break

    for index, point in enumerate(points):
        point["point_index"] = index
        timestamp = parse_iso_datetime(point.get("timestamp"))
        if start_timestamp and timestamp:
            elapsed = max((timestamp - start_timestamp).total_seconds(), 0.0)
        else:
            elapsed = float(index)
        point["elapsed_seconds"] = round(elapsed, 3)


def parse_rmc(fields: list[str]) -> dict[str, Any] | None:
    # $GNRMC,time,status,lat,N/S,lon,E/W,speed_knots,course,date,...
    if len(fields) < 10 or fields[2].upper() != "A":
        return None

    lat = parse_nmea_coord(fields[3], fields[4])
    lon = parse_nmea_coord(fields[5], fields[6])
    if lat is None or lon is None:
        return None

    point: dict[str, Any] = {
        "lat": lat,
        "lon": lon,
        "timestamp": parse_rmc_datetime(fields[1], fields[9]),
        "source": "RMC",
    }

    try:
        point["speed_kmh"] = float(fields[7]) * 1.852 if fields[7] else None
    except ValueError:
        point["speed_kmh"] = None

    try:
        point["course"] = float(fields[8]) if fields[8] else None
    except ValueError:
        point["course"] = None

    return point


def parse_gga(fields: list[str]) -> dict[str, Any] | None:
    # $GNGGA,time,lat,N/S,lon,E/W,fix_quality,num_satellites,hdop,altitude,...
    if len(fields) < 10:
        return None

    fix_quality = fields[6]
    if fix_quality in {"", "0"}:
        return None

    lat = parse_nmea_coord(fields[2], fields[3])
    lon = parse_nmea_coord(fields[4], fields[5])
    if lat is None or lon is None:
        return None

    point: dict[str, Any] = {
        "lat": lat,
        "lon": lon,
        "timestamp": fields[1] or None,
        "source": "GGA",
    }

    try:
        point["altitude_m"] = float(fields[9]) if fields[9] else None
    except ValueError:
        point["altitude_m"] = None

    return point


def parse_nmea_file(path: Path, use_gga_fallback: bool = True) -> list[dict[str, Any]]:
    rmc_points: list[dict[str, Any]] = []
    gga_points: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("$"):
                continue

            fields = strip_checksum(line[1:]).split(",")
            sentence_type = fields[0].upper()

            if sentence_type in RMC_TYPES:
                point = parse_rmc(fields)
                if point:
                    rmc_points.append(point)
            elif use_gga_fallback and sentence_type in GGA_TYPES:
                point = parse_gga(fields)
                if point:
                    gga_points.append(point)

    return rmc_points or gga_points


def find_nmea_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() == ".nmea"
    )


def find_matching_video(path: Path) -> Path | None:
    for suffix in (".MP4", ".mp4", ".MOV", ".mov", ".TS", ".ts"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def build_point_meta(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    meta: list[dict[str, Any]] = []
    for point in points:
        speed = point.get("speed_kmh")
        meta.append(
            {
                "point_index": point.get("point_index"),
                "timestamp": point.get("timestamp"),
                "elapsed_seconds": point.get("elapsed_seconds"),
                "speed_kmh": round(speed, 2) if isinstance(speed, (int, float)) else None,
                "course": point.get("course"),
                "source": point.get("source"),
            }
        )
    return meta


def load_geocode_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_geocode_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def location_key(lat: float, lon: float) -> str:
    return f"{lat:.5f},{lon:.5f}"


def first_address_value(address: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = address.get(key)
        if value:
            return str(value)
    return None


def unique_parts(parts: list[str | None]) -> list[str]:
    values: list[str] = []
    for part in parts:
        if part and part not in values:
            values.append(part)
    return values


def summarize_geocode(payload: dict[str, Any]) -> dict[str, Any]:
    address = payload.get("address") if isinstance(payload.get("address"), dict) else {}
    road = first_address_value(
        address,
        (
            "road",
            "pedestrian",
            "footway",
            "cycleway",
            "path",
            "residential",
            "neighbourhood",
            "suburb",
        ),
    )
    place_parts = unique_parts(
        [
            first_address_value(address, ("village", "town", "city", "municipality")),
            first_address_value(address, ("city_district", "district", "county")),
            first_address_value(address, ("state",)),
            first_address_value(address, ("country",)),
        ]
    )

    return {
        "road": road,
        "place": " / ".join(place_parts) if place_parts else None,
        "display_name": payload.get("display_name"),
        "osm_type": payload.get("osm_type"),
        "osm_id": payload.get("osm_id"),
    }


def reverse_geocode(
    lat: float,
    lon: float,
    cache: dict[str, Any],
    delay_seconds: float,
) -> dict[str, Any]:
    key = location_key(lat, lon)
    if key in cache:
        return cache[key]

    query = urllib.parse.urlencode(
        {
            "format": "jsonv2",
            "lat": f"{lat:.7f}",
            "lon": f"{lon:.7f}",
            "zoom": "18",
            "addressdetails": "1",
            "accept-language": "zh-TW",
        }
    )
    request = urllib.request.Request(
        f"{NOMINATIM_REVERSE_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        cache[key] = summarize_geocode(payload)
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        cache[key] = {"road": None, "place": None, "display_name": None, "error": str(exc)}

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    return cache[key]


def sample_track_points(points: list[dict[str, Any]]) -> list[tuple[str, int]]:
    if not points:
        return []

    candidates = [
        ("start", 0),
        ("middle", len(points) // 2),
        ("end", len(points) - 1),
    ]
    sampled: list[tuple[str, int]] = []
    seen_indexes: set[int] = set()
    for role, index in candidates:
        if index in seen_indexes:
            continue
        seen_indexes.add(index)
        sampled.append((role, index))
    return sampled


def build_location_feature(
    relative_path: str,
    role: str,
    point: dict[str, Any],
    geocode: dict[str, Any] | None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "file": relative_path,
        "role": role,
        "point_index": point.get("point_index"),
        "timestamp": point.get("timestamp"),
        "elapsed_seconds": point.get("elapsed_seconds"),
        "speed_kmh": round(point["speed_kmh"], 2) if isinstance(point.get("speed_kmh"), (int, float)) else None,
        "course": point.get("course"),
        "lat": round(point["lat"], 7),
        "lon": round(point["lon"], 7),
    }

    if geocode:
        properties.update(
            {
                "road": geocode.get("road"),
                "place": geocode.get("place"),
                "display_name": geocode.get("display_name"),
                "geocode_error": geocode.get("error"),
            }
        )

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [point["lon"], point["lat"]],
        },
        "properties": properties,
    }


def build_geojson(
    input_dir: Path,
    recursive: bool,
    geocode_locations: bool,
    geocode_limit: int,
    geocode_cache: dict[str, Any],
    geocode_delay: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    files = find_nmea_files(input_dir, recursive)
    features: list[dict[str, Any]] = []
    location_features: list[dict[str, Any]] = []
    total_points = 0
    geocoded_locations = 0
    tracks_with_video = 0

    for path in files:
        points = parse_nmea_file(path)
        if not points:
            continue

        annotate_elapsed_seconds(points)
        coordinates = [[point["lon"], point["lat"]] for point in points]
        total_points += len(points)
        relative_path = path.relative_to(input_dir).as_posix()
        video_path = find_matching_video(path)
        video_relative_path = video_path.relative_to(input_dir).as_posix() if video_path else None
        if video_path:
            tracks_with_video += 1

        timestamps = [point.get("timestamp") for point in points if point.get("timestamp")]
        speeds = [point.get("speed_kmh") for point in points if isinstance(point.get("speed_kmh"), (int, float))]
        track_locations: list[dict[str, Any]] = []

        for role, point_index in sample_track_points(points):
            point = points[point_index]
            geocode: dict[str, Any] | None = None
            if geocode_locations and geocoded_locations < geocode_limit:
                geocode = reverse_geocode(point["lat"], point["lon"], geocode_cache, geocode_delay)
                geocoded_locations += 1

            location_feature = build_location_feature(relative_path, role, point, geocode)
            location_features.append(location_feature)
            track_locations.append(location_feature["properties"])

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates,
                },
                "properties": {
                    "file": relative_path,
                    "points": len(points),
                    "start_time": timestamps[0] if timestamps else None,
                    "end_time": timestamps[-1] if timestamps else None,
                    "max_speed_kmh": round(max(speeds), 2) if speeds else None,
                    "avg_speed_kmh": round(sum(speeds) / len(speeds), 2) if speeds else None,
                    "locations": track_locations,
                    "point_meta": build_point_meta(points),
                    "video_file": video_relative_path,
                    "video_exists": bool(video_path),
                },
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    locations_geojson = {
        "type": "FeatureCollection",
        "features": location_features,
    }
    summary = {
        "input_dir": input_dir.name,
        "nmea_files_found": len(files),
        "tracks_with_points": len(features),
        "location_markers": len(location_features),
        "geocoded_locations": geocoded_locations,
        "total_points": total_points,
        "tracks_with_video": tracks_with_video,
    }
    return geojson, locations_geojson, summary


def path_to_relative_url(path: Path, base_dir: Path) -> str:
    try:
        relative = os.path.relpath(path, base_dir).replace(os.sep, "/")
    except ValueError:
        return path.resolve().as_uri()
    return urllib.parse.quote(relative, safe="/._-~")


def attach_video_urls(geojson: dict[str, Any], input_dir: Path, output_dir: Path) -> None:
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        video_file = properties.get("video_file")
        if not video_file:
            continue

        video_path = input_dir / video_file
        if video_path.exists():
            properties["video_url"] = path_to_relative_url(video_path, output_dir)


def html_document(geojson: dict[str, Any], locations_geojson: dict[str, Any], summary: dict[str, Any]) -> str:
    geojson_text = json.dumps(geojson, ensure_ascii=False)
    locations_text = json.dumps(locations_geojson, ensure_ascii=False)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    title = "NMEA GPS 軌跡地圖"

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body {{
      height: 100%;
      margin: 0;
      font-family: Arial, sans-serif;
    }}
    #map {{
      height: 100%;
      width: 100%;
    }}
    .panel {{
      position: absolute;
      z-index: 1000;
      top: 12px;
      right: 12px;
      width: min(360px, calc(100vw - 24px));
      max-height: calc(100vh - 24px);
      overflow: auto;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid #c9ced6;
      border-radius: 6px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
      padding: 12px;
      box-sizing: border-box;
      color: #1f2937;
    }}
    .panel h1 {{
      font-size: 16px;
      margin: 0 0 8px;
    }}
    .panel pre {{
      white-space: pre-wrap;
      font-size: 12px;
      line-height: 1.4;
      margin: 0;
    }}
    .hint {{
      border-top: 1px solid #d1d5db;
      color: #374151;
      font-size: 12px;
      line-height: 1.45;
      margin-top: 10px;
      padding-top: 10px;
    }}
    .popup-table {{
      border-collapse: collapse;
      font-size: 12px;
      line-height: 1.45;
    }}
    .popup-table th {{
      color: #4b5563;
      font-weight: 600;
      padding: 2px 8px 2px 0;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    .popup-table td {{
      padding: 2px 0;
      vertical-align: top;
    }}
    .popup-title {{
      font-weight: 700;
      margin-bottom: 6px;
      max-width: 340px;
      word-break: break-word;
    }}
    .location-list {{
      margin: 6px 0 0;
      padding-left: 16px;
      max-width: 340px;
    }}
    .leaflet-popup-content {{
      min-width: min(440px, calc(100vw - 96px));
    }}
    .recorder-video {{
      width: 100%;
      aspect-ratio: 16 / 9;
      display: block;
      margin: 8px 0;
      background: #111827;
      border-radius: 4px;
    }}
    .video-note {{
      color: #4b5563;
      font-size: 12px;
      margin: 4px 0 8px;
    }}
    .empty {{
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      z-index: 1001;
      background: #f7f8fa;
      color: #1f2937;
      font: 16px Arial, sans-serif;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <aside class="panel">
    <h1>{html.escape(title)}</h1>
    <pre id="summary">{html.escape(summary_text)}</pre>
    <div class="hint">點彩色軌跡線即可顯示該線段的經過時間與對應紀錄器畫面；GPS 點圖層預設關閉，可從右上角打開。</div>
  </aside>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const geojson = {geojson_text};
    const locationsGeojson = {locations_text};

    if (!geojson.features.length) {{
      document.body.insertAdjacentHTML("beforeend", '<div class="empty">No GPS tracks found.</div>');
    }}

    const map = L.map("map", {{ preferCanvas: true }});
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const palette = [
      "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
      "#0891b2", "#be123c", "#4f46e5", "#15803d", "#a16207"
    ];
    const reverseCache = new Map();
    const featuresByFile = new Map(geojson.features.map((feature) => [feature.properties?.file, feature]));

    function escapeHtml(value) {{
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }}

    function row(label, value) {{
      if (value === null || value === undefined || value === "") return "";
      return `<tr><th>${{escapeHtml(label)}}</th><td>${{escapeHtml(value)}}</td></tr>`;
    }}

    function formatElapsed(seconds) {{
      const value = Number(seconds);
      if (!Number.isFinite(value)) return "";
      const total = Math.max(0, Math.round(value));
      const hours = Math.floor(total / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const secs = total % 60;
      if (hours > 0) {{
        return `${{String(hours).padStart(2, "0")}}:${{String(minutes).padStart(2, "0")}}:${{String(secs).padStart(2, "0")}}`;
      }}
      return `${{String(minutes).padStart(2, "0")}}:${{String(secs).padStart(2, "0")}}`;
    }}

    function interpolateNumber(a, b, ratio) {{
      const left = Number(a);
      const right = Number(b);
      if (Number.isFinite(left) && Number.isFinite(right)) {{
        return left + (right - left) * ratio;
      }}
      if (Number.isFinite(left)) return left;
      if (Number.isFinite(right)) return right;
      return null;
    }}

    function interpolatedMeta(pointMeta, index, ratio) {{
      const left = pointMeta[index] || {{}};
      const right = pointMeta[index + 1] || left;
      const elapsed = interpolateNumber(left.elapsed_seconds, right.elapsed_seconds, ratio);
      const speed = interpolateNumber(left.speed_kmh, right.speed_kmh, ratio);
      const course = interpolateNumber(left.course, right.course, ratio);

      return {{
        point_index: `${{index}}-${{Math.min(index + 1, pointMeta.length - 1)}}`,
        timestamp: ratio < 0.5 ? left.timestamp : right.timestamp,
        elapsed_seconds: elapsed === null ? null : Math.round(elapsed * 1000) / 1000,
        speed_kmh: speed === null ? null : Math.round(speed * 100) / 100,
        course: course === null ? null : Math.round(course * 100) / 100,
        source: left.source || right.source
      }};
    }}

    function nearestTrackPoint(feature, latlng) {{
      const coordinates = feature.geometry?.coordinates || [];
      const pointMeta = feature.properties?.point_meta || [];
      const clickPoint = map.latLngToLayerPoint(latlng);
      let best = null;

      if (coordinates.length === 1) {{
        const coord = coordinates[0];
        const pointLatLng = L.latLng(coord[1], coord[0]);
        const projected = map.latLngToLayerPoint(pointLatLng);
        return {{
          index: 0,
          distancePx: clickPoint.distanceTo(projected),
          latlng: pointLatLng,
          meta: pointMeta[0] || {{}}
        }};
      }}

      for (let index = 0; index < coordinates.length - 1; index += 1) {{
        const startCoord = coordinates[index];
        const endCoord = coordinates[index + 1];
        const startLatLng = L.latLng(startCoord[1], startCoord[0]);
        const endLatLng = L.latLng(endCoord[1], endCoord[0]);
        const startPoint = map.latLngToLayerPoint(startLatLng);
        const endPoint = map.latLngToLayerPoint(endLatLng);

        const dx = endPoint.x - startPoint.x;
        const dy = endPoint.y - startPoint.y;
        const lengthSquared = dx * dx + dy * dy;
        const ratio = lengthSquared === 0
          ? 0
          : Math.max(0, Math.min(1, ((clickPoint.x - startPoint.x) * dx + (clickPoint.y - startPoint.y) * dy) / lengthSquared));
        const projected = L.point(startPoint.x + dx * ratio, startPoint.y + dy * ratio);
        const distancePx = clickPoint.distanceTo(projected);

        if (!best || distancePx < best.distancePx) {{
          best = {{
            index: `${{index}}-${{index + 1}}`,
            distancePx,
            latlng: map.layerPointToLatLng(projected),
            meta: interpolatedMeta(pointMeta, index, ratio)
          }};
        }}
      }}

      return best;
    }}

    function nearestFeaturePoint(latlng, maxDistancePx = 42) {{
      let bestHit = null;

      geojson.features.forEach((feature) => {{
        const nearest = nearestTrackPoint(feature, latlng);
        if (!nearest) return;
        if (!bestHit || nearest.distancePx < bestHit.nearest.distancePx) {{
          bestHit = {{ feature, nearest }};
        }}
      }});

      if (!bestHit || bestHit.nearest.distancePx > maxDistancePx) {{
        return null;
      }}
      return bestHit;
    }}

    function videoBlock(props, nearest) {{
      const elapsed = Number(nearest?.meta?.elapsed_seconds ?? 0);
      if (!props.video_url) {{
        return '<div class="video-note">找不到同名影片檔，無法顯示紀錄器畫面。</div>';
      }}
      const seekSeconds = Number.isFinite(elapsed) ? Math.max(0, elapsed) : 0;
      const videoUrl = `${{props.video_url}}#t=${{seekSeconds.toFixed(3)}}`;

      return `
        <video class="recorder-video" controls preload="auto" muted playsinline
          data-seek="${{escapeHtml(seekSeconds)}}" src="${{escapeHtml(videoUrl)}}"></video>
        <div class="video-note">影片會自動跳到 GPS 點對應時間；若畫面未更新，請按播放或拖動時間軸。</div>
      `;
    }}

    function activatePopupVideo() {{
      const video = document.querySelector(".leaflet-popup-content .recorder-video");
      if (!video) return;

      const target = Number(video.dataset.seek || 0);
      const seek = () => {{
        try {{
          if (Number.isFinite(target)) {{
            video.currentTime = Math.max(0, target);
          }}
          video.pause();
        }} catch (error) {{
          // Some browsers reject early seeks until metadata is ready.
        }}
      }};

      video.pause();
      video.load();
      if (video.readyState >= 1) {{
        seek();
      }} else {{
        video.addEventListener("loadedmetadata", seek, {{ once: true }});
      }}
      video.addEventListener("canplay", seek, {{ once: true }});
      video.addEventListener("seeked", () => video.pause(), {{ once: true }});
      setTimeout(seek, 250);
      setTimeout(seek, 1000);
    }}

    function roleLabel(role) {{
      return {{
        start: "起點",
        middle: "中段",
        end: "終點"
      }}[role] || role || "";
    }}

    function locationSummary(location) {{
      if (!location) return "";
      if (location.road && location.place) return `${{location.road}}｜${{location.place}}`;
      if (location.road) return location.road;
      if (location.place) return location.place;
      return location.display_name || "";
    }}

    function markerPopup(props) {{
      const title = `${{roleLabel(props.role)}}：${{props.road || props.place || props.file || "GPS point"}}`;
      return `
        <div class="popup-title">${{escapeHtml(title)}}</div>
        <table class="popup-table">
          ${{row("路名", props.road)}}
          ${{row("地點", props.place)}}
          ${{row("地址", props.display_name)}}
          ${{row("時間", props.timestamp)}}
          ${{row("經過時間", formatElapsed(props.elapsed_seconds))}}
          ${{row("速度", props.speed_kmh !== null && props.speed_kmh !== undefined ? `${{props.speed_kmh}} km/h` : "")}}
          ${{row("座標", `${{Number(props.lat).toFixed(6)}}, ${{Number(props.lon).toFixed(6)}}`)}}
          ${{row("檔案", props.file)}}
          ${{row("查詢狀態", props.geocode_error ? `失敗：${{props.geocode_error}}` : "")}}
        </table>
      `;
    }}

    function trackPopup(props, clickedLocation, nearest) {{
      const locations = (props.locations || [])
        .map((location) => {{
          const summary = locationSummary(location);
          const suffix = summary ? `：${{escapeHtml(summary)}}` : "";
          return `<li>${{escapeHtml(roleLabel(location.role))}}${{suffix}}</li>`;
        }})
        .join("");
      const meta = nearest?.meta || {{}};
      const nearestLocation = nearest
        ? `${{nearest.latlng.lat.toFixed(6)}}, ${{nearest.latlng.lng.toFixed(6)}}`
        : "";
      const speed = meta.speed_kmh !== null && meta.speed_kmh !== undefined ? `${{meta.speed_kmh}} km/h` : "";
      const course = meta.course !== null && meta.course !== undefined ? `${{meta.course}}°` : "";

      return `
        <div class="popup-title">${{escapeHtml(props.file || "NMEA file")}}</div>
        ${{videoBlock(props, nearest)}}
        <table class="popup-table">
          ${{row("GPS 點序號", nearest?.index)}}
          ${{row("經過時間", formatElapsed(meta.elapsed_seconds))}}
          ${{row("GPS 時間", meta.timestamp)}}
          ${{row("當下速度", speed)}}
          ${{row("航向", course)}}
          ${{row("最近 GPS 點", nearestLocation)}}
          ${{row("影片檔", props.video_file)}}
          ${{row("點數", props.points)}}
          ${{row("開始", props.start_time)}}
          ${{row("結束", props.end_time)}}
          ${{row("平均速度", props.avg_speed_kmh !== null && props.avg_speed_kmh !== undefined ? `${{props.avg_speed_kmh}} km/h` : "")}}
          ${{row("最高速度", props.max_speed_kmh !== null && props.max_speed_kmh !== undefined ? `${{props.max_speed_kmh}} km/h` : "")}}
          ${{row("點擊位置", clickedLocation)}}
        </table>
        ${{locations ? `<ul class="location-list">${{locations}}</ul>` : ""}}
      `;
    }}

    function parseNominatim(payload) {{
      const address = payload.address || {{}};
      const road = address.road || address.pedestrian || address.footway || address.cycleway ||
        address.path || address.residential || address.neighbourhood || address.suburb || "";
      const placeParts = [
        address.village || address.town || address.city || address.municipality,
        address.city_district || address.district || address.county,
        address.state,
        address.country
      ].filter(Boolean);
      const uniquePlaces = [...new Set(placeParts)];
      return {{
        road,
        place: uniquePlaces.join(" / "),
        display_name: payload.display_name || ""
      }};
    }}

    async function reverseLookup(latlng) {{
      const key = `${{latlng.lat.toFixed(5)}},${{latlng.lng.toFixed(5)}}`;
      if (reverseCache.has(key)) return reverseCache.get(key);

      const params = new URLSearchParams({{
        format: "jsonv2",
        lat: latlng.lat.toFixed(7),
        lon: latlng.lng.toFixed(7),
        zoom: "18",
        addressdetails: "1",
        "accept-language": "zh-TW"
      }});
      const response = await fetch(`https://nominatim.openstreetmap.org/reverse?${{params.toString()}}`);
      if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
      const location = parseNominatim(await response.json());
      reverseCache.set(key, location);
      return location;
    }}

    async function openLocationPopup(latlng, baseHtml = "") {{
      const coordinates = `${{latlng.lat.toFixed(6)}}, ${{latlng.lng.toFixed(6)}}`;
      const loading = `
        ${{baseHtml}}
        <table class="popup-table">
          ${{row("座標", coordinates)}}
          ${{row("地理位置", "查詢中")}}
        </table>
      `;
      const popup = L.popup({{ maxWidth: 520 }}).setLatLng(latlng).setContent(loading).openOn(map);
      activatePopupVideo();

      try {{
        const location = await reverseLookup(latlng);
        popup.setContent(`
          ${{baseHtml}}
          <table class="popup-table">
            ${{row("路名", location.road)}}
            ${{row("地點", location.place)}}
            ${{row("地址", location.display_name)}}
            ${{row("座標", coordinates)}}
          </table>
        `);
        activatePopupVideo();
      }} catch (error) {{
        popup.setContent(`
          ${{baseHtml}}
          <table class="popup-table">
            ${{row("座標", coordinates)}}
            ${{row("地理位置", "查詢失敗，請確認網路連線")}}
          </table>
        `);
        activatePopupVideo();
      }}
    }}

    let suppressNextMapClick = false;

    function withSuppressedMapClick(callback) {{
      suppressNextMapClick = true;
      callback();
      setTimeout(() => {{
        suppressNextMapClick = false;
      }}, 0);
    }}

    function openTrackPopup(feature, latlng, nearest) {{
      const props = feature.properties || {{}};
      const baseHtml = trackPopup(
        props,
        `${{latlng.lat.toFixed(6)}}, ${{latlng.lng.toFixed(6)}}`,
        nearest
      );
      openLocationPopup(latlng, baseHtml);
    }}

    function handleTrackClick(feature, event) {{
      if (event.originalEvent) {{
        L.DomEvent.stop(event.originalEvent);
      }}

      withSuppressedMapClick(() => {{
        openTrackPopup(feature, event.latlng, nearestTrackPoint(feature, event.latlng));
      }});
    }}

    const visibleTrackLayer = L.geoJSON(geojson, {{
      style: (feature) => {{
        const index = geojson.features.indexOf(feature);
        return {{
          color: palette[index % palette.length],
          weight: 5,
          opacity: 0.88,
          lineCap: "round",
          lineJoin: "round"
        }};
      }},
      onEachFeature: (feature, featureLayer) => {{
        featureLayer.on("click", (event) => {{
          handleTrackClick(feature, event);
        }});
      }},
      bubblingMouseEvents: false
    }});

    const hitTrackLayer = L.geoJSON(geojson, {{
      style: () => ({{
        color: "#000000",
        weight: 22,
        opacity: 0.001,
        lineCap: "round",
        lineJoin: "round"
      }}),
      onEachFeature: (feature, featureLayer) => {{
        featureLayer.on("click", (event) => {{
          handleTrackClick(feature, event);
        }});
      }},
      bubblingMouseEvents: false
    }});

    const trackLayer = L.featureGroup([visibleTrackLayer, hitTrackLayer]).addTo(map);

    const pointRenderer = L.canvas({{ padding: 0.5 }});
    const gpsPointLayer = L.layerGroup();

    function shouldRenderGpsPoint(index, total) {{
      return index === 0 || index === total - 1 || index % 10 === 0;
    }}

    geojson.features.forEach((feature) => {{
      const props = feature.properties || {{}};
      const coordinates = feature.geometry?.coordinates || [];
      const pointMeta = props.point_meta || [];

      coordinates.forEach((coord, index) => {{
        if (!shouldRenderGpsPoint(index, coordinates.length)) return;
        const latlng = L.latLng(coord[1], coord[0]);
        const marker = L.circleMarker(latlng, {{
          renderer: pointRenderer,
          radius: 2.2,
          color: "#111827",
          weight: 1,
          fillColor: "#fbbf24",
          fillOpacity: 0.82,
          opacity: 0.65
        }});

        marker.on("click", (event) => {{
          if (event.originalEvent) {{
            L.DomEvent.stop(event.originalEvent);
          }}
          const nearest = {{
            index,
            distancePx: 0,
            latlng,
            meta: pointMeta[index] || {{}}
          }};
          withSuppressedMapClick(() => {{
            openTrackPopup(feature, latlng, nearest);
          }});
        }});

        marker.addTo(gpsPointLayer);
      }});
    }});

    const locationLayer = L.geoJSON(locationsGeojson, {{
      pointToLayer: (feature, latlng) => {{
        const props = feature.properties || {{}};
        return L.circleMarker(latlng, {{
          radius: props.road || props.place ? 5 : 3,
          color: "#111827",
          weight: 1,
          fillColor: props.road || props.place ? "#f59e0b" : "#6b7280",
          fillOpacity: 0.92
        }});
      }},
      onEachFeature: (feature, featureLayer) => {{
        const props = feature.properties || {{}};
        featureLayer.on("click", (event) => {{
          if (event.originalEvent) {{
            L.DomEvent.stop(event.originalEvent);
          }}
          const trackFeature = featuresByFile.get(props.file);
          if (!trackFeature) {{
            featureLayer.bindPopup(markerPopup(props)).openPopup();
            return;
          }}
          const pointMeta = trackFeature.properties?.point_meta || [];
          const pointIndex = Number(props.point_index);
          const nearest = {{
            index: Number.isFinite(pointIndex) ? pointIndex : "",
            distancePx: 0,
            latlng: event.latlng,
            meta: Number.isFinite(pointIndex)
              ? (pointMeta[pointIndex] || props)
              : props
          }};
          withSuppressedMapClick(() => {{
            openTrackPopup(trackFeature, event.latlng, nearest);
          }});
        }});
      }}
    }}).addTo(map);

    L.control.layers(null, {{
      "GPS 軌跡": trackLayer,
      "GPS 點": gpsPointLayer,
      "代表位置": locationLayer
    }}, {{ collapsed: false }}).addTo(map);

    map.on("click", (event) => {{
      if (suppressNextMapClick) {{
        return;
      }}
      const hit = nearestFeaturePoint(event.latlng);
      if (hit) {{
        openTrackPopup(hit.feature, event.latlng, hit.nearest);
        return;
      }}
      openLocationPopup(event.latlng);
    }});

    if (trackLayer.getBounds().isValid()) {{
      map.fitBounds(trackLayer.getBounds(), {{ padding: [28, 28] }});
    }} else {{
      map.setView([23.8, 121.0], 7);
    }}
  </script>
</body>
</html>
"""


def write_outputs(
    geojson: dict[str, Any],
    locations_geojson: dict[str, Any],
    summary: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    attach_video_urls(geojson, input_dir, output_dir)
    (output_dir / "tracks.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "locations.geojson").write_text(
        json.dumps(locations_geojson, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "tracks.html").write_text(
        html_document(geojson, locations_geojson, summary),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPS tracks from all .NMEA/.nmea files in a folder.")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing .NMEA/.nmea files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Folder for tracks.html, tracks.geojson, and summary.json.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the input folder itself.",
    )
    parser.add_argument(
        "--geocode",
        action="store_true",
        help="Pre-fill road and place names for sampled track points with Nominatim reverse geocoding.",
    )
    parser.add_argument(
        "--geocode-limit",
        type=int,
        default=40,
        help="Maximum sampled points to reverse geocode when --geocode is used.",
    )
    parser.add_argument(
        "--geocode-delay",
        type=float,
        default=1.0,
        help="Delay between new reverse geocoding requests, in seconds.",
    )
    parser.add_argument(
        "--geocode-cache",
        type=Path,
        default=Path("output/geocode_cache.json"),
        help="JSON cache for reverse geocoding results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    geocode_cache = load_geocode_cache(args.geocode_cache.resolve())
    geojson, locations_geojson, summary = build_geojson(
        input_dir,
        recursive=not args.no_recursive,
        geocode_locations=args.geocode,
        geocode_limit=max(args.geocode_limit, 0),
        geocode_cache=geocode_cache,
        geocode_delay=max(args.geocode_delay, 0),
    )
    write_outputs(geojson, locations_geojson, summary, input_dir, output_dir)
    if args.geocode:
        save_geocode_cache(args.geocode_cache.resolve(), geocode_cache)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"HTML map: {output_dir / 'tracks.html'}")
    print(f"GeoJSON:  {output_dir / 'tracks.geojson'}")
    print(f"Locations: {output_dir / 'locations.geojson'}")
    print("Serve from the folder that contains the input data and output folder:")
    print("python -m http.server 8000 --bind 127.0.0.1")
    map_location = path_to_relative_url(output_dir / "tracks.html", Path.cwd().resolve())
    if map_location.startswith("file:"):
        print(f"Map file: {map_location}")
    else:
        print(f"Map URL: http://127.0.0.1:8000/{map_location}")


if __name__ == "__main__":
    main()
