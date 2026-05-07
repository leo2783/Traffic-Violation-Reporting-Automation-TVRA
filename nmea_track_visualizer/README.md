# NMEA GPS Track Visualizer

Convert `.NMEA` recorder GPS logs into an interactive Leaflet map with GeoJSON exports, sampled location markers, optional reverse geocoding, and matching video playback when same-name video files are present.

The parser uses only the Python standard library. The generated HTML loads Leaflet and OpenStreetMap tiles from CDNs.

## Features

- Scans `.NMEA` and `.nmea` files recursively.
- Uses RMC sentences first, with GGA fallback when no valid RMC points exist.
- Exports `tracks.html`, `tracks.geojson`, `locations.geojson`, and `summary.json`.
- Links same-name recorder videos such as `FILE001.MP4` for `FILE001.NMEA`.
- Can pre-fill start/middle/end road and place names with Nominatim reverse geocoding.

## Usage

Run from this folder:

```powershell
python .\visualize_nmea_tracks.py .\examples -o .\output
```

Then serve the folder:

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8000/output/tracks.html
```

For recorder folders that also contain matching video files, serve the folder that contains both the input data and the output folder so the generated map can load those videos.

## Reverse Geocoding

To pre-fill sampled road and place names:

```powershell
python .\visualize_nmea_tracks.py .\examples -o .\output --geocode --geocode-limit 40
```

This sends sampled coordinates to Nominatim and writes a local `output/geocode_cache.json`. Respect the Nominatim usage policy when running large batches.

The generated map can also query Nominatim from the browser when you click a route or map location.

## Outputs

- `output/tracks.html`
- `output/tracks.geojson`
- `output/locations.geojson`
- `output/summary.json`
- `output/geocode_cache.json` when `--geocode` is used

Generated outputs are intentionally ignored by git because they can contain private routes, addresses, and local file names.

## Test

```powershell
python -m unittest discover -s .\tests
```
