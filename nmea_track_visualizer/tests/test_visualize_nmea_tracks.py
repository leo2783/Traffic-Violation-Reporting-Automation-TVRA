from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import visualize_nmea_tracks as visualizer  # noqa: E402


SAMPLE_NMEA = """\
$GNRMC,010000.000,A,2501.9800,N,12133.9240,E,10.0,90.0,010126,,,A*00
$GNRMC,010010.000,A,2501.9860,N,12133.9420,E,12.5,91.0,010126,,,A*00
$GNRMC,010020.000,A,2501.9920,N,12133.9600,E,14.0,92.0,010126,,,A*00
"""


class NMEATrackVisualizerTests(unittest.TestCase):
    def test_parse_nmea_coord(self) -> None:
        self.assertAlmostEqual(visualizer.parse_nmea_coord("2501.9800", "N"), 25.033)
        self.assertAlmostEqual(visualizer.parse_nmea_coord("12133.9240", "E"), 121.5654)
        self.assertAlmostEqual(visualizer.parse_nmea_coord("2501.9800", "S"), -25.033)
        self.assertIsNone(visualizer.parse_nmea_coord("", "N"))

    def test_parse_rmc_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.NMEA"
            path.write_text(SAMPLE_NMEA, encoding="utf-8")

            points = visualizer.parse_nmea_file(path)

        self.assertEqual(len(points), 3)
        self.assertEqual(points[0]["timestamp"], "2026-01-01T01:00:00Z")
        self.assertAlmostEqual(points[0]["speed_kmh"], 18.52)

    def test_build_geojson_finds_lowercase_files_and_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            nested_dir = input_dir / "nested"
            nested_dir.mkdir(parents=True)
            (nested_dir / "sample.nmea").write_text(SAMPLE_NMEA, encoding="utf-8")
            (nested_dir / "sample.MP4").write_bytes(b"")

            geojson, locations_geojson, summary = visualizer.build_geojson(
                input_dir,
                recursive=True,
                geocode_locations=False,
                geocode_limit=0,
                geocode_cache={},
                geocode_delay=0,
            )

        self.assertEqual(summary["nmea_files_found"], 1)
        self.assertEqual(summary["tracks_with_points"], 1)
        self.assertEqual(summary["tracks_with_video"], 1)
        self.assertEqual(len(locations_geojson["features"]), 3)
        self.assertEqual(geojson["features"][0]["properties"]["file"], "nested/sample.nmea")
        self.assertEqual(geojson["features"][0]["properties"]["video_file"], "nested/sample.MP4")


if __name__ == "__main__":
    unittest.main()
