"""Verification tests for metro-flow-ml deliverables and codebase."""

from __future__ import annotations

import json
import pathlib
import tomllib
import unittest
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "metro-flow-ml.ipynb"
RAW_DATASET_PATH = ROOT / "dataset" / "Metro_Interstate_Traffic_Volume.csv"
CLEANED_DATASET_PATH = ROOT / "dataset" / "traffic_data_cleaned.csv"
MODEL_PATH = ROOT / "model" / "traffic_xgboost_model.pkl"
MAP_HTML_PATH = ROOT / "visualizations" / "traffic_map_view.html"

EXPECTED_CLEAN_COLUMNS = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "traffic_volume",
    "hour",
    "month",
    "day_of_week",
    "year",
    "is_weekend",
    "is_holiday",
    "weather_code",
]


class ProjectConfigTests(unittest.TestCase):
    """Validate project, IDE, and Pyrefly configuration files."""

    def test_python_version_file(self) -> None:
        version_text = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
        self.assertEqual(version_text, "3.14.7")

    def test_toml_and_json_configs_parse(self) -> None:
        with (ROOT / "pyrefly.toml").open("rb") as fh:
            pyrefly_cfg = tomllib.load(fh)
        self.assertIn("**/*.ipynb", pyrefly_cfg.get("project-includes", []))

        with (ROOT / "pyproject.toml").open("rb") as fh:
            project_cfg = tomllib.load(fh)
        self.assertIn("3.14.7", project_cfg["project"]["requires-python"])

        for rel in (".vscode/settings.json", ".vscode/extensions.json"):
            data = json.loads((ROOT / rel).read_text(encoding="utf-8"))
            self.assertIsInstance(data, dict)

    def test_notebook_cells_compile_and_have_no_errors(self) -> None:
        nb: dict[str, Any] = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(nb["metadata"]["language_info"]["version"], "3.14.7")
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreater(len(code_cells), 0)

        for idx, cell in enumerate(code_cells):
            source = "".join(cell["source"])
            compile(source, f"notebook_cell_{idx}", "exec")
            self.assertIsNotNone(
                cell.get("execution_count"),
                f"Code cell {idx} has not been executed",
            )
            errors = [
                out
                for out in cell.get("outputs", [])
                if out.get("output_type") == "error"
            ]
            self.assertEqual(errors, [], f"Code cell {idx} contains error output")
            warnings = [
                "".join(out.get("text", []))
                for out in cell.get("outputs", [])
                if out.get("output_type") == "stream"
                and "Warning" in "".join(out.get("text", []))
            ]
            self.assertEqual(warnings, [], f"Code cell {idx} emitted warnings")


class DeliverableTests(unittest.TestCase):
    """Validate datasets, serialized XGBoost model, and Folium map artifact."""

    def test_raw_and_cleaned_dataset_invariants(self) -> None:
        self.assertTrue(RAW_DATASET_PATH.is_file())
        raw_df = pd.read_csv(RAW_DATASET_PATH, keep_default_na=False)
        self.assertEqual(raw_df.shape, (48204, 9))
        self.assertEqual(int(raw_df.duplicated().sum()), 17)

        self.assertTrue(CLEANED_DATASET_PATH.is_file())
        clean_df = pd.read_csv(CLEANED_DATASET_PATH)
        self.assertEqual(clean_df.shape, (48187, 12))
        self.assertEqual(list(clean_df.columns), EXPECTED_CLEAN_COLUMNS)
        self.assertEqual(int(clean_df.isna().sum().sum()), 0)
        self.assertEqual(
            clean_df["is_holiday"].value_counts().to_dict(),
            {0: 48126, 1: 61},
        )
        self.assertEqual(
            clean_df["is_weekend"].value_counts().to_dict(),
            {0: 34487, 1: 13700},
        )

    def test_serialized_model_loads_and_reproduces_accuracy(self) -> None:
        self.assertTrue(MODEL_PATH.is_file())
        model = joblib.load(MODEL_PATH)

        clean_df = pd.read_csv(CLEANED_DATASET_PATH)
        features = clean_df.drop(columns=["traffic_volume"])
        target = clean_df["traffic_volume"]
        _, x_test, _, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42,
        )
        preds = model.predict(x_test)
        self.assertEqual(preds.shape, (9638,))
        self.assertTrue(np.isfinite(preds).all())
        self.assertGreater(float(r2_score(y_test, preds)), 0.96)

    def test_visualizations_exist_and_use_esri_tiles(self) -> None:
        self.assertTrue(MAP_HTML_PATH.is_file(), "Missing traffic_map_view.html")
        self.assertGreater(MAP_HTML_PATH.stat().st_size, 1024)
        html = MAP_HTML_PATH.read_text(encoding="utf-8")
        self.assertIn("leaflet", html.lower())
        self.assertIn("server.arcgisonline.com", html)
        self.assertIn('"maxWidth": 280', html)
        self.assertNotIn("tile.openstreetmap.org", html)
        self.assertNotIn("basemaps.cartocdn.com", html)


if __name__ == "__main__":
    unittest.main()
