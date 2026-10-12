
import json
import unittest
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = OUTPUT_DIR / "hotel_booking_model.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics.json"


class TestHotelMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists() or not METRICS_PATH.exists():
            raise FileNotFoundError(
                "Model artifacts are missing. Run train_model.py first."
            )

        cls.model = joblib.load(MODEL_PATH)

        with open(METRICS_PATH, "r") as file:
            cls.metrics = json.load(file)

    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.exists())

    def test_metrics_file_exists(self):
        self.assertTrue(METRICS_PATH.exists())

    def test_accuracy_is_valid(self):
        accuracy = self.metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_dataset_information(self):
        self.assertEqual(self.metrics["target"], "booking_status")
        self.assertGreater(self.metrics["records"], 0)
        self.assertGreater(self.metrics["features"], 0)

    def test_model_can_predict(self):
        df = pd.read_csv(BASE_DIR / "hotel.csv")
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["booking_status"])

        X = df.drop(columns=["booking_status"])
        id_columns = [
            col for col in X.columns
            if col.lower() in ("booking_id", "id", "user_id")
        ]
        X = X.drop(columns=id_columns)

        prediction = self.model.predict(X.head(5))
        self.assertEqual(len(prediction), min(5, len(X)))

    def test_confusion_matrix_is_present(self):
        matrix = self.metrics["confusion_matrix"]
        self.assertTrue(len(matrix) > 0)


if __name__ == "__main__":
    unittest.main()
