import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.telemetry.generator import generate_dataset
from src.telemetry.processor import process_telemetry


class AnomalyDetector:
    """
    Isolation Forest based anomaly detector for ADIS.

    The model is trained only on normal behavioral data.
    """

    def __init__(
        self,
        contamination=0.10,
        random_state=42,
    ):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )

    def train(self, X_train):
        """
        Train the anomaly detector using normal behavior only.
        """

        self.model.fit(X_train)

        print(
            f"[ADIS] Anomaly detector trained on "
            f"{len(X_train)} normal samples."
        )

    def predict(self, X):
        """
        Predict whether each sample is normal or anomalous.

        Isolation Forest:
            1  = normal
           -1  = anomaly

        ADIS output:
            0  = normal
            1  = anomaly
        """

        predictions = self.model.predict(X)

        return (predictions == -1).astype(int)

    def anomaly_score(self, X):
        """
        Return anomaly scores.

        Lower Isolation Forest decision_function values
        indicate more anomalous behavior.
        """

        return self.model.decision_function(X)


def evaluate_detector(y_true, y_pred):
    """
    Evaluate anomaly detection performance.
    """

    print("\n" + "=" * 50)
    print("ADIS ANOMALY DETECTION EVALUATION")
    print("=" * 50)

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["normal", "anomaly"],
            zero_division=0,
        )
    )


if __name__ == "__main__":

    print("[ADIS] Starting M2.1 — Isolation Forest")

    
    # 1. Generate synthetic telemetry
    df = generate_dataset()

    print(f"[ADIS] Dataset shape: {df.shape}")


    # 2. Process telemetry
    features, labels = process_telemetry(df)

    # ---------------------------------
    # 3. Create binary evaluation label
    # ---------------------------------
    #
    # normal = 0
    # anything suspicious = 1
    #

    y_true = (labels != "normal").astype(int)

    
    # 4. Select normal data for training
    normal_mask = labels == "normal"

    X_train = features[normal_mask]

    print(
        f"[ADIS] Normal training samples: "
        f"{len(X_train)}"
    )

    
    # 5. Create detector
    detector = AnomalyDetector(
        contamination=0.10,
        random_state=42,
    )

    
    # 6. Train
    detector.train(X_train)

    
    # 7. Predict all telemetry
    predictions = detector.predict(features)

    
    # 8. Evaluate
    evaluate_detector(
        y_true,
        predictions,
    )


    # 9. Show anomaly counts
    normal_predictions = (predictions == 0).sum()
    anomaly_predictions = (predictions == 1).sum()

    print("\nPrediction Summary:")
    print(f"Normal:   {normal_predictions}")
    print(f"Anomaly:  {anomaly_predictions}")

    print("\n[ADIS] M2.1 completed successfully.")
