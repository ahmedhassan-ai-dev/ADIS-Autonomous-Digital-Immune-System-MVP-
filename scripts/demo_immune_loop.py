from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from src.analysis.threat_analyzer import BehavioralThreatAnalyzer
from src.sandbox.isolation_chamber import IsolationSandbox
from src.memory.immune_memory import ImmuneMemory
from src.recognition_policy import RecognitionPolicy


# ================================================================
# Configuration
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "merged_binary_detector.joblib"
)

ENCODER_PATH = (
    PROJECT_ROOT
    / "models"
    / "behavioral_encoder.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cicids2017"
    / "Portscan-Friday-no-metadata.parquet"
)

# Separate demo memory so running the demo repeatedly
# does not pollute the real production immune memory.
DEMO_MEMORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "immune_memory_demo"
)

KNOWN_THRESHOLD = 0.92
NEAR_THRESHOLD = 0.80

# Set True when you want a clean reproducible demonstration.
RESET_DEMO_MEMORY = True


# ================================================================
# Helpers
# ================================================================

def clean_for_detector(df: pd.DataFrame, features):
    """
    Clean input specifically for the binary detector.

    The behavioral encoder performs its own imputation.
    """

    X = df[features].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.infer_objects(copy=False)


    medians = X.median(
        numeric_only=True
    )

    X = X.fillna(medians)

    return X


def print_separator():
    print(
        "\n"
        + "=" * 78
    )


# ================================================================
# Main
# ================================================================

def main():

    print_separator()

    print(
        "[ADIS] M3 + M4 — End-to-End Immune Loop Demo"
    )

    print_separator()

    # ------------------------------------------------------------
    # Validate files
    # ------------------------------------------------------------

    for path in [
        MODEL_PATH,
        ENCODER_PATH,
        DATA_PATH,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # ------------------------------------------------------------
    # Load detector
    # ------------------------------------------------------------

    print("\n[1] Loading binary detector...")

    detector_bundle = joblib.load(
        MODEL_PATH
    )

    detector = detector_bundle["model"]
    feature_cols = detector_bundle["features"]

    print(
        f"    Detector features: {len(feature_cols)}"
    )

    # ------------------------------------------------------------
    # Load behavioral encoder
    # ------------------------------------------------------------

    print("\n[2] Loading behavioral encoder...")

    analyzer = BehavioralThreatAnalyzer(
        encoder_path=ENCODER_PATH,
        input_dim=len(feature_cols),
    )

    # ------------------------------------------------------------
    # Load sandbox
    # ------------------------------------------------------------

    sandbox = IsolationSandbox(
        analyzer=analyzer,
        investigation_delay_ms=10.0,
    )

    # ------------------------------------------------------------
    # Prepare demo memory & Recognition Policy
    # ------------------------------------------------------------

    if RESET_DEMO_MEMORY and DEMO_MEMORY_PATH.exists():

        print(
            "\n[3] Resetting demo immune memory..."
        )

        shutil.rmtree(
            DEMO_MEMORY_PATH
        )

    memory = ImmuneMemory(
        storage_path=DEMO_MEMORY_PATH,
        embedding_dim=analyzer.embedding_dim,
        known_threshold=KNOWN_THRESHOLD,
        near_threshold=NEAR_THRESHOLD,
        encoder_version=analyzer.encoder_version,
        feature_schema_hash=analyzer.feature_schema_hash,
    )

    print(
        f"    Existing memories: "
        f"{memory.count_memories()}"
    )

    recognition_policy = RecognitionPolicy()
    print(
        f"    Recognition policy: {recognition_policy.VERSION}"
    )
    print(
        f"    Known threshold: {recognition_policy.known_threshold}"
    )
    print(
        f"    Novel threshold: {recognition_policy.novel_threshold}"
    )

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    print("\n[4] Loading attack dataset...")

    df = pd.read_parquet(
        DATA_PATH
    )

    print(
        f"    Dataset rows: {len(df):,}"
    )

    # ------------------------------------------------------------
    # Identify attack flows
    # ------------------------------------------------------------

    label_series = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    attack_mask = (
        label_series != "benign"
    )

    attack_flows = (
        df.loc[attack_mask]
        .copy()
        .reset_index(drop=True)
    )

    if len(attack_flows) == 0:
        raise RuntimeError(
            "No attack flows found."
        )

    print(
        f"    Attack flows: {len(attack_flows):,}"
    )

    # ============================================================
    # EXPOSURE #1
    # ============================================================

    print_separator()

    print(
        "[5] EXPOSURE #1 — First encounter"
    )

    flow_1 = attack_flows.iloc[0]

    flow_1_df = flow_1[
        feature_cols
    ].to_frame().T

    # ------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------

    detector_start = time.perf_counter()

    detector_X = clean_for_detector(
        flow_1_df,
        feature_cols,
    )

    anomaly_score_1 = float(
        detector.predict_proba(
            detector_X
        )[:, 1][0]
    )

    detector_ms = (
        time.perf_counter()
        - detector_start
    ) * 1000.0

    print(
        f"    Detector score: "
        f"{anomaly_score_1:.6f}"
    )

    print(
        f"    Detector latency: "
        f"{detector_ms:.3f} ms"
    )

    # ------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------

    encode_start = time.perf_counter()

    antigen_1 = analyzer.analyze_and_extract(
        flow_data=flow_1,
        anomaly_score=anomaly_score_1,
        source_dataset=DATA_PATH.name,
    )

    encode_ms = (
        time.perf_counter()
        - encode_start
    ) * 1000.0

    print(
        f"    Behavior family: "
        f"{antigen_1.metadata['behavior_family']}"
    )

    print(
        f"    Embedding dimension: "
        f"{len(antigen_1.embedding)}"
    )

    print(
        f"    Encoder latency: "
        f"{encode_ms:.3f} ms"
    )

    # ------------------------------------------------------------
    # Immune memory recognition & Policy decision
    # ------------------------------------------------------------

    memory_start = time.perf_counter()

    recognition_1 = (
        memory.recognize_threat(
            antigen_1.embedding
        )
    )

    memory_ms = (
        time.perf_counter()
        - memory_start
    ) * 1000.0

    similarity_1 = recognition_1["similarity"]
    decision_1 = recognition_policy.classify(similarity_1)

    print(
        f"    Memory classification: "
        f"{recognition_1['classification']}"
    )

    print(
        f"    Similarity: "
        f"{similarity_1:.6f}"
    )

    print(
        f"    Memory latency: "
        f"{memory_ms:.3f} ms"
    )

    print(
        f"    Recognition policy: {decision_1.classification}"
    )
    print(
        f"    Policy confidence: {decision_1.confidence:.4f}"
    )

    # ------------------------------------------------------------
    # Immune response based on Policy
    # ------------------------------------------------------------

    if decision_1.classification == "KNOWN":

        print(
            "\n    ✓ IMMUNE MEMORY HIT"
        )
        print(
            "    → Sandbox bypassed."
        )
        print(
            "    → Fast-path response."
        )

        memory.record_reexposure(
            recognition_1
        )

    elif decision_1.classification == "UNCERTAIN":

        print(
            "\n    ? WEAK IMMUNE MEMORY MATCH"
        )
        print(
            "    → Investigation required."
        )
        print(
            "    → Sandbox enabled."
        )

        sandbox_start = time.perf_counter()

        investigated_antigen = (
            sandbox.investigate(
                flow_data=flow_1,
                anomaly_score=anomaly_score_1,
                source_dataset=DATA_PATH.name,
            )
        )

        sandbox_ms = (
            time.perf_counter()
            - sandbox_start
        ) * 1000.0

        print(
            f"    Investigation mode: "
            f"{investigated_antigen.metadata['investigation_mode']}"
        )

        print(
            f"    Validation status: "
            f"{investigated_antigen.metadata['validation_status']}"
        )

        print(
            f"    Investigation latency: "
            f"{sandbox_ms:.3f} ms"
        )

        commit_start = time.perf_counter()

        commit_result = (
            memory.commit_antigen(
                investigated_antigen
            )
        )

        commit_ms = (
            time.perf_counter()
            - commit_start
        ) * 1000.0

        print(
            "\n    → Antigen committed "
            "to immune memory."
        )

        print(
            f"    Commit status: "
            f"{commit_result['status']}"
        )

        print(
            f"    Memory ID: "
            f"{commit_result['memory_id']}"
        )

        print(
            f"    Commit latency: "
            f"{commit_ms:.3f} ms"
        )

    else:

        print(
            "\n    → Threat is novel."
        )
        print(
            "    → Sending to isolation chamber."
        )

        sandbox_start = time.perf_counter()

        investigated_antigen = (
            sandbox.investigate(
                flow_data=flow_1,
                anomaly_score=anomaly_score_1,
                source_dataset=DATA_PATH.name,
            )
        )

        sandbox_ms = (
            time.perf_counter()
            - sandbox_start
        ) * 1000.0

        print(
            f"    Investigation mode: "
            f"{investigated_antigen.metadata['investigation_mode']}"
        )

        print(
            f"    Validation status: "
            f"{investigated_antigen.metadata['validation_status']}"
        )

        print(
            f"    Investigation latency: "
            f"{sandbox_ms:.3f} ms"
        )

        commit_start = time.perf_counter()

        commit_result = (
            memory.commit_antigen(
                investigated_antigen
            )
        )

        commit_ms = (
            time.perf_counter()
            - commit_start
        ) * 1000.0

        print(
            "\n    → New antigen committed "
            "to immune memory."
        )

        print(
            f"    Commit status: "
            f"{commit_result['status']}"
        )

        print(
            f"    Memory ID: "
            f"{commit_result['memory_id']}"
        )

        print(
            f"    Commit latency: "
            f"{commit_ms:.3f} ms"
        )

    # ============================================================
    # EXPOSURE #2
    # ============================================================

    print_separator()

    print(
        "[6] EXPOSURE #2 — Re-exposure"
    )

    second_index = (
        5 if len(attack_flows) > 5
        else 1 if len(attack_flows) > 1
        else 0
    )

    flow_2 = attack_flows.iloc[
        second_index
    ]

    flow_2_df = flow_2[
        feature_cols
    ].to_frame().T

    # ------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------

    detector_start = time.perf_counter()

    detector_X_2 = clean_for_detector(
        flow_2_df,
        feature_cols,
    )

    anomaly_score_2 = float(
        detector.predict_proba(
            detector_X_2
        )[:, 1][0]
    )

    detector_ms_2 = (
        time.perf_counter()
        - detector_start
    ) * 1000.0

    print(
        f"    Detector score: "
        f"{anomaly_score_2:.6f}"
    )

    # ------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------

    encode_start = time.perf_counter()

    antigen_2 = analyzer.analyze_and_extract(
        flow_data=flow_2,
        anomaly_score=anomaly_score_2,
        source_dataset=DATA_PATH.name,
    )

    encode_ms_2 = (
        time.perf_counter()
        - encode_start
    ) * 1000.0

    print(
        f"    Behavior family: "
        f"{antigen_2.metadata['behavior_family']}"
    )

    # ------------------------------------------------------------
    # Memory recognition & Policy decision
    # ------------------------------------------------------------

    memory_start = time.perf_counter()

    recognition_2 = (
        memory.recognize_threat(
            antigen_2.embedding
        )
    )

    memory_ms_2 = (
        time.perf_counter()
        - memory_start
    ) * 1000.0

    similarity_2 = recognition_2["similarity"]
    decision_2 = recognition_policy.classify(similarity_2)

    print(
        f"    Memory classification: "
        f"{recognition_2['classification']}"
    )

    print(
        f"    Similarity: "
        f"{similarity_2:.6f}"
    )

    print(
        f"    Memory latency: "
        f"{memory_ms_2:.3f} ms"
    )

    print(
        f"    Recognition policy: {decision_2.classification}"
    )
    print(
        f"    Policy confidence: {decision_2.confidence:.4f}"
    )

    # ------------------------------------------------------------
    # Immune response based on Policy
    # ------------------------------------------------------------

    if decision_2.classification == "KNOWN":

        print(
            "\n    ✓ IMMUNE MEMORY HIT"
        )

        print(
            "    → Sandbox bypassed."
        )

        print(
            "    → Fast-path response."
        )

        memory.record_reexposure(
            recognition_2
        )

    elif decision_2.classification == "UNCERTAIN":

        print(
            "\n    ? WEAK IMMUNE MEMORY MATCH"
        )

        print(
            "    → Investigation required."
        )

        print(
            "    → Sandbox enabled."
        )

        sandbox_start = time.perf_counter()

        investigated_antigen_2 = (
            sandbox.investigate(
                flow_data=flow_2,
                anomaly_score=anomaly_score_2,
                source_dataset=DATA_PATH.name,
            )
        )

        sandbox_ms_2 = (
            time.perf_counter()
            - sandbox_start
        ) * 1000.0

        print(
            f"    Investigation latency: "
            f"{sandbox_ms_2:.3f} ms"
        )

        commit_result_2 = (
            memory.commit_antigen(
                investigated_antigen_2
            )
        )

        print(
            f"    Memory result: "
            f"{commit_result_2['status']}"
        )

    else:

        print(
            "\n    → Threat is novel."
        )

        print(
            "    → Sending to isolation chamber."
        )

        sandbox_start = time.perf_counter()

        investigated_antigen_2 = (
            sandbox.investigate(
                flow_data=flow_2,
                anomaly_score=anomaly_score_2,
                source_dataset=DATA_PATH.name,
            )
        )

        sandbox_ms_2 = (
            time.perf_counter()
            - sandbox_start
        ) * 1000.0

        print(
            f"    Investigation latency: "
            f"{sandbox_ms_2:.3f} ms"
        )

        commit_result_2 = (
            memory.commit_antigen(
                investigated_antigen_2
            )
        )

        print(
            f"    Memory result: "
            f"{commit_result_2['status']}"
        )

    # ============================================================
    # Final status
    # ============================================================

    print_separator()

    print(
        "[7] FINAL IMMUNE MEMORY STATUS"
    )

    print(
        f"    Memories stored: "
        f"{memory.count_memories()}"
    )

    print(
        f"    Encoder version: "
        f"{analyzer.encoder_version}"
    )

    print(
        f"    Feature schema hash: "
        f"{analyzer.feature_schema_hash}"
    )

    print_separator()

    print(
        "[ADIS] End-to-end immune loop completed."
    )

    print_separator()

    memory.close()


if __name__ == "__main__":
    main()