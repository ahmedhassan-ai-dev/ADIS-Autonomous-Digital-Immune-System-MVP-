from pathlib import Path
import time
import joblib
import pandas as pd
import numpy as np

from src.analysis.threat_analyzer import BehavioralThreatAnalyzer
from src.sandbox.isolation_chamber import IsolationSandbox
from src.memory.immune_memory import ImmuneMemory

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "merged_binary_detector.joblib"
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "cicids2017" / "Portscan-Friday-no-metadata.parquet"

print("=" * 80)
print("[ADIS] DEMONSTRATING FULL END-TO-END IMMUNE LOOP (M3)")
print("=" * 80)

# 1. Load Model & Assets
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]

# 2. Load Traffic Data
df = pd.read_parquet(DATA_FILE).replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))
attack_flows = df[df["Label"].astype(str).str.lower().ne("benign")].reset_index(drop=True)

# 3. Initialize Modules
ENCODER_PATH = (
    PROJECT_ROOT
    / "models"
    / "behavioral_encoder.joblib"
)

analyzer = BehavioralThreatAnalyzer(
    encoder_path=str(ENCODER_PATH)
)


sandbox = IsolationSandbox(analyzer=analyzer)


MEMORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "immune_memory"
)

memory = ImmuneMemory(
    storage_path=str(MEMORY_PATH),
    embedding_dim=32,
)
# =============================================================
# EXPOSURE 1: Unseen Threat -> Detection -> Sandbox -> Commit
# =============================================================
print("\n[EXPOSURE 1] Detecting Unseen Threat...")
exp1_df = pd.DataFrame([attack_flows.iloc[0][features]])

t0 = time.perf_counter()
prob_exp1 = model.predict_proba(exp1_df)[:, 1][0]
t_detect1 = (time.perf_counter() - t0) * 1000

print(f"  + Innate Anomaly Score: {prob_exp1:.4f} (Flagged in {t_detect1:.2f} ms)")

exp1_vector = analyzer.generate_antigen_vector(exp1_df.to_numpy())
rec_exp1 = memory.recognize_threat(exp1_vector, similarity_threshold=0.85)
matched_1 = rec_exp1 is not None and rec_exp1.get("matched", False)
print(f"  + Immune Memory Search: Matched={matched_1} (Novel Threat Detected)")

print("  + Routing to Sandbox for Deeper Behavioral Inspection...")
t_sand_start = time.perf_counter()
antigen = sandbox.investigate(
    event_id="THREAT-001",
    flow_dict=attack_flows.iloc[0].to_dict(),
    raw_features=exp1_df.to_numpy(),
    anomaly_score=prob_exp1,
)
t_sand = (time.perf_counter() - t_sand_start) * 1000

print(f"  + MITRE ATT&CK Identified: {antigen.technique_id} ({antigen.technique_name}) | Tactic: {antigen.tactic}")
print(f"  + Investigation Latency: {t_sand:.2f} ms")

# Commit to Qdrant
memory.commit_antigen(antigen)
print(f"  + Committed to Qdrant Immune Memory. Total Signatures: {memory.count_memories()}")

# =============================================================
# EXPOSURE 2: Re-exposure -> Immediate Recognition -> Containment
# =============================================================
print("\n" + "-" * 80)
print("[EXPOSURE 2] Re-exposure to Related Threat Vector...")
exp2_df = pd.DataFrame([attack_flows.iloc[5][features]])

t0 = time.perf_counter()
prob_exp2 = model.predict_proba(exp2_df)[:, 1][0]
t_detect2 = (time.perf_counter() - t0) * 1000
print(f"  + Innate Anomaly Score: {prob_exp2:.4f} (Flagged in {t_detect2:.2f} ms)")

t_mem_start = time.perf_counter()
exp2_vector = analyzer.generate_antigen_vector(exp2_df.to_numpy())
rec_exp2 = memory.recognize_threat(exp2_vector, similarity_threshold=0.85)
t_mem = (time.perf_counter() - t_mem_start) * 1000

print(f"  + Immune Memory Lookup Latency: {t_mem:.2f} ms")

if rec_exp2 and rec_exp2.get("matched", False):
    print(f"  + IMMUNE RECOGNITION HIT! Similarity: {rec_exp2['similarity_score'] * 100:.2f}%")
    print(f"  + Recalled Technique: {rec_exp2['payload']['technique_id']} - {rec_exp2['payload']['technique_name']}")
    print(f"  + ACTION: Autonomous Instant Containment (Skipping Sandbox!)")
    
    total_t1 = t_detect1 + t_sand
    total_t2 = t_detect2 + t_mem
    speedup = total_t1 / max(total_t2, 0.01)
    print(f"  + Operational Speedup: {speedup:.1f}x Faster Defense Loop ({total_t1:.2f} ms -> {total_t2:.2f} ms)")
else:
    sim = rec_exp2["similarity_score"] if rec_exp2 else 0.0
    print(f"  + Partial Match (Similarity: {sim * 100:.2f}%). Re-investigating...")

print("\n[ADIS] End-to-End Loop Demonstration Finished.")