from src.recognition_policy import RecognitionPolicy


def main():
    print("=" * 75)
    print("[ADIS] M4 — Recognition Policy v1 Test")
    print("=" * 75)

    policy = RecognitionPolicy()

    print("\nPolicy:")
    print(f"  Version:          {policy.VERSION}")
    print(f"  Novel threshold:  {policy.novel_threshold}")
    print(f"  Known threshold:  {policy.known_threshold}")

    test_values = [
        0.10,
        0.50,
        0.74,
        0.75,
        0.80,
        0.85,
        0.89,
        0.90,
        0.91,
        0.92,
        0.95,
        0.9999,
        1.0,
    ]

    print("\nDecisions:")
    print("-" * 75)

    for similarity in test_values:
        decision = policy.classify(similarity)

        print(
            f"Similarity={similarity:7.4f} | "
            f"{decision.classification:9s} | "
            f"Action={decision.action:12s} | "
            f"Confidence={decision.confidence:.4f}"
        )

    print("\nBoundary assertions...")

    assert policy.classify(0.7499).classification == "NOVEL"
    assert policy.classify(0.75).classification == "UNCERTAIN"
    assert policy.classify(0.9199).classification == "UNCERTAIN"
    assert policy.classify(0.92).classification == "KNOWN"
    assert policy.classify(1.0).classification == "KNOWN"

    print("✓ All boundary tests passed.")

    print("\n" + "=" * 75)
    print("[ADIS] Recognition Policy v1 PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()