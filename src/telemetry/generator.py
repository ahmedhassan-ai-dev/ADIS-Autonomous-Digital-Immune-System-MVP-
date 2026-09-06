import random
from datetime import datetime, timedelta
import pandas as pd


# Normal Behavior Configuration

NORMAL_PROCESSES = [
    ("chrome", "explorer"),
    ("firefox", "explorer"),
    ("system_update", "systemd"),
    ("file_manager", "explorer"),
    ("ssh", "systemd"),
]

NORMAL_PORTS = [22, 53, 80, 443]


def generate_normal_event(timestamp=None):
    """
    Generate one synthetic normal system telemetry event.
    """

    if timestamp is None:
        timestamp = datetime.now()

    process_name, parent_process = random.choice(NORMAL_PROCESSES)

    return {
        "timestamp": timestamp,
        "process_name": process_name,
        "parent_process": parent_process,
        "cpu_usage": round(random.uniform(1, 35), 2),
        "memory_usage": round(random.uniform(10, 60), 2),
        "network_connection": random.choice([0, 1]),
        "destination_port": random.choice(NORMAL_PORTS),
        "failed_logins": random.randint(0, 1),
        "file_operations": random.randint(0, 5),
        "process_spawn_count": random.randint(0, 3),
        "label": "normal",
    }


# SSH Brute Force

def generate_ssh_bruteforce_event(timestamp=None):
    """
    Generate one synthetic SSH brute-force telemetry event.

    This simulates suspicious behavior without executing
    any real attack.
    """

    if timestamp is None:
        timestamp = datetime.now()

    return {
        "timestamp": timestamp,
        "process_name": "sshd",
        "parent_process": "systemd",
        "cpu_usage": round(random.uniform(10, 45), 2),
        "memory_usage": round(random.uniform(20, 70), 2),
        "network_connection": 1,
        "destination_port": 22,
        "failed_logins": random.randint(15, 50),
        "file_operations": random.randint(0, 2),
        "process_spawn_count": random.randint(1, 4),
        "label": "ssh_bruteforce",
    }



# SSH Brute Force Variant


def generate_ssh_bruteforce_variant(timestamp=None):
    """
    Generate a behavioral variant of an SSH brute-force event.

    The values are intentionally different from the original
    scenario while preserving similar behavioral characteristics.
    """

    if timestamp is None:
        timestamp = datetime.now()

    return {
        "timestamp": timestamp,
        "process_name": "sshd",
        "parent_process": "systemd",
        "cpu_usage": round(random.uniform(8, 40), 2),
        "memory_usage": round(random.uniform(18, 65), 2),
        "network_connection": 1,
        "destination_port": 22,
        "failed_logins": random.randint(10, 40),
        "file_operations": random.randint(0, 3),
        "process_spawn_count": random.randint(1, 5),
        "label": "ssh_bruteforce_variant",
    }



# Dataset Generator


def generate_dataset(
    normal_count=1000,
    attack_count=50,
    variant_count=50,
    seed=42,
):
    """
    Generate the complete synthetic telemetry dataset.

    Returns:
        pandas.DataFrame
    """

    random.seed(seed)

    events = []

    start_time = datetime.now()

    # Normal telemetry
    for i in range(normal_count):
        timestamp = start_time + timedelta(seconds=i)
        events.append(generate_normal_event(timestamp))

    # First exposure
    for i in range(attack_count):
        timestamp = start_time + timedelta(seconds=normal_count + i)
        events.append(generate_ssh_bruteforce_event(timestamp))

    # Subsequent exposure / variant
    for i in range(variant_count):
        timestamp = start_time + timedelta(
            seconds=normal_count + attack_count + i
        )
        events.append(generate_ssh_bruteforce_variant(timestamp))

    df = pd.DataFrame(events)

    return df



# Standalone Test

if __name__ == "__main__":
    df = generate_dataset()

    print("[ADIS] Synthetic telemetry generated successfully.")
    print()
    print(df.head())
    print()
    print("Dataset shape:", df.shape)
    print()
    print("Label distribution:")
    print(df["label"].value_counts())