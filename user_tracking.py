import json
import os
import hashlib
from datetime import datetime

COUNTER_FILE = "user_stats.json"


def load_stats():
    if not os.path.exists(COUNTER_FILE):
        return {"unique_users": set(), "visits": 0, "dates": {}}
    with open(COUNTER_FILE, "r") as f:
        data = json.load(f)
        # convert back to set
        data["unique_users"] = set(data["unique_users"])
        return data


def save_stats(stats):
    # convert set to list for JSON
    stats_copy = stats.copy()
    stats_copy["unique_users"] = list(stats_copy["unique_users"])
    with open(COUNTER_FILE, "w") as f:
        json.dump(stats_copy, f, indent=2)


def track_user(ip):
    stats = load_stats()
    hashed = hashlib.sha256(ip.encode()).hexdigest()

    stats["visits"] += 1
    stats["unique_users"].add(hashed)

    today = datetime.now().strftime("%Y-%m-%d")
    stats["dates"][today] = stats["dates"].get(today, 0) + 1

    save_stats(stats)
    return stats
