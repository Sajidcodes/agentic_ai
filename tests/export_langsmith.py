"""
Export all queries from LangSmith
"""

from langsmith import Client
from datetime import datetime, timedelta
import json
import csv
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

client = Client()
PROJECT_NAME = "socrates"  # Your project name

def export_runs(days: int = 7):
    """Export recent runs to CSV and JSON"""
    
    print(f"📥 Fetching runs from last {days} days...")
    
    runs = list(client.list_runs(
        project_name=PROJECT_NAME,
        start_time=datetime.now() - timedelta(days=days),
        limit=100
    ))
    
    print(f"Found {len(runs)} runs")
    
    # Extract data
    data = []
    for run in runs:
        data.append({
            "timestamp": str(run.start_time),
            "name": run.name,
            "input": str(run.inputs)[:500] if run.inputs else "",
            "output": str(run.outputs)[:500] if run.outputs else "",
            "latency_s": run.latency,
            "status": run.status,
            "error": run.error
        })
    
    # Save as JSON
    with open("langsmith_export.json", "w") as f:
        json.dump(data, f, indent=2, default=str)
    print("✅ Saved to langsmith_export.json")
    
    # Save as CSV
    with open("langsmith_export.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print("✅ Saved to langsmith_export.csv")
    
    # Print sample
    print(f"\n📝 Sample queries:")
    for row in data[:5]:
        print(f"  - {row['input'][:60]}...")
    
    return data


if __name__ == "__main__":
    export_runs(days=7)