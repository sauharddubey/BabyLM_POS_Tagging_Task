import os
from pathlib import Path

# Paths
WORKSPACE_DIR = Path("/dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge")
RESULTS_DIR = WORKSPACE_DIR / "babylm-eval" / "strict" / "results"
OUTPUT_FILE = WORKSPACE_DIR / "reports" / "results_summary.md"

FINETUNE_METRIC = {
    "boolq":   "accuracy",
    "mnli":    "accuracy",
    "mrpc":    "f1",
    "multirc": "accuracy",
    "qqp":     "f1",
    "rte":     "accuracy",
    "wsc":     "accuracy",
}

def parse_average_score(report_path: Path) -> float | None:
    if not report_path.exists():
        return None
    try:
        lines = report_path.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("### AVERAGE"):
                if i + 1 < len(lines):
                    return float(lines[i + 1].strip())
    except Exception:
        pass
    return None

def parse_reading_scores(report_path: Path) -> dict[str, float]:
    scores = {}
    if not report_path.exists():
        return scores
    try:
        for line in report_path.read_text().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_").replace("-", "_")
                try:
                    scores[key] = float(val.strip())
                except ValueError:
                    pass
    except Exception:
        pass
    return scores

def parse_finetune_score(results_path: Path, metric: str) -> float | None:
    if not results_path.exists():
        return None
    try:
        for line in results_path.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == metric:
                return float(value.strip()) * 100
    except Exception:
        pass
    return None

def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory {RESULTS_DIR} does not exist.")
        return

    models = sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()])
    
    # Store all data
    data = {m: {} for m in models}
    
    zeroshot_paths = {
        "BLiMP (filtered)": "main/zero_shot/mlm/blimp/blimp_filtered/best_temperature_report.txt",
        "BLiMP Supplement (filtered)": "main/zero_shot/mlm/blimp/supplement_filtered/best_temperature_report.txt",
        "COMPS": "main/zero_shot/mlm/comps/comps/best_temperature_report.txt",
        "Entity Tracking": "main/zero_shot/mlm/entity_tracking/entity_tracking/best_temperature_report.txt",
    }
    
    reading_path = "main/zero_shot/mlm/reading/report.txt"
    
    for model in models:
        m_dir = RESULTS_DIR / model
        
        # Zero-shot
        for label, rel_path in zeroshot_paths.items():
            score = parse_average_score(m_dir / rel_path)
            if score is not None:
                data[model][label] = score
                
        # Reading
        r_scores = parse_reading_scores(m_dir / reading_path)
        if "eye_tracking_score" in r_scores:
            data[model]["Reading (Eye Tracking)"] = r_scores["eye_tracking_score"]
        if "self_paced_reading_score" in r_scores:
            data[model]["Reading (Self-Paced)"] = r_scores["self_paced_reading_score"]
            
        # Finetuning
        for task, metric in FINETUNE_METRIC.items():
            f_path = m_dir / "main" / "finetune" / task / "results.txt"
            score = parse_finetune_score(f_path, metric)
            if score is not None:
                data[model][f"GLUE: {task} ({metric})"] = score

    lines = []
    lines.append("# BabyLM Evaluation Results Summary")
    lines.append("\nThis file summarizes the evaluation results across all models for both **Zero-shot** and **Finetuning (GLUE)** tasks.")
    
    zeroshot_accuracy_tasks = [
        "BLiMP (filtered)",
        "BLiMP Supplement (filtered)",
        "COMPS",
        "Entity Tracking"
    ]
    
    zeroshot_reading_tasks = [
        "Reading (Eye Tracking)",
        "Reading (Self-Paced)"
    ]
    
    finetune_tasks = [f"GLUE: {task} ({metric})" for task, metric in FINETUNE_METRIC.items()]
    
    def render_table(tasks, title):
        table_lines = []
        table_lines.append(f"## {title}\n")
        
        # Table Header
        header = ["Task"] + models
        table_lines.append("| " + " | ".join(header) + " |")
        table_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # Table Rows
        for task in tasks:
            if not any(task in data[m] for m in models):
                continue
            
            valid_scores = [data[m][task] for m in models if task in data[m]]
            best_score = max(valid_scores) if valid_scores else None
            
            row = [task]
            for m in models:
                val = data[m].get(task)
                if val is None:
                    row.append("")
                elif val == best_score:
                    row.append(f"**{val:.2f}**")
                else:
                    row.append(f"{val:.2f}")
            table_lines.append("| " + " | ".join(row) + " |")
        return "\n".join(table_lines)

    lines.append("\n" + render_table(zeroshot_accuracy_tasks + zeroshot_reading_tasks, "Zero-Shot Results"))
    lines.append("\n" + render_table(finetune_tasks, "Finetuning Results (GLUE)"))
    
    # Summary of averages
    lines.append("\n## Macro-Average Performance by Category")
    lines.append("\n| Model | Zero-Shot Accuracy Avg | Zero-Shot Reading Avg | GLUE Avg |")
    lines.append("| --- | --- | --- | --- |")
    
    model_averages = {}
    for m in models:
        zs_acc = [data[m][t] for t in zeroshot_accuracy_tasks if t in data[m]]
        zs_read = [data[m][t] for t in zeroshot_reading_tasks if t in data[m]]
        glue_scores = [data[m][t] for t in finetune_tasks if t in data[m]]
        
        zs_acc_avg = sum(zs_acc) / len(zs_acc) if zs_acc else 0.0
        zs_read_avg = sum(zs_read) / len(zs_read) if zs_read else 0.0
        glue_avg = sum(glue_scores) / len(glue_scores) if glue_scores else 0.0
        
        model_averages[m] = (zs_acc_avg, zs_read_avg, glue_avg)
        
    best_zs_acc = max(avg[0] for avg in model_averages.values())
    best_zs_read = max(avg[1] for avg in model_averages.values())
    best_glue = max(avg[2] for avg in model_averages.values())
    
    for m in models:
        zs_acc_avg, zs_read_avg, glue_avg = model_averages[m]
        zs_acc_str = f"**{zs_acc_avg:.2f}**" if zs_acc_avg == best_zs_acc else f"{zs_acc_avg:.2f}"
        zs_read_str = f"**{zs_read_avg:.2f}**" if zs_read_avg == best_zs_read else f"{zs_read_avg:.2f}"
        glue_str = f"**{glue_avg:.2f}**" if glue_avg == best_glue else f"{glue_avg:.2f}"
        lines.append(f"| {m} | {zs_acc_str} | {zs_read_str} | {glue_str} |")

    OUTPUT_FILE.write_text("\n".join(lines) + "\n")
    print(f"Summary written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
