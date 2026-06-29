# Orchestrator script for the BabyLM pretraining and evaluation pipeline.

import os
import sys
import argparse
import subprocess
import json

MODELS = {
    "BASEBERT": {"tasks": "mlm,nsp", "dir": "./checkpoints/basebert", "loss": "$L_{MLM} + L_{NSP}$"},
    "POSBERT": {"tasks": "pos", "dir": "./checkpoints/posbert", "loss": "$\\alpha L_{POS}$"},
    "MLMPOSBert": {"tasks": "mlm,pos", "dir": "./checkpoints/mlmposbert", "loss": "$L_{MLM} + \\alpha L_{POS}$"},
    "NSPPOSBERT": {"tasks": "nsp,pos", "dir": "./checkpoints/nspposbert", "loss": "$L_{NSP} + \\alpha L_{POS}$"},
    "MLM_POS_NSPBERT": {"tasks": "mlm,nsp,pos", "dir": "./checkpoints/mlm_pos_nspbert", "loss": "$L_{MLM} + L_{NSP} + \\alpha L_{POS}$"},
    "MLM_ONLY": {"tasks": "mlm", "dir": "./checkpoints/mlm_only", "loss": "$L_{MLM}$"},
    "MLMPOS_alpha02": {"tasks": "mlm,pos", "dir": "./checkpoints/mlmpos_alpha02", "loss": "$L_{MLM} + 0.2 L_{POS}$", "alpha": 0.2},
    "MLMPOSNSP_alpha02": {"tasks": "mlm,nsp,pos", "dir": "./checkpoints/mlmposnsp_alpha02", "loss": "$L_{MLM} + L_{NSP} + 0.2 L_{POS}$", "alpha": 0.2},
    "LAYERED_POS_MLM_SOFT": {"tasks": "mlm,pos", "dir": "./checkpoints/layered_pos_mlm_soft", "loss": "$L_{MLM} + \\alpha L_{POS}$ (Soft)", "model_type": "LayeredPOSMLMBert", "mask_type": "soft"},
    "LAYERED_POS_MLM_HARD": {"tasks": "mlm,pos", "dir": "./checkpoints/layered_pos_mlm_hard", "loss": "$L_{MLM} + \\alpha L_{POS}$ (Hard)", "model_type": "LayeredPOSMLMBert", "mask_type": "hard"},
    "LAYERED_POS_MLM_GOLD": {"tasks": "mlm,pos", "dir": "./checkpoints/layered_pos_mlm_gold", "loss": "$L_{MLM} + \\alpha L_{POS}$ (Gold)", "model_type": "LayeredPOSMLMBert", "mask_type": "gold"}
}

def run_cmd(cmd, description):
    print(f"\n>>> Running: {description}...")
    print(f"Command: {' '.join(cmd)}")
    # Flush stdout to ensure output order
    sys.stdout.flush()
    result = subprocess.run(cmd, check=True)
    return result

def main():
    parser = argparse.ArgumentParser(description="BabyLM Multi-task BERT Pretraining and Evaluation Suite")
    parser.add_argument("--stage", type=str, default="all", choices=["preprocess", "train", "eval", "report", "all"],
                        help="Pipeline stage to run: preprocess, train, eval, report, or all (default)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for POS loss")
    args = parser.parse_args()

    print("==========================================================")
    print("      BABYLM MULTI-TASK PRETRAINING & EVALUATION SUITE    ")
    print("==========================================================\n")

    # Stage 1: Preprocessing
    if args.stage in ["preprocess", "all"]:
        if not os.path.exists("./data/preprocessed_train") or not os.path.exists("./data/eval_data"):
            run_cmd([sys.executable, "preprocess.py"], "Dataset Preprocessing and Evaluation Caching")
        else:
            print("Preprocessed data already exists. Skipping preprocessing.")

    # Stage 2: Pretraining
    if args.stage in ["train", "all"]:
        for model_name, cfg in MODELS.items():
            print(f"\n==============================================")
            print(f"   PRETRAINING MODEL: {model_name}")
            print(f"==============================================")
            
            # Pass configurations as command line arguments directly to train.py
            alpha_val = cfg.get("alpha", args.alpha)
            train_cmd = [
                sys.executable, 
                "-u",
                "train.py", 
                "--tasks", cfg["tasks"],
                "--model_dir", cfg["dir"],
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--alpha", str(alpha_val),
                "--model_type", cfg.get("model_type", "MultiTaskBERT"),
                "--mask_type", cfg.get("mask_type", "soft"),
                "--resume"
            ]
            run_cmd(train_cmd, f"Pretraining {model_name}")

    # Stage 3: Evaluation
    if args.stage in ["eval", "all"]:
        for model_name, cfg in MODELS.items():
            chk_dir = cfg["dir"]
            
            # Check if evaluation results already exist
            eval_path = os.path.join(chk_dir, "checkpoint_best_eval_results.json")
            if os.path.exists(eval_path):
                print(f"Offline evaluation for {model_name} already exists. Skipping.")
                continue
                
            print(f"\n==============================================")
            print(f"   EVALUATING MODEL: {model_name}")
            print(f"==============================================")
            
            checkpoint_path = os.path.join(chk_dir, "checkpoint_best.pt")
            
            # If best checkpoint doesn't exist, try latest
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(chk_dir, "checkpoint_latest.pt")
                
            if not os.path.exists(checkpoint_path):
                print(f"Warning: No checkpoint found for {model_name} in {chk_dir}. Skipping evaluation.")
                continue
                
            eval_cmd = [
                sys.executable,
                "-u",
                "eval.py",
                "--checkpoint_path", checkpoint_path,
                "--tasks", "perplexity,blimp,glue,bleu",
                "--model_type", cfg.get("model_type", "MultiTaskBERT"),
                "--mask_type", cfg.get("mask_type", "soft")
            ]
            run_cmd(eval_cmd, f"Running Offline Evaluation Suite for {model_name}")

    # Stage 4: Compile Report
    if args.stage in ["report", "eval", "all"]:
        print("\n==============================================")
        print("   COMPILING EVALUATION COMPARATIVE REPORT    ")
        print("==============================================")
        
        results = {}
        for model_name, cfg in MODELS.items():
            chk_dir = cfg["dir"]
            # Try to read best first
            eval_path = os.path.join(chk_dir, "checkpoint_best_eval_results.json")
            if not os.path.exists(eval_path):
                eval_path = os.path.join(chk_dir, "checkpoint_latest_eval_results.json")
                
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    results[model_name] = json.load(f)
            else:
                print(f"No evaluation results found for {model_name}.")
                results[model_name] = {}
                
        # Build Markdown Table
        headers = [
            "Model Variant", 
            "Loss Calculation",
            "BabyLM PPL ⬇️", 
            "WikiText-2 PPL ⬇️", 
            "BLiMP SVA Acc ⬆️", 
            "BLiMP PG Acc ⬆️",
            "GLUE MRPC Acc ⬆️",
            "GLUE MRPC F1 ⬆️",
            "GLUE CoLA MCC ⬆️",
            "GLUE SST-2 Acc ⬆️",
            "Multi30k BLEU ⬆️"
        ]
        
        table_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]
        
        for model_name, cfg in MODELS.items():
            r = results[model_name]
            loss_calc = cfg["loss"]
            
            # Extract metrics with fallbacks
            babylm_ppl = f"{r.get('babylm_perplexity', float('inf')):.2f}" if 'babylm_perplexity' in r else "N/A"
            wt_ppl = f"{r.get('wikitext2_perplexity', float('inf')):.2f}" if 'wikitext2_perplexity' in r else "N/A"
            blimp_sva = f"{r.get('blimp_regular_plural_subject_verb_agreement_1_accuracy', 0.0):.2f}%" if 'blimp_regular_plural_subject_verb_agreement_1_accuracy' in r else "N/A"
            blimp_pg = f"{r.get('blimp_anaphor_gender_agreement_accuracy', 0.0):.2f}%" if 'blimp_anaphor_gender_agreement_accuracy' in r else "N/A"
            mrpc_acc = f"{r.get('glue_mrpc_accuracy', 0.0):.4f}" if 'glue_mrpc_accuracy' in r else "N/A"
            mrpc_f1 = f"{r.get('glue_mrpc_f1', 0.0):.4f}" if 'glue_mrpc_f1' in r else "N/A"
            cola_mcc = f"{r.get('glue_cola_mcc', 0.0):.4f}" if 'glue_cola_mcc' in r else "N/A"
            sst2_acc = f"{r.get('glue_sst2_accuracy', 0.0):.4f}" if 'glue_sst2_accuracy' in r else "N/A"
            bleu = f"{r.get('multi30k_bleu', 0.0):.2f}" if 'multi30k_bleu' in r else "N/A"
            
            row = [
                model_name,
                loss_calc,
                babylm_ppl,
                wt_ppl,
                blimp_sva,
                blimp_pg,
                mrpc_acc,
                mrpc_f1,
                cola_mcc,
                sst2_acc,
                bleu
            ]
            table_lines.append("| " + " | ".join(row) + " |")
            
        markdown_table = "\n".join(table_lines)
        
        report_content = f"""# BabyLM Multi-Task pretraining Evaluation Report

This report compares five BERT-Mini models trained on the `BabyLM-2026-Strict-Small` dataset (10M words) with different pretraining tasks.

## Objectives Configuration:
1. **BASEBERT**: MLM + NSP
2. **POSBERT**: POS only
3. **MLMPOSBert**: MLM + POS
4. **NSPPOSBERT**: NSP + POS
5. **MLM_POS_NSPBERT**: MLM + NSP + POS

## Comparative Results Table

{markdown_table}

*Notes:*
- **PPL (Perplexity)**: Lower is better. Computed using pseudo-perplexity from masked tokens.
- **Accuracy / F1 / MCC / BLEU**: Higher is better.
- **BLiMP**: Measured zero-shot grammatical accuracy on minimal pairs.
- **GLUE tasks**: Fine-tuned for 3 epochs.
- **BLEU**: Fine-tuned an Encoder-Decoder model on Multi30k for 3 epochs.
"""
        
        print("\n=== Final Pretraining & Evaluation Report ===")
        print(markdown_table)
        print("==============================================\n")
        
        # Save to disk
        report_path = "docs/evaluation_report.md"
        with open(report_path, "w") as f:
            f.write(report_content)
        print(f"Saved report to {report_path}")

if __name__ == "__main__":
    main()