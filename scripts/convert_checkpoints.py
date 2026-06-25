import os
import torch
from transformers import BertTokenizerFast, BertConfig, BertModel

def main():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
        classifier_dropout=0.1
    )
    
    checkpoints = {
        "BASEBERT": "./checkpoints/basebert/checkpoint_best.pt",
        "POSBERT": "./checkpoints/posbert/checkpoint_best.pt",
        "MLMPOSBert": "./checkpoints/mlmposbert/checkpoint_best.pt",
        "NSPPOSBERT": "./checkpoints/nspposbert/checkpoint_best.pt",
        "MLM_POS_NSPBERT": "./checkpoints/mlm_pos_nspbert/checkpoint_best.pt",
        "MLM_ONLY": "./checkpoints/mlm_only/checkpoint_best.pt",
        "MLMPOS_alpha02": "./checkpoints/mlmpos_alpha02/checkpoint_best.pt",
        "MLMPOSNSP_alpha02": "./checkpoints/mlmposnsp_alpha02/checkpoint_best.pt"
    }
    
    output_base = "./hf_models"
    os.makedirs(output_base, exist_ok=True)
    
    for name, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: Checkpoint not found at {path}")
            continue
            
        print(f"\nConverting {name} from {path}...")
        try:
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint['model_state_dict']
            
            # Extract weights belonging to the base bert encoder
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("bert."):
                    cleaned_state_dict[k[5:]] = v
            
            # Initialize a standard HF BertModel and load weights
            hf_model = BertModel(config)
            hf_model.load_state_dict(cleaned_state_dict, strict=True)
            
            # Save Hugging Face model and tokenizer files
            out_dir = os.path.join(output_base, name)
            os.makedirs(out_dir, exist_ok=True)
            hf_model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"Successfully saved Hugging Face model for {name} to {out_dir}")
            
        except Exception as e:
            print(f"Error converting {name}: {e}")

if __name__ == "__main__":
    main()
