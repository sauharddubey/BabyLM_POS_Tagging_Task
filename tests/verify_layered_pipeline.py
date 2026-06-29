import os
import sys
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertConfig

# Add project root to path so src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import PretrainDataset, MultiTaskCollator
from src.models import LayeredPOSMLMBert

def run_layered_tests():
    print("=== STARTING LAYERED PIPELINE VERIFICATION TESTS ===")
    
    # 1. Initialize Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    vocab_size = len(tokenizer)
    
    # 2. Mock POS map and data
    mock_pos_to_id = {'NN': 0, 'VB': 1, 'JJ': 2, 'PRP': 3, 'IN': 4, '.': 5}
    num_pos_tags = len(mock_pos_to_id)
    
    # Create mock dataset
    mock_data = [
        {
            "words": ["the", "cat", "sat", "on", "the", "mat", "."],
            "pos_labels": [2, 0, 1, 4, 2, 0, 5]
        },
        {
            "words": ["she", "likes", "happy", "cats", "."],
            "pos_labels": [3, 1, 2, 0, 5]
        }
    ]
    
    # PretrainDataset and Collator
    dataset = PretrainDataset(mock_data, nsp_active=True)
    collator = MultiTaskCollator(tokenizer, mlm_active=True, mlm_probability=0.15)
    batch = collator([dataset[0], dataset[1]])
    
    batch_size, seq_len = batch["input_ids"].shape
    
    # 3. Create dummy pos_to_vocab.json for test purposes, or verify loaded weights
    # We will save a test mapping to checkpoints/test_pos_to_vocab.json
    os.makedirs("checkpoints", exist_ok=True)
    test_vocab_map_path = "checkpoints/test_pos_to_vocab.json"
    
    # Assign some vocab ids to each POS
    # e.g., NN gets tokens 1000-1100, etc.
    test_map = {}
    for pos_id in range(num_pos_tags):
        # Allow special tokens
        tokens = list(tokenizer.all_special_ids)
        # Allow a unique range
        tokens.extend(list(range(1000 + pos_id*50, 1050 + pos_id*50)))
        test_map[str(pos_id)] = tokens
        
    import json
    with open(test_vocab_map_path, "w") as f:
        json.dump(test_map, f)
        
    # 4. Initialize model configuration (BERT-Mini scale)
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # 5. Run tests for all three masking options
    mask_types = ["soft", "hard", "gold"]
    
    for mask_type in mask_types:
        print(f"\n--- Testing Mask Type: {mask_type.upper()} ---")
        model = LayeredPOSMLMBert(config, num_pos_tags=num_pos_tags, pos_to_vocab_path=test_vocab_map_path)
        
        # Verify model initialization loaded the mapping
        assert hasattr(model, "pos_to_vocab_mask")
        assert model.pos_to_vocab_mask.shape == (num_pos_tags, vocab_size)
        
        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            next_sentence_label=batch["next_sentence_label"],
            word_ids=batch["word_ids"],
            pos_labels=batch["pos_labels"],
            tasks=["mlm", "nsp", "pos"],
            alpha=1.0,
            mask_type=mask_type
        )
        
        assert "loss" in outputs
        assert "mlm_loss" in outputs
        assert "pos_loss" in outputs
        assert "nsp_loss" in outputs
        assert "mlm_logits" in outputs
        assert "pos_logits" in outputs
        
        # Assert loss is greater than 0
        assert outputs["loss"].item() > 0.0
        assert outputs["mlm_loss"] > 0.0
        assert outputs["pos_loss"] > 0.0
        assert outputs["nsp_loss"] > 0.0
        
        # Verify shape of logits
        assert outputs["mlm_logits"].shape == (batch_size, seq_len, vocab_size)
        assert outputs["pos_logits"].shape == (batch_size, seq_len, num_pos_tags)
        
        # Test backpropagation
        loss = outputs["loss"]
        loss.backward()
        
        # Verify gradients are populated for both classification heads and base model
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient for mask_type={mask_type}!"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient for mask_type={mask_type}!"
                
        print(f"Mask Type {mask_type.upper()} Passed: Forward pass, output shapes, and backward pass are fully correct.")
        
    # Clean up test file
    if os.path.exists(test_vocab_map_path):
        os.remove(test_vocab_map_path)
        
    print("\n=== ALL LAYERED PIPELINE TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_layered_tests()
