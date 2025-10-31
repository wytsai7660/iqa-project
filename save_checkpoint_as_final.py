#!/usr/bin/env python3
"""
Quick script to save a checkpoint as a complete model with tokenizer.
Usage: python save_checkpoint_as_final.py <checkpoint_path> <output_path>
"""

import sys
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

def main():
    if len(sys.argv) != 3:
        print("Usage: python save_checkpoint_as_final.py <checkpoint_path> <output_path>")
        print("Example: python save_checkpoint_as_final.py outputs/10281300/03_quality/checkpoint-600 outputs/10281300/final_model")
        sys.exit(1)
    
    checkpoint_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    base_model = "src/owl3"  # Base model with config and tokenizer
    
    print(f"📂 Checkpoint: {checkpoint_path}")
    print(f"🔧 Base model: {base_model}")
    print(f"📦 Output: {output_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy model weights
    print("\n1️⃣ Copying model weights...")
    model_file = checkpoint_path / "model.safetensors"
    if model_file.exists():
        shutil.copy2(model_file, output_path / "model.safetensors")
        print(f"   ✅ Copied model.safetensors")
    else:
        print(f"   ❌ model.safetensors not found!")
        sys.exit(1)
    
    # Copy config from base model
    print("\n2️⃣ Copying config from base model...")
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    config.save_pretrained(output_path)
    print(f"   ✅ Saved config.json")
    
    # Copy tokenizer from base model
    print("\n3️⃣ Copying tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    print(f"   ✅ Saved tokenizer files")
    
    # Copy other necessary files from base model
    print("\n4️⃣ Copying other necessary files from base model...")
    base_model_path = Path(base_model)
    files_to_copy = [
        "configuration_mplugowl3.py",
        "modeling_mplugowl3.py",
        "processing_mplugowl3.py",
        "image_processing_mplugowl3.py",
        "configuration_hyper_qwen2.py",
        "modeling_hyper_qwen2.py",
        "generation_config.json",
    ]
    
    for filename in files_to_copy:
        src_file = base_model_path / filename
        if src_file.exists():
            shutil.copy2(src_file, output_path / filename)
            print(f"   ✅ Copied {filename}")
    
    print(f"\n✅ Complete model saved successfully to: {output_path}")
    print(f"\n🎯 Now you can evaluate with:")
    print(f"   uv run -m eval_sequential_model --model_path {output_path} --dataset_paths datasets/koniq-10k/ --split testing")

if __name__ == "__main__":
    main()
