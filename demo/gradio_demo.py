"""
Gradio Demo for Image Quality Assessment

This demo allows users to upload an image and get:
1. Scene type prediction
2. Distortion type prediction
3. Quality token distribution (bad, low, fair, good, awesome)
4. Final predicted quality score (1-5 scale)

The demo uses the same sequential Q&A pipeline as eval_sequential_model.py
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.new_train.model_wrapper import IQAModelWrapper
from src.new_train.processor_no_cut import create_processor_no_cut


class IQADemo:
    """Wrapper class for IQA model demo"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model_path = model_path

        print(f"Loading model from {model_path}...")

        # Convert to absolute path
        # If relative path, resolve from current working directory (where user runs the command)
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            # Resolve relative to current working directory
            model_path_abs = Path.cwd() / model_path_obj
        else:
            model_path_abs = model_path_obj

        model_path_abs = model_path_abs.resolve()

        # Check if this is a LoRA adapter directory or full model
        adapter_config_path = model_path_abs / "adapter_config.json"
        is_lora_adapter = adapter_config_path.exists()

        if is_lora_adapter:
            print("  Detected LoRA adapter, loading base model + adapter...")
            # Load base model config to get the base model path
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path", "src/owl3")

            # If base_model_path is relative, resolve it from project root
            if not Path(base_model_path).is_absolute():
                base_model_path = str(project_root / base_model_path)

            print(f"  Base model: {base_model_path}")
            print(f"  Adapter: {model_path_abs}")

            # Load tokenizer from base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path, trust_remote_code=True
            )
            self.processor = create_processor_no_cut(self.tokenizer)

            # Load base model first
            print("  Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            print("  Loading LoRA adapter...")
            model_with_adapter = PeftModel.from_pretrained(
                base_model, str(model_path_abs)
            )

            # Create wrapper without re-initializing LoRA
            self.model = IQAModelWrapper.__new__(IQAModelWrapper)
            nn.Module.__init__(self.model)
            self.model.model = model_with_adapter
            self.model.tokenizer = self.tokenizer

            # Initialize quality level configuration
            level_tokens = ["bad", "low", "fair", "good", "awesome"]
            level_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
            self.model.level_tokens = level_tokens
            self.model.level_scores = torch.tensor(level_scores, dtype=torch.float32)

            # Get token IDs for quality levels
            level_words = ["bad", "low", "fair", "good", "awesome"]
            self.model.level_token_sequences = []
            for word in level_words:
                token_ids = self.tokenizer.encode(f" {word}", add_special_tokens=False)
                self.model.level_token_sequences.append(token_ids)
        else:
            print("  Loading full model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path_abs), trust_remote_code=True
            )
            self.processor = create_processor_no_cut(self.tokenizer)
            self.model = IQAModelWrapper(str(model_path_abs))

        self.model.to(device)
        self.model.eval()

        # Quality level configuration
        self.level_names = ["bad", "low", "fair", "good", "awesome"]
        self.level_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.level_token_ids = [
            self.tokenizer.encode(f" {level_name}", add_special_tokens=False)[0]
            for level_name in self.level_names
        ]

        print("✅ Model loaded successfully!")

    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, list]:
        """Preprocess image for model input"""
        # Use processor to handle image
        # We just need to preprocess the image, the actual messages will be created during inference
        messages = [
            {
                "role": "user",
                "content": "<|image|>\n",
            }
        ]

        # Process with processor - processor needs messages parameter
        processed = self.processor(images=[image], messages=messages)

        pixel_values = processed["pixel_values"].to(self.device)
        media_offset = processed["media_offset"]  # Use media_offset from processor

        return pixel_values, media_offset

    def predict_scene(self, pixel_values: torch.Tensor, media_offset: list) -> str:
        """Step 1: Predict scene type"""
        scene_messages = [
            {"role": "user", "content": "<|image|>\n"},
            {"role": "user", "content": "What is the scene type of this image?"},
        ]

        scene_text = self.tokenizer.apply_chat_template(
            scene_messages, tokenize=False, add_generation_prompt=True
        )

        scene_encoding = self.tokenizer(
            scene_text, return_tensors="pt", add_special_tokens=False
        )
        scene_input_ids = scene_encoding["input_ids"].to(self.device)
        scene_attention_mask = scene_encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            scene_out = self.model.model.generate(
                input_ids=scene_input_ids,
                pixel_values=pixel_values,
                media_offset=media_offset,
                attention_mask=scene_attention_mask,
                tokenizer=self.tokenizer,
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
            )

        scene_response = self.tokenizer.decode(
            scene_out[0], skip_special_tokens=True
        ).strip()

        return scene_response

    def predict_distortion(
        self, pixel_values: torch.Tensor, media_offset: list, scene_response: str
    ) -> str:
        """Step 2: Predict distortion type (with scene context)"""
        distortion_messages = [
            {"role": "user", "content": "<|image|>\n"},
            {"role": "user", "content": "What is the scene type of this image?"},
            {"role": "assistant", "content": scene_response},
            {"role": "user", "content": "What is the distortion type of this image?"},
        ]

        distortion_text = self.tokenizer.apply_chat_template(
            distortion_messages, tokenize=False, add_generation_prompt=True
        )

        distortion_encoding = self.tokenizer(
            distortion_text, return_tensors="pt", add_special_tokens=False
        )
        distortion_input_ids = distortion_encoding["input_ids"].to(self.device)
        distortion_attention_mask = distortion_encoding["attention_mask"].to(
            self.device
        )

        with torch.no_grad():
            distortion_out = self.model.model.generate(
                input_ids=distortion_input_ids,
                pixel_values=pixel_values,
                media_offset=media_offset,
                attention_mask=distortion_attention_mask,
                tokenizer=self.tokenizer,
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
            )

        distortion_response = self.tokenizer.decode(
            distortion_out[0], skip_special_tokens=True
        ).strip()

        return distortion_response

    def predict_quality(
        self,
        pixel_values: torch.Tensor,
        media_offset: list,
        scene_response: str,
        distortion_response: str,
    ) -> Tuple[float, Dict[str, float]]:
        """Step 3: Predict quality score and distribution"""
        quality_messages = [
            {"role": "user", "content": "<|image|>\n"},
            {"role": "user", "content": "What is the scene type of this image?"},
            {"role": "assistant", "content": scene_response},
            {"role": "user", "content": "What is the distortion type of this image?"},
            {"role": "assistant", "content": distortion_response},
            {
                "role": "user",
                "content": "What do you think about the quality of this image?",
            },
            {"role": "assistant", "content": "The quality of this image is "},
        ]

        quality_text = self.tokenizer.apply_chat_template(
            quality_messages,
            tokenize=False,
            continue_final_message=True,
        )

        quality_encoding = self.tokenizer(
            quality_text, return_tensors="pt", add_special_tokens=False
        )
        quality_input_ids = quality_encoding["input_ids"].to(self.device)
        quality_attention_mask = quality_encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            quality_outputs = self.model.model(
                input_ids=quality_input_ids,
                pixel_values=pixel_values,
                media_offset=media_offset,
                attention_mask=quality_attention_mask,
            )

        # Extract logits at last position
        last_logits = quality_outputs.logits[0, -1, :]

        # Get logits for the 5 quality tokens
        level_logits = last_logits[self.level_token_ids]

        # Compute softmax probabilities
        level_probs = F.softmax(level_logits, dim=0)

        # Compute expected score
        expected_score = sum(
            prob.item() * score for prob, score in zip(level_probs, self.level_scores)
        )

        # Create distribution dictionary
        distribution = {
            level_name: prob.item()
            for level_name, prob in zip(self.level_names, level_probs)
        }

        return expected_score, distribution

    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Full pipeline: analyze image and return all predictions

        Returns:
            Dictionary with:
                - scene: Scene type prediction
                - distortion: Distortion type prediction
                - quality_score: Final quality score (1-5)
                - quality_distribution: Distribution over quality levels
        """
        if image is None:
            return {
                "scene": "No image provided",
                "distortion": "No image provided",
                "quality_score": 0.0,
                "quality_distribution": {},
            }

        # Preprocess image
        pixel_values, media_offset = self.preprocess_image(image)

        # Step 1: Predict scene
        scene_response = self.predict_scene(pixel_values, media_offset)

        # Step 2: Predict distortion (with scene context)
        distortion_response = self.predict_distortion(
            pixel_values, media_offset, scene_response
        )

        # Step 3: Predict quality (with scene + distortion context)
        quality_score, quality_distribution = self.predict_quality(
            pixel_values, media_offset, scene_response, distortion_response
        )

        return {
            "scene": scene_response,
            "distortion": distortion_response,
            "quality_score": quality_score,
            "quality_distribution": quality_distribution,
        }


def create_gradio_interface(model_path: str, device: str = "cuda"):
    """Create Gradio interface for IQA demo"""

    # Initialize demo
    demo_model = IQADemo(model_path, device)

    def process_image(image):
        """Process uploaded image and return formatted results"""
        if image is None:
            return (
                "❌ Please upload an image",
                "❌ Please upload an image",
                "❌ Please upload an image",
                None,
            )

        try:
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            # Analyze image
            results = demo_model.analyze_image(image)

            # Format scene output with larger font
            scene_output = f"### 🏞️ Scene Type: {results['scene']}"
            distortion_output = f"### 🔍 Distortion Type: {results['distortion']}"
            quality_score = results["quality_score"]
            quality_output = f"### ⭐ Quality Score: {quality_score:.2f} / 5.00"

            # Format distribution as bar chart data
            distribution = results["quality_distribution"]
            labels = list(distribution.keys())
            values = list(distribution.values())

            # Create distribution plot data
            import base64
            import io

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(
                labels,
                values,
                color=["#d32f2f", "#ff9800", "#fdd835", "#7cb342", "#388e3c"],
            )
            ax.set_ylabel("Probability", fontsize=12)
            ax.set_xlabel("Quality Level", fontsize=12)
            ax.set_title("Quality Token Distribution", fontsize=14, fontweight="bold")
            ax.set_ylim([0, 1.0])
            ax.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            plt.tight_layout()

            return scene_output, distortion_output, quality_output, fig

        except Exception as e:
            import traceback

            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            return error_msg, error_msg, error_msg, None

    # Create Gradio interface
    with gr.Blocks(title="Image Quality Assessment Demo") as demo:
        gr.Markdown(
            """
            # 🖼️ Image Quality Assessment Demo
            
            Upload an image to analyze its quality. The model will predict:
            1. **Scene Type**: What type of scene is in the image
            2. **Distortion Type**: What kind of distortion affects the image
            3. **Quality Score**: Overall quality score from 1 (bad) to 5 (awesome)
            4. **Quality Distribution**: Probability distribution over quality levels
            
            The model uses a sequential question-answering approach where each prediction 
            builds on previous context.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Image", type="pil", height=430.14, image_mode="RGB"
                )
                analyze_btn = gr.Button(
                    "Analyze Image", variant="primary", size="lg", interactive=False
                )

            with gr.Column(scale=1):
                scene_output = gr.Markdown(label="Scene Type")
                distortion_output = gr.Markdown(label="Distortion Type")
                quality_output = gr.Markdown(label="Quality Score")
                distribution_plot = gr.Plot(label="Quality Token Distribution")

        # Enable/disable analyze button based on image upload
        image_input.change(
            fn=lambda img: gr.update(interactive=img is not None),
            inputs=[image_input],
            outputs=[analyze_btn],
        )

        # Set up event handler
        analyze_btn.click(
            fn=process_image,
            inputs=[image_input],
            outputs=[
                scene_output,
                distortion_output,
                quality_output,
                distribution_plot,
            ],
        )

    return demo


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Launch IQA Gradio Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/10302000_full/final_model",
        help="Path to trained model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public share link"
    )
    parser.add_argument(
        "--server_name", type=str, default="0.0.0.0", help="Server name for Gradio"
    )
    parser.add_argument(
        "--server_port", type=int, default=7860, help="Server port for Gradio"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 LAUNCHING IMAGE QUALITY ASSESSMENT DEMO")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Share: {args.share}")
    print()

    # Create and launch interface
    demo = create_gradio_interface(args.model_path, args.device)

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
