#!/usr/bin/env python3
"""
Quick test script to verify the demo components work correctly
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    try:
        import gradio as gr
        import torch
        from PIL import Image
        from transformers import AutoTokenizer

        from src.new_train.model_wrapper import IQAModelWrapper
        from src.new_train.processor_no_cut import create_processor_no_cut

        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gradio():
    """Test that Gradio can create a basic interface"""
    print("\nTesting Gradio interface creation...")
    try:
        import gradio as gr

        def dummy_fn(x):
            return f"Received: {x}"

        demo = gr.Interface(fn=dummy_fn, inputs=gr.Textbox(), outputs=gr.Textbox())
        print("✅ Gradio interface creation successful!")
        return True
    except Exception as e:
        print(f"❌ Gradio test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_path():
    """Check if default model path exists"""
    print("\nChecking model paths...")
    model_paths = [
        "outputs/10302000_full/final_model",
        "outputs/10310800_full_2/final_model",
        "outputs/10301000_scene/final_model",
    ]

    found = False
    for path in model_paths:
        full_path = project_root / path
        if full_path.exists():
            print(f"✅ Found model: {path}")
            found = True
        else:
            print(f"⚠️  Model not found: {path}")

    if not found:
        print(
            "\n⚠️  No models found. You'll need to specify a valid model path when running the demo."
        )

    return True


def main():
    print("=" * 80)
    print("IQA DEMO TEST SUITE")
    print("=" * 80)
    print()

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Gradio", test_gradio()))
    results.append(("Model Paths", test_model_path()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if not result:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests passed! Demo should work correctly.")
        print("\nTo launch the demo:")
        print("  ./run_demo.sh")
        print("\nOr:")
        print("  uv run python gradio_demo.py --model_path <your_model_path>")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
