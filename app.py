# Gradio interface for nanochat model (local ./model directory)

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import gradio as gr

from model import NanochatModel  # your NanochatModel class

# By default, look for weights in a folder named "model" next to this file.
# You can still override via MODEL_DIR env var if you want.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))).resolve()

_model: NanochatModel | None = None


def ensure_local_model_dir() -> None:
    """Validate that ./model exists and has files."""
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            "Create a folder named 'model' next to this script and place the model files inside "
            "(e.g., meta_*.json, model_*.pt, token_bytes.pt, tokenizer.pkl), "
            "or override with the MODEL_DIR environment variable."
        )
    if not any(MODEL_DIR.iterdir()):
        raise FileNotFoundError(
            f"Model directory is empty: {MODEL_DIR}\n"
            "Place your model files in this folder."
        )


def load_model() -> None:
    """Load the nanochat model from the local ./model folder."""
    global _model
    if _model is None:
        ensure_local_model_dir()
        _model = NanochatModel(model_dir=str(MODEL_DIR), device="cpu")


load_model()


def respond(
    message: str,
    history: list[dict[str, str]],
    temperature: float,
    top_k: int,
    system_prompt: str,
) -> Generator[str, Any, None]:
    """Generate a response using the nanochat model."""
    conversation: list[dict[str, str]] = []

    # If a system message is provided, put it at the start of the conversation.
    system_prompt = (system_prompt or "").strip()
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})

    # Replay prior turns
    for msg in history:
        conversation.append(msg)

    # Current user turn
    conversation.append({"role": "user", "content": message})

    response = ""
    for token in _model.generate(
        history=conversation,
        max_tokens=512,
        temperature=temperature,
        top_k=top_k,
    ):
        response += token
        yield response


chatbot = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=200, value=50, step=1, label="Top-k sampling"),
        gr.Textbox(
            label="System message (optional)",
            placeholder="e.g., You are a concise assistant that answers in markdown.",
            lines=3,
        ),
    ],
)

with gr.Blocks(title="nanochat") as demo:
    gr.Markdown("# nanochat")
    gr.Markdown("Chat with an AI trained in 4 hours for $100")
    gr.Markdown(
        "**Note:** This model is a research experiment. "
        "Obviously do not rely on the outputs!"
    )
    chatbot.render()

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7860, share=True)
