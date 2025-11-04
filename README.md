---
title: Nanochat
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: mit
---

A lightweight chatbot powered by [nanochat](https://huggingface.co/sdobson/nanochat), a small GPT-based language model trained in 4 hours for $100. The model runs on CPU using PyTorch for fast, private inference.

Built with [Gradio](https://gradio.app) for the interface and [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index) for model distribution.
