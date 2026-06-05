"""Fine-tuning entry points (Phases 2 & 3).

Exposes ``register_cli`` once the training commands exist so the top-level CLI
can attach ``train-image`` / ``train-audio`` only when the ML stack is present.
"""
