"""M7: self-play PPO training for the model-free baseline (environment spec §7.1 family 1).
Requires the `learn` extra (torch, tensorboard). Depends on `core/`, `env/`, `agents/` (including
`agents/nn/`) and `eval/` for the training-loop's own paired evaluation checks -- never on `ui/`.
"""
