def residual_post_hook_name(layer: int) -> str:
    """TransformerLens 3 canonical residual stream hook for a layer output."""
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    return f"blocks.{layer}.hook_out"
