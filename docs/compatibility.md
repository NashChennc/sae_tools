# Compatibility Notes

## Version target

This codebase is adapted for:

* `transformer-lens>=3.1,<4`
* `sae-lens>=6.43,<7`
* `transformers>=4.57,<5`
* Python `>=3.10`

The recommended isolated environment is `sae-tl3`.

## Offline loading

Runtime model loading should not rely on online Hugging Face resolution. The default path is:

1. Load tokenizer and HF model from `${MODEL_ROOT}` with `local_files_only=True`.
2. Pass the local objects into `TransformerBridge.boot_transformers`.
3. Cache TL3 activations using the project hook helper.

Do not reintroduce the old `AutoConfig` monkey patch for the TL3 path. The old `load_hooked_transformer_offline` function remains only as a compatibility reference.

## Hook convention

Residual post activations use the TL3 canonical hook:

```python
blocks.{layer}.hook_out
```

Use `sae_tools.model.residual_post_hook_name(layer)` everywhere.

## SAE layout

SAE file locations are profile-owned. Business scripts should use `get_sae_profile(...).layer_path(SAE_ROOT, layer)`.

The default Qwen-Scope SAE profile is:

```text
${SAE_ROOT}/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50/layer18.sae.pt
```

That checkpoint is a PyTorch dict with `W_enc`, `W_dec`, `b_enc`, and `b_dec`. The adapter validates expected Qwen3-8B shapes and exposes `W_dec` as `[d_sae, d_model]` for dashboard compatibility.

The `adamkarvonen` profile intentionally preserves the original BatchTopK-to-JumpReLU loader. Its adapter is only a routing layer; the conversion logic stays in `load_custom_batch_topk_as_jumprelu`.

## SAE downloads

Use `download_saes.py` for Qwen-Scope files. The script prefers `HF_ENDPOINT` from `.env`; if it is unset, it defaults to `https://hf-mirror.com`. The endpoint is configured before `huggingface_hub` is imported.

Default layer 18 command:

```bash
python download_saes.py --sae_profile qwen-scope-qwen3-8b-l0-50 --layers 18
```

Check-only command:

```bash
python download_saes.py --sae_profile qwen-scope-qwen3-8b-l0-50 --layers 18 --check
```
