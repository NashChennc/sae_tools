# model/

### Overview

The **Model** module is the computational core of this toolkit. It handles the loading of Large Language Models (LLMs) and Sparse Autoencoders (SAEs), and executes inference to extract sparse feature activations.

It bridges `transformer_lens` (for model hooking) and `sae_lens` (for SAE encoding), providing adapters for specific SAE weight formats (e.g., BatchTopK).

### Core Components

#### 1. `adapters/models` and `adapters/saes`: stable path and format profiles
Profiles are the source of truth for model paths, SAE paths, adapter names, dimensions, and file naming.

* `qwen3-8b-guard`: `${MODEL_ROOT}/Qwen/Qwen3Guard-Gen-8B`
* `qwen3-8b-base`: `${MODEL_ROOT}/Qwen/Qwen3-8B`
* `qwen-scope-qwen3-8b-l0-50`: `${SAE_ROOT}/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50/layer{layer}.sae.pt`
* `adamkarvonen`: the existing Adam Karvonen BatchTopK checkpoint, loaded through the original conversion logic

#### 2. `load_model.py`: LLM loading
The TL3 path is `load_transformer_bridge_offline`. It loads the Hugging Face model and tokenizer from a local directory with `local_files_only=True`, then boots a TransformerLens `TransformerBridge`.

The old `load_hooked_transformer_offline` function remains in place for reference and for the Adam Karvonen compatibility path, but the default generation script uses TransformerBridge.

#### 3. `adapters/saes`: SAE format adapters
SAE loading is routed through `sae_tools.adapters.saes.load_sae_adapter(profile=...)`.

* `qwen_scope_topk`: loads Qwen-Scope `.pt` files with `W_enc/W_dec/b_enc/b_dec`, validates shapes, and implements TopK `encode`.
* `adamkarvonen`: thin adapter that preserves and calls the original `load_custom_batch_topk_as_jumprelu` function.

Business scripts should not inspect checkpoint keys or construct SAE filenames directly.

#### 4. `run.py`: Activation Generation
Manages the forward pass and feature extraction pipeline.
* **Formatting**: Processes input text using prompt templates (User/Assistant) and identifies valid token start/end indices.
* **Caching & Encoding**: Uses the TL3 residual output hook from `hooks.py`, encodes activations via the selected SAE adapter, and filters high-norm outliers.
* **Sparse Storage**: Converts activation results into Sparse COO tensors for downstream analysis.

### Hook convention

The residual stream hook for TL3 is:

```python
blocks.{layer}.hook_out
```

Always call `residual_post_hook_name(layer)` instead of hard-coding hook names.

### Usage

This module is typically used in conjunction with `sae_tools.adapters.datasets`. Below is a minimal example:

```python
from sae_tools.adapters.models import get_model_profile, load_model_from_profile
from sae_tools.adapters.saes import get_sae_profile, load_sae_adapter
from sae_tools.model import residual_post_hook_name
from sae_tools.model.run import generate_activations

model_profile = get_model_profile("qwen3-8b-guard")
sae_profile = get_sae_profile("qwen-scope-qwen3-8b-l0-50")
layer = 18

tokenizer, model, model_profile = load_model_from_profile(
    model_root="/path/from/MODEL_ROOT",
    profile_name=model_profile.name,
    device="cuda",
)

sae = load_sae_adapter(
    profile=sae_profile,
    sae_path="/path/from/SAE_ROOT/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50/layer18.sae.pt",
    model_name=model_profile.hf_name,
    layer=layer,
    hook_name=residual_post_hook_name(layer),
    device="cuda",
)

# 4. Generate Activations
# Assuming 'dataset' is a HuggingFace Dataset with a "prompt" column
results = generate_activations(
    tokenizer=tokenizer,
    model=model,
    sae=sae,
    layer=layer,
    dataset=dataset,
    data_type="prompt", 
    batch_size=1
)

# 5. Save Results
torch.save(results, "activations.pt")

```

### Output Structure

`generate_activations` returns a dictionary containing the necessary data for downstream analysis:

| Key | Type | Description |
| --- | --- | --- |
| `sparse_acts` | `torch.sparse_coo_tensor` | The core sparse activation matrix with shape `[Total_Tokens, d_sae]`, optimized via `coalesce()`. |
| `valid_token_idx` | `torch.Tensor` | Shape `[Batch, 2]`. Stores the start and end indices of valid text for each sample (excluding padding/special tokens). |
| `seq_lens` | `torch.Tensor` | Shape `[Batch]`. Records the sequence length of each sample. |
| `shape` | `torch.Size` | The original shape of the flattened activation tensor `[Batch * Length, d_sae]`. |

### Integration

* **Script**: See `0_generate_activations.py` in the project root for the full batch processing workflow.
* **Analysis**: Use `sae_tools.analysis` for category, statistical, geometric, and dashboard workflows.
* **Dependencies**: Requires `sae_lens` for SAE operations and `transformer_lens` for model instrumentation.
