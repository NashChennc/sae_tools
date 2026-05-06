import os
import yaml
import argparse
from pathlib import Path
from dotenv import load_dotenv

import torch

from sae_tools.adapters.datasets import get_adapter
from sae_tools.utils import FilenameConstructor
from sae_tools.adapters.models import get_model_profile, load_model_from_profile
from sae_tools.adapters.saes import get_sae_profile, load_sae_adapter
from sae_tools.model import residual_post_hook_name
from sae_tools.model.run import generate_activations

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_ROOT = os.getenv("MODEL_ROOT")
SAE_ROOT = os.getenv("SAE_ROOT")
DATASET_ROOT = os.getenv("DATASET_ROOT")

def _require_env(name: str, value: str | None) -> str:
    if not value:
        raise EnvironmentError(f"{name} is not set. Configure it in {BASE_DIR / '.env'}.")
    return value


def load_model_and_sae(args):
    model_root = _require_env("MODEL_ROOT", MODEL_ROOT)
    sae_root = _require_env("SAE_ROOT", SAE_ROOT)

    sae_profile = get_sae_profile(args.sae_profile)
    layer = sae_profile.default_layer if args.layer is None else args.layer

    sae_path = sae_profile.layer_path(sae_root, layer)
    hook_name = residual_post_hook_name(layer)

    tokenizer, model, model_profile = load_model_from_profile(
        model_root=model_root,
        profile_name=args.model_profile,
        device=args.device,
        dtype=args.dtype,
    )

    sae = load_sae_adapter(
        profile=sae_profile,
        sae_path=sae_path,
        model_name=model_profile.hf_name,
        layer=layer,
        hook_name=hook_name,
        device=args.device,
        dtype=args.dtype,
    )
    return tokenizer, model, sae, model_profile, sae_profile, layer

def main(args):
    tokenizer, model, sae, model_profile, sae_profile, layer = load_model_and_sae(args)

    with open(args.dataset_config, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    defaults = dataset_config.get('defaults', {})
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = dataset_config.get('max_samples', defaults.get('max_samples', -1))

    datasets = dataset_config.get('datasets', [])
    if args.dataset_names:
        requested = {name.strip() for name in args.dataset_names.split(",") if name.strip()}
        datasets = [item for item in datasets if item.get("name") in requested]
        missing = requested.difference({item.get("name") for item in datasets})
        if missing:
            raise ValueError(f"Requested dataset names not found in config: {sorted(missing)}")

    print(f">>> Found {len(datasets)} datasets in {args.dataset_config}")

    filecon = FilenameConstructor(
        f"{model_profile.name}_{sae_profile.name}_L{layer}",
        args.output_dir
    )
    for dataset_info in datasets:
        dataset_name = dataset_info.get('name')
        dataset_type = dataset_info.get('type', 'prompt')
        dataset_folder = dataset_info.get('folder', '')
        dataset_path = os.path.join(DATASET_ROOT, dataset_folder)
        
        adapter = get_adapter(dataset_name)

        dataset = adapter.load(
            dataset_path,
            max_samples,
            split = dataset_info.get('split', None),
            subset = dataset_info.get('subset', None)
        )

        print(">>> Start Inference...")
        print(f">>> Data type: {dataset_type}")
        
        results = generate_activations(
            tokenizer=tokenizer,
            model=model,
            sae=sae,
            layer=layer,
            dataset=dataset,
            data_type=dataset_type,
            batch_size=args.batch_size
        )
        output_file = filecon.file_name("SAE", dataset_name, "predictions", "pt")
        torch.save(results, output_file)
        print(f">>> Results saved to {output_file}")
    
    print(f"\n>>> All datasets processed successfully!")

DEFAULT_MODEL_PROFILE = "qwen3-8b-guard"
DEFAULT_SAE_PROFILE = "qwen-scope-qwen3-8b-l0-50"
DEFAULT_DATASETS = BASE_DIR / "configs/datasets/datasets_prompt.yaml"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"
DEFAULT_DEVICE = "cuda"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate New Method Model")
    parser.add_argument("--model_profile", type=str, default=DEFAULT_MODEL_PROFILE)
    parser.add_argument("--sae_profile", type=str, default=DEFAULT_SAE_PROFILE)
    parser.add_argument("--layer", type=int, default=None)
    
    parser.add_argument("--dataset_config", type=str, default=DEFAULT_DATASETS)
    parser.add_argument("--dataset_names", type=str, default=None, help="Comma-separated dataset names to run")
    parser.add_argument("--max_samples", type=int, default=None)
    
    parser.add_argument("--batch_size", type=int, default=1)
    
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()
    main(args)
