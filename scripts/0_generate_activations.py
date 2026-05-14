import os
import yaml
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sae_tools.adapters.datasets import get_adapter
from sae_tools.adapters.models import get_model_profile, load_model_from_profile
from sae_tools.adapters.saes import get_sae_profile, load_sae_adapter
from sae_tools.model import residual_post_hook_name
from sae_tools.model.run import generate_activations
from sae_tools.workflow.artifacts import (
    activation_path,
    artifact_meta_path,
    atomic_torch_save,
    build_artifact_meta,
    done_path,
    mark_done,
    write_json_atomic,
)

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

    data_root = _require_env("DATASET_ROOT", DATASET_ROOT)
    print(f">>> Found {len(datasets)} datasets in {args.dataset_config}")

    for dataset_info in datasets:
        dataset_name = dataset_info.get('name')
        dataset_type = dataset_info.get('type', 'prompt')
        dataset_folder = dataset_info.get('folder', '')
        dataset_key = dataset_info.get("id") or f"{dataset_name}_{dataset_type}"
        dataset_max_samples = args.max_samples
        if dataset_max_samples is None:
            dataset_max_samples = dataset_info.get("max_samples", max_samples)
        dataset_path = os.path.join(data_root, dataset_folder)
        output_file = activation_path(
            root=args.output_dir,
            model=args.model_profile,
            sae=args.sae_profile,
            layer=layer,
            dataset=dataset_key,
            split=dataset_info.get("split", None),
            max_samples=dataset_max_samples,
        )
        if output_file.exists() and done_path(output_file).exists() and not args.overwrite:
            print(f">>> Reusing existing artifact: {output_file}")
            continue
        if output_file.exists() and not args.overwrite:
            raise FileExistsError(f"Artifact exists without --overwrite: {output_file}")
        
        adapter = get_adapter(dataset_name)

        dataset = adapter.load(
            dataset_path,
            dataset_max_samples,
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
        atomic_torch_save(results, output_file)
        meta = build_artifact_meta(
            script="0_generate_activations.py",
            repo_dir=BASE_DIR,
            params={
                "model": args.model_profile,
                "sae": args.sae_profile,
                "layer": layer,
                "dataset": dataset_key,
                "adapter": dataset_name,
                "data_type": dataset_type,
                "split": dataset_info.get("split", None),
                "subset": dataset_info.get("subset", None),
                "max_samples": dataset_max_samples,
                "batch_size": args.batch_size,
                "device": args.device,
                "dtype": args.dtype,
                "output": str(output_file),
            },
            inputs={
                "dataset_config": str(args.dataset_config),
                "model_path": model_profile.local_path,
                "sae_path": str(sae_profile.layer_path(_require_env("SAE_ROOT", SAE_ROOT), layer)),
                "dataset_folder": dataset_folder,
            },
        )
        write_json_atomic(artifact_meta_path(output_file), meta)
        mark_done(output_file)
        print(f">>> Results saved to {output_file}")
    
    print(f"\n>>> All datasets processed successfully!")

DEFAULT_MODEL_PROFILE = "qwen3-8b-guard"
DEFAULT_SAE_PROFILE = "qwen-scope-qwen3-8b-l0-50"
DEFAULT_DATASETS = BASE_DIR / "configs/datasets/datasets_prompt.yaml"
DEFAULT_OUTPUT_DIR = BASE_DIR / "artifacts"
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
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()
    main(args)
