import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from sae_tools.adapters.saes import get_sae_profile


load_dotenv(BASE_DIR / ".env")

SAE_ROOT = os.getenv("SAE_ROOT")
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_HF_ENDPOINT = os.getenv("HF_ENDPOINT") or "https://hf-mirror.com"


def parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return layers


def ensure_sae_root() -> Path:
    if not SAE_ROOT:
        raise EnvironmentError(f"SAE_ROOT is not set. Configure it in {BASE_DIR / '.env'}.")
    root = Path(SAE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    return root


def is_ready(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def configure_hf_endpoint(endpoint: str | None) -> str:
    resolved = endpoint or DEFAULT_HF_ENDPOINT
    os.environ["HF_ENDPOINT"] = resolved
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SAE checkpoints into SAE_ROOT.")
    parser.add_argument("--sae_profile", default="qwen-scope-qwen3-8b-l0-50")
    parser.add_argument("--layers", default=None, help="Comma-separated layer list or ranges, e.g. 18 or 0-3,18")
    parser.add_argument("--endpoint", default=None, help="Hugging Face endpoint. Defaults to HF_ENDPOINT or hf-mirror.")
    parser.add_argument("--check", action="store_true", help="Only check expected local files")
    args = parser.parse_args()

    endpoint = configure_hf_endpoint(args.endpoint)
    from huggingface_hub import hf_hub_download

    root = ensure_sae_root()
    profile = get_sae_profile(args.sae_profile)
    layers = parse_layers(args.layers) if args.layers else [profile.default_layer]

    if profile.adapter != "qwen_scope_topk":
        raise ValueError(
            f"download_saes.py only knows the Qwen-Scope layerN.sae.pt layout. "
            f"Profile '{profile.name}' uses adapter '{profile.adapter}'."
        )

    local_dir = root / profile.local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"ENDPOINT {endpoint}")
    print(f"LOCALDIR {local_dir}")
    for layer in layers:
        filename = profile.layer_filename(layer)
        target = local_dir / filename
        if is_ready(target):
            print(f"READY  {target}")
            continue
        if args.check:
            print(f"MISS   {target}")
            continue

        print(f"FETCH  {profile.repo_id}/{filename} -> {target}")
        hf_hub_download(
            repo_id=profile.repo_id,
            repo_type="model",
            filename=filename,
            local_dir=str(local_dir),
            token=HF_TOKEN,
        )
        if not is_ready(target):
            raise FileNotFoundError(f"Download completed but expected file is missing: {target}")
        print(f"READY  {target}")


if __name__ == "__main__":
    main()
