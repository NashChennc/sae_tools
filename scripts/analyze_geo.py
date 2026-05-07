from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from workflow_common import REPO_ROOT, add_registry_args, load_repo_env, require_env, resolve_registry

from sae_tools.adapters.saes import get_sae_profile
from sae_tools.model.load_model import load_decoder_matrix
from sae_tools.workflow.artifacts import artifact_meta_path, build_artifact_meta, mark_done, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one geometric SAE decoder analysis artifact.")
    add_registry_args(parser)
    parser.add_argument("--sae", required=True, help="SAE registry key.")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--method", required=True, choices=["norm", "topk_cosine", "seed_topk_cosine"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _norm_payload(W_dec: torch.Tensor, top_k: int) -> dict:
    norms = torch.norm(W_dec.float(), p=2, dim=1).cpu()
    k = min(top_k, norms.numel())
    values, indices = torch.topk(norms, k=k)
    return {
        "method": "norm",
        "shape": list(W_dec.shape),
        "summary": {
            "mean": float(norms.mean().item()),
            "std": float(norms.std().item()),
            "min": float(norms.min().item()),
            "max": float(norms.max().item()),
        },
        "top_features": [
            {"feature": int(idx.item()), "norm": float(value.item())}
            for idx, value in zip(indices, values)
        ],
    }


def _topk_cosine_payload(W_dec: torch.Tensor, top_k: int, chunk_size: int, device: str) -> dict:
    matrix = F.normalize(W_dec.float(), p=2, dim=1).to(device)
    n_features = matrix.shape[0]
    k = min(top_k + 1, n_features)
    records = []
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        sims = torch.mm(matrix[start:end], matrix.t())
        row_ids = torch.arange(start, end, device=sims.device)
        sims[torch.arange(end - start, device=sims.device), row_ids] = -float("inf")
        values, indices = torch.topk(sims, k=k, dim=1)
        values = values.cpu()
        indices = indices.cpu()
        for offset in range(end - start):
            neighbors = []
            for idx, value in zip(indices[offset], values[offset]):
                if len(neighbors) >= top_k:
                    break
                if value == -float("inf"):
                    continue
                neighbors.append({"feature": int(idx.item()), "cosine": float(value.item())})
            records.append({"feature": start + offset, "neighbors": neighbors})
    return {
        "method": "topk_cosine",
        "shape": list(W_dec.shape),
        "top_k": top_k,
        "neighbors": records,
    }


def _read_seed_features(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    features = payload.get("features", [])
    if not features:
        raise ValueError(f"No seed features found in {path}")
    return [int(item) for item in features]


def _seed_topk_cosine_payload(
    W_dec: torch.Tensor,
    *,
    seed_features: list[int],
    top_k: int,
    chunk_size: int,
    device: str,
) -> dict:
    matrix = F.normalize(W_dec.float(), p=2, dim=1).to(device)
    n_features = matrix.shape[0]
    seeds = [feature for feature in seed_features if 0 <= feature < n_features]
    if not seeds:
        raise ValueError("No valid seed features remain after range filtering.")

    records = []
    for start in range(0, len(seeds), chunk_size):
        chunk = seeds[start : start + chunk_size]
        seed_tensor = torch.tensor(chunk, dtype=torch.long, device=matrix.device)
        sims = torch.mm(matrix[seed_tensor], matrix.t())
        sims[torch.arange(len(chunk), device=sims.device), seed_tensor] = -float("inf")
        values, indices = torch.topk(sims, k=min(top_k + 1, n_features), dim=1)
        values = values.cpu()
        indices = indices.cpu()
        for row, seed in enumerate(chunk):
            neighbors = []
            for idx, value in zip(indices[row], values[row]):
                if len(neighbors) >= top_k:
                    break
                if value == -float("inf"):
                    continue
                neighbors.append({"feature": int(idx.item()), "cosine": float(value.item())})
            records.append({"feature": int(seed), "neighbors": neighbors})

    return {
        "method": "seed_topk_cosine",
        "shape": list(W_dec.shape),
        "top_k": top_k,
        "n_seeds": len(seeds),
        "seed_features": seeds,
        "neighbors": records,
    }


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        print(f"READY {args.out}")
        return

    load_repo_env()
    registry = resolve_registry(args)
    sae_spec = registry.sae(args.sae)
    sae_profile = get_sae_profile(args.sae)
    layer = sae_profile.default_layer if args.layer is None else args.layer
    sae_path = sae_profile.layer_path(require_env("SAE_ROOT"), layer)
    W_dec = load_decoder_matrix(sae_path)
    if W_dec is None:
        raise ValueError(f"Could not load decoder matrix from {sae_path}")

    seed_payload = None
    if args.method == "norm":
        payload = _norm_payload(W_dec, args.top_k)
    elif args.method == "topk_cosine":
        payload = _topk_cosine_payload(W_dec, args.top_k, args.chunk_size, args.device)
    else:
        if args.seeds is None:
            raise ValueError("--method seed_topk_cosine requires --seeds.")
        seed_features = _read_seed_features(args.seeds)
        seed_payload = {"seeds": str(args.seeds), "n_requested_seeds": len(seed_features)}
        payload = _seed_topk_cosine_payload(
            W_dec,
            seed_features=seed_features,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            device=args.device,
        )
    payload.update({"sae": args.sae, "layer": layer})
    write_json_atomic(args.out, payload)
    meta = build_artifact_meta(
        script="analyze_geo.py",
        repo_dir=REPO_ROOT,
        params={
            "sae": args.sae,
            "layer": layer,
            "method": args.method,
            "seeds": str(args.seeds) if args.seeds else None,
            "top_k": args.top_k,
            "chunk_size": args.chunk_size,
            "device": args.device,
            "output": str(args.out),
        },
        inputs={
            "sae_path": str(sae_path),
            "repo_id": sae_spec.repo_id,
            "seed_payload": seed_payload,
        },
    )
    write_json_atomic(artifact_meta_path(args.out), meta)
    mark_done(args.out)
    print(f"DONE {args.out}")


if __name__ == "__main__":
    main()
