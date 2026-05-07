# GPU Memory Allocation Table

Measured on 2026-05-07 with NVIDIA A40 46GB cards. Empty cards show about 7 MiB driver baseline, so the runner treats `used <= 512 MiB` as unoccupied by default.

| Workload | Target Scope | Batch Size | Samples | Peak Delta MiB | Recommended Free MiB | Runtime Sec | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| activation | qwen3-8b + qwen-scope-qwen3-8b-l0-50, ToxicChat_prompt, layer 18 | 2 | 1000 | 19131 | 24000 | 90.6 | saved `acts.pt`, reusable |
| activation | qwen3-8b + qwen-scope-qwen3-8b-l0-50, HarmBench_prompt, layer 18 | 2 | 200 actual / 1000 requested | 17723 | 23000 | 60.4 | saved `acts.pt`, reusable |
| activation | qwen3-8b + qwen-scope-qwen3-8b-l0-50, ToxicChat_response, layer 18 | 2 | 1000 | 20981 | 26000 | 151.0 | saved `acts.pt`, reusable; labels use `prompt_label` |
| activation | qwen3-8b + qwen-scope-qwen3-8b-l0-50, Aegis2.0_response, layer 18 | 2 | 1000 requested / 626 labeled | 21803 | 27000 | 120.7 | saved `acts.pt`, reusable |
| activation | qwen3-8b + qwen-scope-qwen3-8b-l0-50, BeaverTails_response, layer 18 | 2 | 1000 | 18741 | 24000 | 90.5 | saved `acts.pt`, reusable |
| stat batch | per dataset + aggregation | n/a | 200-1000 | 0 | 1024 | 15.3-30.7 | CPU-bound in this run |
| stat batch | response_grid per dataset + aggregation | n/a | 626-1000 labeled | 0 | 1024 | 60.4-60.7 | CPU-bound in this run |
| geometric norm | qwen-scope-qwen3-8b-l0-50, layer 18 | n/a | 65536 features | 0 | 1024 | 15.2 | CPU-bound in this run |
| geometric seed_topk_cosine | qwen-scope-qwen3-8b-l0-50, layer 18, 196 seeds, top_k 50 | n/a | 196 seeds | 0 | 1024 | 15.4 | CPU-bound in this run |
| geometric seed_topk_cosine | response_grid, qwen-scope-qwen3-8b-l0-50, layer 18, top_k 50 | n/a | 200 seeds | 0 | 1024 | 30.3 | CPU-bound in this run |

Detailed raw records:

- `runs/gpu_memory/20260507_091517/memory_usage.md`
- `runs/gpu_memory/20260507_091937/memory_usage.md`
- `runs/gpu_memory/20260507_093358/memory_usage.md`
