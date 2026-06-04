# Topology-Aware Privacy Amplification in Decentralized Learning

**Master thesis** — Niels Tomassen

Simulation code for peer-to-peer decentralized learning on image classification, with **membership inference attack (MIA)** evaluation. The thesis compares topology-aware chunking, fixed-*K* chunking, DP-SGD, and **ChunkDP** (topology-aware chunking with degree-scaled DP noise).

## Repository layout

| Path | Description |
|------|-------------|
| [`main.py`](main.py) | Training entry point (decentralized rounds + optional MIA) |
| [`args.py`](args.py), [`session_settings.py`](session_settings.py) | CLI and typed run configuration |
| [`src/`](src/) | Training loop, nodes, topologies, models, MIA runner |
| [`experiments/`](experiments/) | Shell scripts that launch training sweeps |
| [`plotting/`](plotting/) | Scripts that turn CSVs into thesis figures and summary tables |
| [`results/`](results/) | **Published experiment outputs** (per-run CSV logs; see below) |
| [`requirements.txt`](requirements.txt) | Python dependencies |

Generated plot outputs are written under `plots/` by default when you run the plotting scripts.

## Experiment results (`results/`)

The [`results/`](results/) tree holds the CSV logs used for the main results in the thesis. Each file is one full training run (round-level metrics and per-node MIA statistics). Layout mirrors the experiment scripts:

| Directory | Experiment script | What was run |
|-----------|-------------------|--------------|
| [`results/topology_analysis/`](results/topology_analysis/) | [`experiments/run_baseline_chunk_topology_sweep.sh`](experiments/run_baseline_chunk_topology_sweep.sh) | Baseline vs topology-aware chunking across ring, star, grid, full, *d*-regular, and ER graphs |
| [`results/hybrid_ablation/`](results/hybrid_ablation/) | [`experiments/run_hybrid_ablation.sh`](experiments/run_hybrid_ablation.sh) | 2×2 factorial: baseline, DP only, topology-aware chunk, ChunkDP (ER *p* ∈ {0.08, 0.16}) |
| [`results/hybrid_noise_clip_sweep/`](results/hybrid_noise_clip_sweep/) | [`experiments/run_hybrid_noise_clip_sweep.sh`](experiments/run_hybrid_noise_clip_sweep.sh) | ChunkDP with varying noise multiplier and clip norm |
| [`results/fixed_k_chunking_sweep/`](results/fixed_k_chunking_sweep/) | [`experiments/run_fixed_k_chunking_sweep.sh`](experiments/run_fixed_k_chunking_sweep.sh) | Fixed-*K* chunking reference runs (K ∈ {8, 16, 32, 64, 128} per `*_standard_chunking_sweep/er_p_*`) |
| [`results/appendix/peer_count_mechanism_sweep/`](results/appendix/peer_count_mechanism_sweep/) | [`experiments/run_peer_count_mechanism_sweep.sh`](experiments/run_peer_count_mechanism_sweep.sh) | ChunkDP ablation vs number of peers |
| [`results/appendix/flatBroadcastSame/`](results/appendix/flatBroadcastSame/) | (manual / variant runs) | Topology-aware chunking with same chunks broadcast to all neighbors |

To reproduce CSVs from scratch, run the matching script from the repo root (requires lot of time):

```bash
./experiments/run_hybrid_ablation.sh
./experiments/run_baseline_chunk_topology_sweep.sh   
```

Override output location with `RESULTS_ROOT=/path/to/out ./experiments/run_hybrid_ablation.sh`.

## Plotting (`plotting/`)

Plotting code is grouped by thesis section. Run from the **project root**; outputs go to `plots/` unless you pass `--out-dir` / `--out-path`.

| Folder | Script | Reads | Typical outputs |
|--------|--------|-------|-----------------|
| [`plotting/topology_analysis/`](plotting/topology_analysis/) | `topology_mia_chunk_sweep.py` | `results/topology_analysis/` | `topology_deterministic_bars`, `topology_regular_degree`, `topology_er_by_p`, `topology_privacy_utility` |
| | `topology_heterogeneity.py` | same | `topology_star_hub_leaf`, `topology_grid_degree_roles`, `topology_er_degree_bins`, `topology_star_ring_auc_violin` + `tables/*.tex` |
| [`plotting/hybrid_ablation/`](plotting/hybrid_ablation/) | `hybrid_privacy_tradeoff.py` | `results/hybrid_ablation/er_p_*` (+ optional fixed-*K* dirs) | `hybrid_ablation_bars`, `hybrid_privacy_utility_scatter`, `hybrid_ablation_lambda_grouped`, CSV summaries |
| | `chunkdp_labels.py`, `hybrid_lambda_deployment_scores.py` | — | Shared helpers (imported by other plots) |
| [`plotting/hybrid_noise_clip_sweep/`](plotting/hybrid_noise_clip_sweep/) | `analyze_hybrid_noise_clip_sweep.py` | `results/hybrid_noise_clip_sweep/er_p_*` + ablation & fixed-*K* refs | Scatter, λ panels, bars, score plots |
| [`plotting/appendix_figures/`](plotting/appendix_figures/) | `peer_count_mechanism_sweep.py` | `results/appendix/peer_count_mechanism_sweep/` | Peer-count bars, lines, scatter, λ scores |
| | `normal_vs_broadcast_same_ta_hybrid.py` | `results/hybrid_ablation/` vs `results/appendix/flatBroadcastSame/` | Comparison bar chart + scatter |
| [`plotting/misc/`](plotting/misc/) | `draw_thesis_topologies.py` | — | `plots/graphs/*.pdf` (ring, ER, star, grid, …) |
| | `draw_chunking_schematic.py` | — | Chunking schematic PDF |

**Examples**

```bash
# Topology chapter
python plotting/topology_analysis/topology_mia_chunk_sweep.py \
  --root results/topology_analysis --out-dir plots/topology_analysis

python plotting/topology_analysis/topology_heterogeneity.py \
  --root results/topology_analysis \
  --out-dir-figures plots/topology_analysis \
  --out-dir-tables tables

# ChunkDP ablation (er_p = 0.08)
python plotting/hybrid_ablation/hybrid_privacy_tradeoff.py \
  --results-dir results/hybrid_ablation/er_p_0.08 \
  --additional-results-dir results/fixed_k_chunking_sweep/8_standard_chunking_sweep/er_p_0.08 \
  --additional-results-dir results/fixed_k_chunking_sweep/16_standard_chunking_sweep/er_p_0.08 \
  --additional-results-dir results/fixed_k_chunking_sweep/32_standard_chunking_sweep/er_p_0.08 \
  --additional-results-dir results/fixed_k_chunking_sweep/64_standard_chunking_sweep/er_p_0.08 \
  --additional-results-dir results/fixed_k_chunking_sweep/128_standard_chunking_sweep/er_p_0.08 \
  --out-dir plots/hybrid_ablation/er_p_0.08 --lambda 0.5 --auc-col max_auc

# Noise × clip sweep
python plotting/hybrid_noise_clip_sweep/analyze_hybrid_noise_clip_sweep.py \
  --results-glob 'results/hybrid_noise_clip_sweep/er_p_0.08/*.csv' \
  --out-dir plots/hybrid_noise_clip_sweep/er_p_0.08 \
  --ablation-ref-dir results/hybrid_ablation/er_p_0.08 \
  --er-p 0.08 --lambda 0.5 --auc-col max_auc
```

## Requirements

- Python **3.9+** (3.10 or 3.11 recommended)
- Optional: NVIDIA GPU with CUDA 12.x

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# GPU: install CUDA PyTorch first, then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CPU-only: `pip install -r requirements.txt`

## Quick start (training)

```bash
python main.py --dataset mnist --model cnn --topology ring --peers 10 --rounds 30
```

**Hybrid ablation** (writes to `results/hybrid_ablation/` by default):

```bash
./experiments/run_hybrid_ablation.sh
```

## Main CLI options

| Flag | Purpose |
|------|---------|
| `--topology` | `ring`, `er`, `regular`, `grid`, `star`, `full`, `ws`, `sbm`, … |
| `--dataset` / `--model` | e.g. `cifar100` + `cnn` or `resnet32` |
| `--dp` | Enable DP-SGD (`--dp-noise`, `--dp-clip`) |
| `--chunk` | Enable chunking (`--chunking-mode`, `--chunks-per-neighbor`) |
| `--chunking-mode` | `topology_rowblocks`, `topology_flat_degree`, `standard_chunking` |
| `--mia-attack` | MIA evaluation (see `args.py`) |

Full list: `python main.py --help`

## Citation

```bibtex
@mastersthesis{tomassen2025chunkDP,
  author  = {Tomassen, Niels},
  title   = {Topology-Aware Privacy Amplification in Decentralized Learning: A Hybrid Chunking and Differential Privacy Approach Against Membership Inference Attacks},
  school  = {TU Delft},
  year    = {2026},
  type    = {Master's thesis}
}
```

## Dependencies

- [Opacus](https://opacus.ai/) — DP-SGD
- [mia_attacks](https://github.com/nielstomassen/mia_attacks) — MIA implementations (via `requirements.txt`)
