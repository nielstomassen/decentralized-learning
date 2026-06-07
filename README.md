# Topology-Aware Privacy Amplification in Decentralized Learning

**Master thesis** — Niels Tomassen

## Abstract

Decentralized learning lets data owners train models without a central server, but updates exchanged between peers can still leak information via **membership inference attacks (MIAs)**. Common defenses include **differential privacy (DP)**, which adds calibrated noise at a utility cost, and **chunking**, which shares only subsets of each update. Prior work uses **topology-aware chunking** (chunk count scales with node degree) and **fixed-*K* chunking** (the same *K* for every node), but it remains unclear how the communication graph shapes these defenses.

This thesis evaluates the privacy–utility tradeoff on **CIFAR-100** across ring, star, grid, fully connected, *d*-regular, and Erdős–Rényi topologies, measuring privacy with MIA AUC and utility with global test accuracy. Without defenses, MIA AUC stays near 0.97–0.99. Topology-aware chunking cuts leakage sharply on dense graphs (e.g. AUC ≈ 0.61 on fully connected) but protects unevenly on sparse or heterogeneous topologies. Fixed-*K* chunking is often a stronger, more uniform baseline. **ChunkDP**—topology-aware chunking with degree-scaled DP noise—recovers utility relative to DP-only while keeping AUC near random guessing (~0.53) and can outperform fixed-*K* when privacy and utility are weighted equally. Overall, topology-awareness alone does not guarantee a better tradeoff; outcomes depend on graph density, node degree, and the desired privacy–utility balance.

This repository holds the simulation code, experiment logs, plotting scripts, and published figures/tables for those results.

## Repository layout

| Path | Description |
|------|-------------|
| [`main.py`](main.py) | Training entry point (decentralized rounds + optional MIA) |
| [`args.py`](args.py), [`session_settings.py`](session_settings.py) | CLI and typed run configuration |
| [`src/`](src/) | Training loop, nodes, topologies, models, MIA runner |
| [`experiments/`](experiments/) | Shell scripts that launch training sweeps |
| [`plotting/`](plotting/) | Scripts that turn CSVs into thesis figures and summary tables |
| [`plots/`](plots/) | **Published thesis figures** (PDF/PNG and plot-side CSV summaries); output of the plotting scripts below (`topology_analysis/`, `hybrid_ablation/`, `hybrid_noise_clip_sweep/`, `appendix/`, `misc/`) |
| [`tables/`](tables/) | **Published LaTeX tables** (Used in the thesis); produced by `topology_heterogeneity.py` from `results/topology_analysis/` |
| [`results/`](results/) | **Published experiment outputs** (per-run CSV logs; see below) |
| [`requirements.txt`](requirements.txt) | Python dependencies |

## Experiment results (`results/`)

The [`results/`](results/) tree holds the CSV logs used for the main results in the thesis. Each file is one full training run (round-level metrics and per-node MIA statistics). Layout mirrors the experiment scripts:

| Directory | Experiment script | What was run |
|-----------|-------------------|--------------|
| [`results/topology_analysis/`](results/topology_analysis/) | [`experiments/run_baseline_chunk_topology_sweep.sh`](experiments/run_baseline_chunk_topology_sweep.sh) | Baseline vs topology-aware chunking across ring, star, grid, full, *d*-regular, and ER graphs |
| [`results/hybrid_ablation/`](results/hybrid_ablation/) | [`experiments/run_hybrid_ablation.sh`](experiments/run_hybrid_ablation.sh) | 2×2 factorial: baseline, DP only, topology-aware chunk, ChunkDP (ER *p* ∈ {0.08, 0.16}) |
| [`results/hybrid_noise_clip_sweep/`](results/hybrid_noise_clip_sweep/) | [`experiments/run_hybrid_noise_clip_sweep.sh`](experiments/run_hybrid_noise_clip_sweep.sh) | ChunkDP with varying noise multiplier and clip norm |
| [`results/fixed_k_chunking_sweep/`](results/fixed_k_chunking_sweep/) | [`experiments/run_fixed_k_chunking_sweep.sh`](experiments/run_fixed_k_chunking_sweep.sh) | Fixed-*K* chunking (no DP) for K ∈ {8, 16, 32, 64, 128} × ER *p* × seeds; CSVs under `<K>_standard_chunking_sweep/er_p_*` |
| [`results/appendix/peer_count_mechanism_sweep/`](results/appendix/peer_count_mechanism_sweep/) | [`experiments/run_peer_count_mechanism_sweep.sh`](experiments/run_peer_count_mechanism_sweep.sh) | ChunkDP ablation vs number of peers |
| [`results/appendix/flatBroadcastSame/`](results/appendix/flatBroadcastSame/) | [`experiments/run_chunk_variant.sh`](experiments/run_chunk_variant.sh) | Appendix chunk variant: topology-aware chunk + ChunkDP with `broadcast_same` (`topology_flat_degree`); compared to normal ablation in `normal_vs_broadcast_same_ta_hybrid.py` |

To reproduce CSVs from scratch, run the matching script from the repo root (requires a lot of time):

```bash
./experiments/run_baseline_chunk_topology_sweep.sh
./experiments/run_hybrid_ablation.sh
./experiments/run_fixed_k_chunking_sweep.sh      
./experiments/run_hybrid_noise_clip_sweep.sh
./experiments/run_peer_count_mechanism_sweep.sh
./experiments/run_chunk_variant.sh               
```

Override output location with `RESULTS_ROOT=/path/to/out ./experiments/run_hybrid_ablation.sh` (same pattern for other scripts).

## Plotting (`plotting/`)

Plotting code is grouped by thesis section. Run from the **project root**; outputs go to `plots/` unless you pass `--out-dir` / `--out-path`.

| Folder | Script | Reads | Typical outputs |
|--------|--------|-------|-----------------|
| [`plotting/topology_analysis/`](plotting/topology_analysis/) | `topology_mia_chunk_sweep.py` | `results/topology_analysis/` | `topology_deterministic_bars`, `topology_regular_degree`, `topology_er_by_p`, `topology_privacy_utility` |
| | `topology_heterogeneity.py` | same | `topology_star_hub_leaf`, `topology_grid_degree_roles`, `topology_er_degree_bins`, `topology_star_ring_auc_violin` + `tables/*.tex` |
| [`plotting/hybrid_ablation/`](plotting/hybrid_ablation/) | `hybrid_privacy_tradeoff.py` | `results/hybrid_ablation/er_p_*` (+ optional fixed-*K* dirs) | `hybrid_ablation_bars`, `hybrid_privacy_utility_scatter`, `hybrid_ablation_boxplots`, `hybrid_ablation_score`, `hybrid_ablation_lambda_grouped`, CSV summaries |
| | `chunkdp_labels.py`, `hybrid_lambda_deployment_scores.py` | — | Shared helpers (imported by other plots) |
| [`plotting/hybrid_noise_clip_sweep/`](plotting/hybrid_noise_clip_sweep/) | `analyze_hybrid_noise_clip_sweep.py` | `results/hybrid_noise_clip_sweep/er_p_*` + ablation refs; fixed-*K* refs via `--er-p` | Scatter (full + zoom), λ panels, bars, score plots, summary CSVs |
| [`plotting/appendix/`](plotting/appendix/) | `peer_count_mechanism_sweep.py` | `results/appendix/peer_count_mechanism_sweep/er_p_*` | Peer-count bars, lines, scatter, λ scores |
| | `normal_vs_broadcast_same_ta_hybrid.py` | `results/hybrid_ablation/` vs `results/appendix/flatBroadcastSame/` | Comparison bar chart + `*_comparison.csv` |
| [`plotting/misc/`](plotting/misc/) | `draw_thesis_topologies.py` | — | `plots/misc/*.pdf` (ring, ER, star, grid, …) |
| | `draw_chunking_schematic.py` | — | `plots/misc/chunking_schematic.pdf` |

**Examples** (run from project root; use `python3 -m …` so imports resolve)

```bash
# Topology chapter
python3 -m plotting.topology_analysis.topology_mia_chunk_sweep \
  --root results/topology_analysis \
  --out-dir plots/topology_analysis

python3 -m plotting.topology_analysis.topology_heterogeneity \
  --root results/topology_analysis \
  --out-dir-figures plots/topology_analysis/heterogeneity \
  --out-dir-tables tables

# ChunkDP ablation (er_p = 0.08; repeat for 0.16)
python3 -m plotting.hybrid_ablation.hybrid_privacy_tradeoff \
  --results-dir results/hybrid_ablation/er_p_0.08 \
  --additional-results-dir results/fixed_k_chunking_sweep/8_standard_chunking_sweep/er_p_0.08 \
  --additional-results-dir results/fixed_k_chunking_sweep/16_standard_chunking_sweep/er_p_0.08 \
  --additional-results-dir results/fixed_k_chunking_sweep/32_standard_chunking_sweep/er_p_0.08 \
  --additional-results-dir results/fixed_k_chunking_sweep/64_standard_chunking_sweep/er_p_0.08 \
  --additional-results-dir results/fixed_k_chunking_sweep/128_standard_chunking_sweep/er_p_0.08 \
  --out-dir plots/hybrid_ablation/er_p_0.08 \
  --lambda 0.5 --auc-col max_auc

# Noise × clip sweep (fixed-K refs loaded automatically via --er-p) (er_p = 0.08; repeat for 0.16)
python3 -m plotting.hybrid_noise_clip_sweep.analyze_hybrid_noise_clip_sweep \
  --results-glob 'results/hybrid_noise_clip_sweep/er_p_0.08/*.csv' \
  --out-dir plots/hybrid_noise_clip_sweep/er_p_0.08 \
  --ablation-ref-dir results/hybrid_ablation/er_p_0.08 \
  --er-p 0.08 --lambda 0.5 --auc-col max_auc

# Appendix
# Peer count sweep (er_p = 0.08; repeat for 0.16)
python3 -m plotting.appendix.peer_count_mechanism_sweep \
  --results-root results/appendix/peer_count_mechanism_sweep/er_p_0.08 \
  --out-dir plots/appendix/peer_count_mechanism_sweep/er_p_0.08

python3 -m plotting.appendix.normal_vs_broadcast_same_ta_hybrid \
  --factorial-root results/hybrid_ablation \
  --broadcast-root results/appendix/flatBroadcastSame \
  --out-path plots/appendix/chunk_variant/normal_vs_broadcast_same_ta_hybrid.png

# Schematics (no CSV inputs)
python3 -m plotting.misc.draw_thesis_topologies
python3 -m plotting.misc.draw_chunking_schematic
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

**Fixed-*K* chunking references** (writes to `results/fixed_k_chunking_sweep/<K>_standard_chunking_sweep/er_p_<p>/`; used as reference points in hybrid ablation and noise-sweep plots):

```bash
./experiments/run_fixed_k_chunking_sweep.sh
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
