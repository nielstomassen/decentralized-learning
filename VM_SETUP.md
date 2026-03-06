# Running hybrid_ablation on a VM (with GPU)

One-time setup and how to run the hybrid ablation experiment on a fresh VM.

## Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- **GPU (optional but recommended):** NVIDIA GPU with CUDA 12.x drivers (e.g. `nvidia-smi` works)

## 1. Clone and enter project

```bash
cd /path/to/dl_env   # or wherever you cloned the repo
```

## 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# Windows:  .venv\Scripts\activate
```

## 3. Install dependencies

### Option A: GPU VM (recommended)

Install PyTorch with CUDA first, then the rest (so the GPU build is used):

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If your driver uses **CUDA 11.8**, use `cu118` instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Option B: CPU-only

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Verify GPU (optional)

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## 5. Run hybrid ablation

From the project root (with venv activated):

```bash
./experiments/run_hybrid_ablation.sh
```

Results go to `results/emnist/hybrid_ablation/` by default. Override with:

```bash
RESULTS_ROOT=/path/to/results ./experiments/run_hybrid_ablation.sh
```

### Heavier setup: CIFAR-100 + ResNet (VM with GPU)

For better accuracy with a more expensive model (CIFAR-100, ResNet-32 or ResNet-56):

```bash
# ResNet-32 (default depth) on CIFAR-100
python3 main.py --dataset cifar100 --model resnet32 --rounds 50 --peers 50 --batch-size 32 ...

# ResNet-56 (deeper, slower, often better accuracy)
python3 main.py --dataset cifar100 --model resnet56 --rounds 50 --peers 50 --batch-size 32 ...
```

Use `--model resnet20` for a lighter, faster option. ResNet uses GroupNorm (DP-friendly). First run will download CIFAR-100 to `./data`.

### GPU-friendly options (faster on VM with GPU)

- **Larger batch size** (better GPU utilization):
  ```bash
  BATCH_SIZE=32 ./experiments/run_hybrid_ablation.sh
  # or 64 if you have enough VRAM
  ```
- **Time each round** (to see speedup):
  ```bash
  TIMING=1 ./experiments/run_hybrid_ablation.sh
  ```
- **cuDNN benchmark** (faster convs, slight non-determinism):
  ```bash
  CUDA_BENCHMARK=1 BATCH_SIZE=32 ./experiments/run_hybrid_ablation.sh
  ```
- **Fewer rounds/seeds for a quick test**:
  ```bash
  ROUNDS=5 SEEDS="42" ER_PS="0.08" ./experiments/run_hybrid_ablation.sh
  ```

## 6. One-liner setup (script)

From project root:

```bash
chmod +x scripts/setup_vm.sh
./scripts/setup_vm.sh
```

Then run the ablation as above. Use `./scripts/setup_vm.sh --gpu` to install PyTorch with CUDA.

## Plotting results

After the run finishes:

```bash
python3 src/plotting/hybrid_privacy_tradeoff.py --results-dir results/emnist/hybrid_ablation/er_p_0.08 --out-dir plots/hybrid
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `torch` installs CPU-only on GPU VM | Install torch/torchvision from the CUDA index **before** `pip install -r requirements.txt` (see step 3 Option A). |
| Out of GPU memory | Use `BATCH_SIZE=16` (default) or `BATCH_SIZE=8`. |
| `mia_attacks` install fails | Ensure git is installed and you have network access; the package is installed from GitHub. |
| EMNIST download slow | First run downloads EMNIST to `./data`; ensure disk space and network. |
