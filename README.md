# ACT Policy: Closed-Loop Quantization Study

**Status: work in progress. No results yet.**

Does a quantized robot policy that matches its FP32 outputs offline (high cosine similarity) also succeed as often when it actually controls the robot? This project trains an [ACT](https://arxiv.org/abs/2304.13705) policy on LeRobot's simulated ALOHA transfer-cube task. It then exports the policy to ONNX, quantizes it to INT8, and compares action error, CPU latency and **closed-loop success rate**. Finally, it uses per-layer sensitivity analysis to find which layers to keep at higher precision.

## Setup

Tested with LeRobot 0.6.1, Python 3.12 and CUDA 12.8 PyTorch wheels (NVIDIA driver >= 570). Exact package versions are in `requirements.lock.txt`.

```bash
CONDA_ROOT=/path/to/miniconda3 bash scripts/setup_env.sh
```

Training and simulation evaluation run as SLURM jobs on a GPU machine. To set this up, copy `cluster.local.env.example` to `cluster.local.env`, fill it in, then:

```bash
scripts/submit.sh jobs/eval_pretrained.sbatch   # check the setup with LeRobot's published checkpoint
scripts/submit.sh jobs/train_act.sbatch         # train from scratch
```

## Results

To be filled in from actual runs.
