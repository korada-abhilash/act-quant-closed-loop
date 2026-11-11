#!/usr/bin/env bash
# One-screen status of queued/running jobs and finished closed-loop results. Run on the cluster login node.
cd "$(dirname "$0")/.."
squeue -u "$USER" -o "%.8i %.18j %.9T %.10M %R"
echo; echo "closed-loop results (100 episodes, seeds 2000-2099):"
for d in outputs/closed_loop/*/; do
  f="$d/eval_info.json"; [ -f "$f" ] || continue
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));o=d['overall'];print(f\"  {sys.argv[2]:45s} {o['pc_success']:5.1f}%  n={o['n_episodes']}\")" "$f" "$(basename "$d")"
done
[ -f outputs/sensitivity/results.json ] && echo && echo "sensitivity: done (outputs/sensitivity/results.json)"
