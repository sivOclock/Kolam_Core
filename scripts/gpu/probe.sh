#!/usr/bin/env bash
# The GPU box at a glance + where training time goes (probe.py). Copy probe.sh and probe.py to
# ~/kolam_core_model and run: wsl -d Ubuntu -- bash -l ~/kolam_core_model/probe.sh   (~5 min, uses GPU + CPU)
source ~/kolam_core_model/env.sh
cd ~/kolam_core_model
if tmux has-session -t kolam 2>/dev/null; then echo "training is running (tmux kolam): not probing"; exit 1; fi
echo "== disk";   df -h ~ /mnt/c 2>/dev/null; du -sh data runs 2>/dev/null
echo "== memory"; free -h
echo "== cpu";    nproc; grep -m1 "model name" /proc/cpuinfo
echo "== gpu";    nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,clocks.sm,clocks.max.sm --format=csv
shard=$(ls data/enc_2m_20/*.jsonl | head -1)
PYTHONWARNINGS=ignore python probe.py "$shard" "$@"
echo "== gpu after"; nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,power.draw --format=csv
