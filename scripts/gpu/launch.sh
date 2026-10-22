#!/usr/bin/env bash
# Copy launch.sh and jobs.sh to ~/kolam_core_model on the GPU box (WSL); see todo.md "Running on the GPU laptop".
# The GPU laptop's launcher, all inside ~/kolam_core_model. The Windows scheduled task "kolam_core_model"
# runs it from the desktop session, so WSL stays up. It runs jobs.sh, one line after another: a training
# run "<run name> <train args...>" (resumable), or "sh <command>" (e.g. generating data), with:
#   tmux "kolam": train (the jobs) + watch (http://localhost:8800, the run in progress)
#   the live view opened in the laptop's browser
#   Windows asked not to sleep until the jobs end (awake.ps1; no setting is changed)
source ~/kolam_core_model/env.sh
mkdir -p runs
rm -f runs/DONE
cat > runs/awake.ps1 <<'PS'
param([string]$done)
Add-Type -Name P -Namespace W -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState(0x80000001) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, this process only
while (-not (Test-Path $done)) { Start-Sleep 30 }
PS
cat > runs/jobs_runner.sh <<'RUN'
source ~/kolam_core_model/env.sh
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
while read -r name args; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  if [ "$name" = "sh" ]; then bash -c "$args" 2>&1 | tee -a runs/jobs.out; continue; fi
  mkdir -p runs/$name
  tmux kill-window -t kolam:watch 2>/dev/null
  tmux new-window -d -t kolam -n watch "source ~/kolam_core_model/env.sh; python -m kolam_feed.watch runs/$name --port 8800"
  python -m kolam_feed train $args --out runs/$name --resume 2>&1 | tee -a runs/$name/train.out
done < jobs.sh
touch runs/DONE
RUN
tmux kill-session -t kolam 2>/dev/null || true
tmux new-session -d -s kolam -n train "bash runs/jobs_runner.sh"
first=$(awk '!/^#/ && NF && $1 != "sh" {print $1; exit}' jobs.sh)    # the page is up from the start,
mkdir -p runs/$first                                                    # even while a "sh" step makes data
tmux new-window -d -t kolam -n watch "source ~/kolam_core_model/env.sh; python -m kolam_feed.watch runs/$first --port 8800"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w runs/awake.ps1)" "$(wslpath -w runs)\\DONE" &
for i in $(seq 30); do curl -s -o /dev/null localhost:8800/watch.html && break; sleep 2; done
cmd.exe /c start "" "http://127.0.0.1:8800/watch.html"
while tmux has-session -t kolam 2>/dev/null; do sleep 30; done   # keeps WSL up until the session is closed
