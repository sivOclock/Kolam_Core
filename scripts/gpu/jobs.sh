# one training run per line: <run name> <train args>; run one after another by launch.sh
# model B (grid) chosen; the overnight run continues the bake-off's B from its last.pt
bake_B --model grid --steps 40000 --eval-every 1000 --val-n 200 data/enc_400k_20 --size 384 --batch 24 --warmup 200 --log-every 10 --workers 14 --aug real --not-kolam data/not_kolam --real-kolams data/real_kolams_line --samples 6 --save-every 100
