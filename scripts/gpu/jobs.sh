# one line after another (launch.sh): "<run name> <train args>" trains (resumable), "sh <command>" runs it
# night 1: model B-M (~11.6M: ResNet-34 + 4 context layers) on the 2M kolams up to 20 x 20 (made by
#   kolam_gen generate -n 2000000 -o data/enc_2m_20 --min-size 3 --max-size 20), ~1.4M images.
#   10 workers: 14 ran WSL's 7.7 GB out of memory with 2M kolams.
# night_BM --model grid --backbone resnet34 --ctx-layers 4 --steps 60000 --eval-every 2000 --val-n 200 data/enc_2m_20 --size 384 --batch 24 --warmup 1000 --log-every 20 --workers 10 --aug real --not-kolam data/not_kolam --real-kolams data/real_kolams_line --samples 6 --save-every 200
# night 2: model A-M (seq: ResNet-34 + 4-layer decoder) on the same 2M kolams, only those up to 12 x 12
#   (a 12 x 12 is 312 tokens; 20 x 20 was 882 and filled the 4 GB GPU). Model A chosen over B: todo.md.
# night_AM --model seq --backbone resnet34 --max-dots 12 --steps 60000 --eval-every 2000 --val-n 300 data/enc_2m_20 --size 384 --batch 24 --warmup 1000 --log-every 20 --workers 8 --aug real --not-kolam data/not_kolam --real-kolams data/real_kolams_line --samples 6 --save-every 200
# 2M training pictures drawn ahead (kolam_gen render, milder look, effects on the GPU; ~90-100 GB, ~50 min):
#   kolams 3 x 3 to 20 x 20 from enc_2m_20, val kolams left out. Train: --rendered data/img_2m_20 (--max-dots picks sizes).
# sh export LD_LIBRARY_PATH=$HOME/kolam_core_model/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/lib/wsl/lib; kolam_gen/target/release/kolam_gen render data/enc_2m_20 -o data/img_2m_20 -n 2000000 --max-dots 20 --gpu   (done 2026-10-25: 2M pictures, 88 GB, 61 min)
# night 2: model A-M (seq, ResNet-34 + 4-layer decoder) on the rendered pictures up to 12 x 12 (1.46M), sized
#   for 20 x 20 (--grid-max 20) so night 3 continues from last.pt with --max-dots 20 --batch 16; batches of
#   similar sizes (--bucket). Probe: ~83 pictures/s, 2.6 GB of 4. 100k steps x 24 = 2.4M pictures, ~8 h + checks.
night_AM2 --model seq --backbone resnet34 --max-dots 12 --grid-max 20 --bucket --rendered data/img_2m_20 data/enc_2m_20 --steps 100000 --eval-every 4000 --val-n 200 --size 384 --batch 24 --warmup 1000 --log-every 20 --workers 8 --aug real --not-kolam data/not_kolam --real-kolams data/real_kolams_line --samples 6 --save-every 200
