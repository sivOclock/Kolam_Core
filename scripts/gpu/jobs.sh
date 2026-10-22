# one line after another (launch.sh): "<run name> <train args>" trains (resumable), "sh <command>" runs it
# night 1: model B-M (~11.6M: ResNet-34 + 4 context layers) on the 2M kolams up to 20 x 20 (made by
#   kolam_gen generate -n 2000000 -o data/enc_2m_20 --min-size 3 --max-size 20), ~1.4M images.
#   10 workers: 14 ran WSL's 7.7 GB out of memory with 2M kolams.
night_BM --model grid --backbone resnet34 --ctx-layers 4 --steps 60000 --eval-every 2000 --val-n 200 data/enc_2m_20 --size 384 --batch 24 --warmup 1000 --log-every 20 --workers 10 --aug real --not-kolam data/not_kolam --real-kolams data/real_kolams_line --samples 6 --save-every 200
