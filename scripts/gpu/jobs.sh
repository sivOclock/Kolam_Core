# one line after another (launch.sh): "<run name> <train args>" trains (resumable), "sh <command>" runs it
# night 1: 2M unique kolams up to 20 x 20, then model B-M (~11.6M: ResNet-34 + 4 context layers), ~1.4M images
sh kolam_gen/target/release/kolam_gen generate -n 2000000 -o data/enc_2m_20 --min-size 3 --max-size 20 --threads 16 2>&1 | tail -1
night_BM --model grid --backbone resnet34 --ctx-layers 4 --steps 60000 --eval-every 2000 --val-n 200 data/enc_2m_20 --size 384 --batch 24 --warmup 1000 --log-every 20 --workers 14 --aug real --not-kolam data/not_kolam --real-kolams data/real_kolams_line --samples 6 --save-every 200
