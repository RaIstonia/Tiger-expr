python ./RQ-VAE/main.py \
  --device cuda:0 \
  --data_path ./data/Instruments/Instruments.emb-all-MiniLM-L6-v2-td.npy \
  --alpha 0.01 \
  --beta 0.0001 \
  --cf_emb ./RQ-VAE/ckpt/Instruments-32d-sasrec.pt\
  --ckpt_dir ../checkpoint/


# --epoch 10 \
# --cf_emb 作为条件嵌入，用于监督，使用的是SASRec模型的输出