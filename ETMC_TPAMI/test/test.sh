python TMC/ETMC_TPAMI/test_TMC_uncertainty_clean.py \
  --savedir /home/hzhaobi/Multired/savepath/TMC_channel/nyud/No_pretrain/snr20\
  --model_type TMC \
  --data_path /home/hzhaobi/Multired/nyud2 \
  --save_fig \
  --out_dir /home/hzhaobi/Multired/TMC/ETMC_TPAMI/plots \

python /home/hzhaobi/Multired/TMC/ETMC_TPAMI/test/test_TMC_uncertainty_retransmit.py \
    --savedir /home/hzhaobi/Multired/savepath/TMC_channel/nyud/No_pretrain/snr20/run1/train_TMC_any_evidential_No_pretrain_static_snr20_run1\
    --model_type TMC_channel \
    --data_path /home/hzhaobi/Multired/nyud2

python /home/hzhaobi/Multired/TMC/ETMC_TPAMI/test/test_TMC_uncertainty_dynamic_snr.py \
  --savedir /home/hzhaobi/Multired/savepath/TMC_channel/nyud/No_pretrain/snr20\
  --model_type TMC_channel \
  --data_path /home/hzhaobi/Multired/nyud2 \
  --save_fig \
  --out_dir /home/hzhaobi/Multired/TMC/ETMC_TPAMI/plots
#   --out_dir /home/hzhaobi/Multired/TMC/ETMC_TPAMI/plots