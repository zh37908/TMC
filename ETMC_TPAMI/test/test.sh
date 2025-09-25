python /home/hzhaobi/Multired/TMC/ETMC_TPAMI/test/test_TMC_uncertainty_clean.py \
  --savedir /home/hzhaobi/Multired/savepath/TMC_channel/nyud/DeCUR/snr20/run1/train_TMC_any_evidential_DeCUR_static_snr20_run1\
  --model_type TMC_channel \
  --data_path /home/hzhaobi/Multired/nyud2 \
  --save_fig \
  --out_dir /home/hzhaobi/Multired/TMC/ETMC_TPAMI/plots\
  --snr 20

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

python /home/hzhaobi/Multired/TMC/ETMC_TPAMI/test/test_TMC_uncertainty_retransmit.py \
  --savedir /home/hzhaobi/Multired/savepath/TMC_channel/nyud/No_pretrain/snr20/run1/train_TMC_any_evidential_No_pretrain_static_snr20_run1 \
  --ckpt_file epoch_48_model_best.pth \
  --model_type TMC_channel \
  --data_path /home/hzhaobi/Multired/nyud2 \
  --split val \
  --batch_sz 32 --n_workers 8 \
  --retransmit \
  --reject_score u_evi \
  --rt_trigger high \
  --rt_pick_view max \
  --ds_discount \
  --target_coverage 0.7 \
  --report_selective

  python /home/hzhaobi/Multired/TMC/ETMC_TPAMI/test/test_TMC_uncertainty_retransmit.py \
  --savedir /home/hzhaobi/Multired/savepath/TMC_channel/nyud/DeCUR/snr20/run1/train_TMC_any_evidential_DeCUR_static_snr20_run1 \
  --ckpt_file epoch_48_model_best.pth \
  --model_type TMC_channel \
  --data_path /home/hzhaobi/Multired/nyud2 \
  --split val \
  --batch_sz 32 --n_workers 8 \
  --retransmit \
  --reject_score u_evi \
  --rt_trigger high \
  --rt_pick_view max \
  --target_coverage 1.0\
  --report_selective


  python /home/hzhaobi/Multired/TMC/ETMC_TPAMI/test/test_TMC_uncertainty_retransmit.py \
  --savedir /home/hzhaobi/Multired/savepath/TMC_channel/nyud/DeCUR/snr20/run1/train_TMC_any_evidential_DeCUR_static_snr20_run1 \
  --ckpt_file model_best.pt \
  --model_type TMC_channel \
  --data_path /home/hzhaobi/Multired/nyud2 \
  --split test \
  --batch_sz 32 --n_workers 8 \
  --retransmit \
  --reject_score u_evi \
  --rt_trigger high \
  --target_coverage 0.9 \
  --snr 5 \
  --report_selective

  python /home/hzhaobi/Multired/TMC/ETMC_TPAMI/test/test_TMC_dynamic_retransmit.py \
  --savedir /home/hzhaobi/Multired/savepath/TMC_channel_dynamic/nyud/DeCUR/range0-20/dynamic/run1/train_TMC_any_evidential_DeCUR_range0-20_dynamic_run1 \
  --ckpt_file model_best.pt \
  --data_path /home/hzhaobi/Multired/nyud2 \
  --split test \
  --batch_sz 32 --n_workers 8 \
  --snr_min 0 --snr_max 20 \
  --retransmit \
  --reject_score u_evi \
  --rt_trigger high \
  --target_coverage 0.9 \
  --rt_max_trials 3 \
  --rt_discard_unqualified \
  --report_selective