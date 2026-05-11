GPU_IDS="[7]"

python main.py \
  task=brats_ours \
  task.run_name=brats_ours_SSA \
  dataset=brats_SSA \
  model=unet \
  training=default \
  training.epochs=1 \
  training.batch_size=1 \
  training.eval_batch_size=1 \
  training.num_workers=8 \
  training.gpu_ids=${GPU_IDS} \
  training.optimizer=adam \
  training.optimizers.adam.lr=1e-4 \
  tta.ckpt_path=/path/to/outputs/brats/brats_baseline/20260129_123137/checkpoints/checkpoints/best_model.pth

python main.py \
  task=brats_ours \
  task.run_name=brats_ours_PED \
  dataset=brats_PED \
  model=unet \
  training=default \
  training.epochs=1 \
  training.batch_size=1 \
  training.eval_batch_size=1 \
  training.num_workers=8 \
  training.gpu_ids=${GPU_IDS} \
  training.optimizer=adam \
  training.optimizers.adam.lr=1e-4 \
  tta.ckpt_path=/path/to/outputs/brats/brats_baseline/20260129_123137/checkpoints/checkpoints/best_model.pth