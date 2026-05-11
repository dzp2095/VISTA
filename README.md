# VISTA: Multimodal Test-Time Adaptation

> This work has been early accepted by MICCAI 2026. The paper and BibTeX entry will be updated once the arXiv preprint or official proceedings version is available.

Datasets: BraTS (BraTS25-GLIPRE as source; BraTS23-SSA and BraTS24-PED as target). Set `csv_path` and `save_dir` in `configs/dataset/*.yaml` and `configs/task/*.yaml` to your data and output paths.

**Setup:** `pip install -r requirements.txt`

**Train source model (e.g. BraTS):**  
`python main.py task=brats_source dataset=brats model=unet training=default`

**Run TTA (example):**  
`python main.py task=brats_ours dataset=brats_SSA model=unet tta.ckpt_path=/path/to/best_model.pth training.epochs=1 training.batch_size=1`

See `train_brats.sh`.

## Citation

If you find this repository useful, please consider citing our paper. The BibTeX entry will be added once the arXiv preprint or official MICCAI proceedings version is available.

```bibtex
% Coming soon.
```
