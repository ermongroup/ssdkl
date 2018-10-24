## VAE experiment

Run `bash train_vaes.sh` to train VAEs on all UCI datasets.

This will save VAE model checkpoints for each dataset in `vae_checkpoints/`.
Each dataset is done when their respective directories have the file
'best.ckpt'.

After all datasets are finished, run `python evaluate_vae.py` to extract features using the VAEs and
train an MLP on these features, using cross validation, and print results.
