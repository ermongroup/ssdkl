# Semi-supervised Deep Kernel Learning

This is the code that accompanies the paper [Semi-supervised Deep Kernel Learning: Regression with Unlabeled Data by Minimizing Predictive Variance](https://arxiv.org/abs/1805.10407)

Install via `pip install -e .` in this directory in
a NEW virtualenv.

- Experiments for SSDKL, DKL, VAT, Coreg are in the directory `ssdkl`.
- Experiments for Label Propagation and Mean Teacher are in `labelprop_and_meanteacher`.
- Experiments for VAE are in the directory `vae`.

For more detailed instructions, please see the README files in each directory.

Tested with Python 2.7.12.

If you find this code useful in your research, please cite
```
@article{jeanxieermon_ssdkl_2018,
  title={Semi-supervised Deep Kernel Learning: Regression with Unlabeled Data by Minimizing Predictive Variance},
  author={Jean, Neal and Xie, Sang Michael and Ermon, Stefano},
  journal={Neural Information Processing Systems (NIPS)},
  year={2018},
}
```
