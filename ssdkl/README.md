# SSDKL, DKL, VAT, Coreg Experiments

Download the UCI experiment data `ssdkl_uci_data.tar.gz` from:
```
https://www.dropbox.com/sh/o5nruvv6g4x8bzn/AAAnyHRcQpW1BAUxntQJ_4kVa?dl=0
```
Unzip the file to a path of your choice.

Compile pseudoinverse tensorflow op:
Run 
```
# navigate to ssdkl/models
bash compile_pinv.sh
```
If you do not have Eigen (C++ library), you may need to install it:
```
sudo apt install libeigen3-dev
```

Go into `config.yaml` and change the paths for `data_dir` and `results_dir`.
`data_dir` will be the path where you unzipped the UCI experiment data.
Run all UCI experiments: 
```
python run_all.py --config_file config.yaml
```

Print the results after running:
```
python print_results.py --config_file config.yaml
```

Under Construction: Poverty Experiments
