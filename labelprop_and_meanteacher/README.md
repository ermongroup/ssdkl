Run `python labelprop.py --dataset all` to run label propagation (regression
version) for all UCI datasets. This will save results in `results_labelprop/`.

Run `python mean_teacher.py --dataset all` to run mean teacher for all
UCI datasets. This will save results in `results_mean_teacher/`. 

To print results for a given method, run `python print_results.py --results_dir <RESULT_DIR>`
where `RESULT_DIR` is either `results_labelprop` or `results_mean_teacher`. 
