**install**
1. Install RHF of Cython version: `cd lib/cyrhf`, then `python setup.py install`
2. Run experiments: `python experiments_reg.py --datasets abalone cardio`, datasets: `abalone`, `cardio` are in the folder: `datasets/klf`
3. Parameters: you can find the parameters in the dict: `config`, which is in the main function of `experiments_reg.py`
4. Plot: if you want to draw the plots, set `config.draw_boxplot = True`

**data format**
1. Input: the input file is in `csv` format, which has the columns: feature index + label: `0,1,2,..,label`. Please refer to the file: `datasets/klf/abalone`
```
0,1,2,3,4,5,6,label
0.53,0.42,0.135,0.677,0.2565,0.1415,0.21,0
```
2. Output: `csv` file, please refer to: `results/rhf_regression_abalone.csv`, `jpg` file: `results/rhf_regression_abalone.jpg`