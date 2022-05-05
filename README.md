1. Install RHF of Cython version: `cd lib/cyrhf`, then `python setup.py install`
2. Run experiments: `python experiments_reg.py --datasets abalone cardio`, datasets: `abalone`, `cardio` are in the folder: `datasets/klf`
3. Parameters: you can find the parameters in the dict: `config`, which is in the main function of `experiments_reg.py`
4. Plot: if you want to draw the plots, set `config.draw_boxplot = True`