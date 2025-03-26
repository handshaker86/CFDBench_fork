<h1 align="center">CFDBench fork</h1>

This is a repository forked from [luo-yining/CFDBench](https://github.com/handshaker86/CFDBench_fork). This repository is mainly used to derive analysis data for [JI-FDA-LAB/Tokenize_Flow_Field](https://github.com/JI-FDA-LAB/Tokenize-Flow-Field).


### Main Features
- Add arg `velocity_dim` to control which velocity dimension you want to predict. Add boolean args `visualize` and `cal_case_accuracy` to control whether to visualize and calculate case prediction accuracy .

- Add `get_case_accuracy` function which calculates the SMI(accuracy) of each prediction case. 

- Add `cal_loss` function which calculates loss combining both u and v velocity dimensions.

- Add `cal_predict_time` function which calculates total time for predicting both u and v velocity dimensions.


### How to Run
**Note**: The original model architecture only support **predicting one dimension at a time**. Therefore, we have to predict both u and v dimensions before calculating accuracy and loss.

1. Modify `args.py`. Modify `model`, `data_name` and `data_dir`. Set `velocity_dim` to 0 and set `cal_case_accuracy` as False.

2. Run command `python auto_train.py` in root directory.

3. Modify `args.py`. Set `velocity_dim` to 1 and set `cal_case_accuracy` as True. (You can also set `visualize` as True if visualizing is needed).

4. Repeat step.2 