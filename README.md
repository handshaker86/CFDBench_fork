# CFDBench Fork

This repository is a fork of [luo-yining/CFDBench](https://github.com/handshaker86/CFDBench_fork). It is primarily used to derive analysis data for the [JI-FDA-LAB/Tokenize_Flow_Field](https://github.com/JI-FDA-LAB/Tokenize-Flow-Field) project.

---

## Main Features

- **`velocity_dim` Argument**: Allows you to specify which velocity dimension (u or v) to predict.
- **Visualization and Accuracy Control**: 
  - `visualize`: Boolean argument to enable/disable visualization of predictions.
  - `cal_case_accuracy`: Boolean argument to enable/disable calculation of case prediction accuracy.
- **Accuracy Calculation**: 
  - `get_case_accuracy`: Function to calculate the SMI (accuracy) for each prediction case.
- **Loss Calculation**:
  - `cal_loss`: Function to compute loss by combining both u and v velocity dimensions.
- **Prediction Time Calculation**:
  - `cal_predict_time`: Function to measure the total time required for predicting both u and v velocity dimensions.

---

## How to Run

**Note**: The original model architecture only supports **predicting one dimension at a time**. Therefore, you need to predict both u and v dimensions separately before calculating accuracy and loss.

### Steps:

1. **Modify `args.py`**:
   - Update `model`, `data_name`, and `data_dir`.
   - Set `velocity_dim` to `0` (for u dimension).
   - Set `cal_case_accuracy` to `False`.

2. **Run Training**:
   ```
   python auto_train.py
   ```

3. **Modify args.py Again:**
   - Set velocity_dim to 1 (for v dimension).
   - Set cal_case_accuracy to True.
   - Optionally, set visualize to True if you want to visualize the results.

4. **Run Training Again:**
   ```
   python auto_train.py
   ```

## Contribution
Feel free to open issues or submit pull requests if you have suggestions or improvements!

