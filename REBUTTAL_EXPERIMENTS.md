# Rebuttal metrics

Evaluate a saved CFDBench velocity prediction against matching target frames:

```powershell
python src/rebuttal_evaluate.py --target target.pt --prediction results/run/test/preds.pt --output-dir results/run/rebuttal --dx 0.01 --dy 0.01
```

The evaluator accepts `.pt`, `.npy`, or `.npz`, handles CFDBench's flattened
`(frames*2, h, w)` prediction layout, and writes aggregate JSON plus per-frame
CSV. Use the dataset's physical native-grid spacing.
