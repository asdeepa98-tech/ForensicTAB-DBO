# ForensicTAB (DBO-optimised)

Multi-label prediction of recoverable forensic artifacts (WhatsApp Chats, SMS Messages, Call Logs, Contacts, Media Files, Browser History) from device and
extraction metadata (Device Make, Device Model, OS Type, OS Version, Extraction Tool, Tool Version, Extraction Method) with a TabTransformer + KAN model whose hyper-parameters are
tuned by the Dung Beetle Optimizer (DBO).

## Reproduce (in order)
1. `pip install -r requirements.txt`
2. Obtain the dataset - see `DATA_ACCESS.md` - and place it where the notebook/`src/` can read it.
3. Load the fixed partition from `splits/*.npy` (indices into the de-duplicated table; `splits/raw_row_id.npy`
   maps them to raw-file rows). Categorical vocabularies are in `mappings/categorical_mappings.json`
   (training split only, index 0 = unknown).
4. Seeds are in `seeds.json`; final model + DBO settings are in `configs/forensictab_dbo.yaml`;
   baseline configs go in `configs/baselines/` (use `_template.yaml`).
5. Run `ForensicTAB_DBO.ipynb` top to bottom (set `OPTIMIZATION_MODE = "full"` for the full DBO budget).
6. Outputs: `results/test_predictions.csv` (per-record labels, probabilities, predictions) and
   `results/metrics_summary.csv` (per-label, macro, micro). Decision threshold fixed at 0.5.

## Layout
`src/` code - `configs/` settings - `splits/` fixed partition - `mappings/` encodings - `results/` outputs
