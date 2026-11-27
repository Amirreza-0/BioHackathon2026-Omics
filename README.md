# cureATime challenge: predicting drug action from transcriptomics

Cancer cells (A549 lung cancer) were treated with a drug; the L1000 assay measured how
978 landmark genes responded. Your task: from that fingerprint, predict which target
class the drug hits.

## Files (read-only mount)
| File | What it is |
| --- | --- |
| `X_train.csv.gz` | Training fingerprints. Rows = signatures, columns = genes (symbols). |
| `y_train.csv` | The mechanism class of each training signature. |
| `meta_train.csv` | compound, dose (uM), `cv_fold` for cross-validation. |
| `X_test.csv.gz` | The mystery compounds. Same columns, no labels. |
| `meta_test.csv` | compound and dose for the test signatures. |
| `X_unlabeled.csv.gz` | ~1,000 further compounds, no labels. Free to use. |
| `classes.txt` | The class names. These are exactly your submission columns. |
| `starter_kit/` | `starter_notebook.ipynb` (start here), baseline code and its scores. |

Values are robust z-scores against untreated controls, clipped to [-10, 10]; positive
means the gene went up. Row ids are `<compound>_d<dose>`: the same id is the same
treatment everywhere.

## Start here
Three ways in, pick one:

1. `starter_kit/starter_notebook.ipynb` in JupyterLab, run top to bottom. Seven steps from
   loading the data to a scored submission. Assumes no machine learning background.
2. `starter_kit/solution_minimal.py` - the same idea in 20 lines. Run it and you have a
   valid `submission.csv`. Read it to see exactly what the task is.
3. `starter_kit/solution_template.py` - three functions marked TODO (preprocess, model,
   combine doses). Change any of them and the self-check tells you whether you improved,
   using training data only.

To use the scripts instead:
```bash
cp -r /data/cureatime/starter_kit ~/kit && cd ~/kit
ln -s /data/cureatime data            # the code expects data/ to contain the files
python train.py --model all           # ~2 minutes on CPU, writes results/submission_*.csv
python interpret.py                   # mechanism signatures + nearest neighbours
```

## Submission
CSV with a `compound` column plus one column per class in `classes.txt`, one row per
test compound, holding a score or probability. Up to 3 submissions; the last counts.

## Three rules
1. A compound never spans train and test. Validate with `cv_fold`, which already groups
   by compound - a random split will flatter you by a lot.
2. Use what is here plus any public pretrained model. Do not look up the mystery
   compounds' mechanism in external databases.
3. Report how you validated. With ~30 test compounds, a few percent is noise.

## Worth your attention
- Low doses often look like untreated cells. Decide which signatures to trust.
- Some classes are loud (HDAC, proteasome), others nearly silent. Ask why before
  assuming it is a modelling problem.
- A549 is KRAS-mutant and does not depend on EGFR. Keep that in mind when a class
  refuses to be learned.
