# Data

Raw data is not committed and is never downloaded to the workstation.

| Field | Value |
|---|---|
| Kaggle slug | `mlg-ulb/creditcardfraud` |
| Kaggle reference | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| License | Database: Open Database License (same dataset as FD01; not re-checked for this repo) |
| Description | Real anonymized card transactions, September 2013 (284,807 rows, 492 frauds, PCA features V1-V28) |
| Notebook read path | `/kaggle/input/creditcardfraud/creditcard.csv` (attach the dataset in a Kaggle Notebook) |
| Local override | Set `FRAUD_DATA_PATH` to a local copy, for development only |
| Retrieval date | Not retrieved on this workstation; data is read in place in Kaggle |

The `Time` column is seconds from the first transaction, so the full dataset covers about two days.

The notebook writes `data/fd03_test_scores.csv` (test split only: `Time`, `Class`, `if_score`, `lof_score`). It is derived output, git-ignored, and is created only when the notebook runs on the real data.
