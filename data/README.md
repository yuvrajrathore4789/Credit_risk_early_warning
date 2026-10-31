# Data setup

This project uses the **UCI "Default of Credit Card Clients"** dataset (Yeh & Lien, 2009):
30,000 credit-card holders in Taiwan (April–September 2005) with a binary target indicating
default payment in the following month.

* Source: <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
* Licence: CC BY 4.0 (see the UCI page).

## Get the file

Either of these works — `train.py` auto-detects the file and validates the columns and target:

1. **Kaggle mirror (CSV)** – save as `data/UCI_Credit_Card.csv`
   (<https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset>).
2. **UCI original (XLS)** – download from the UCI page above and save as
   `data/default of credit card clients.xls` (needs `openpyxl`/`xlrd`; the loader handles the
   extra banner row).

If the file lives elsewhere: `python train.py --data-path /path/to/file.csv`.

## Columns (after standardization)

| Column | Meaning |
|---|---|
| `id` | customer identifier — **never used as a feature** |
| `limit_bal` | credit limit (NT$) |
| `sex`, `education`, `marriage`, `age` | demographics (`sex` is excluded from the model by default, see README) |
| `pay_1` … `pay_6` | repayment status, month 1 = latest (UCI calls the first one `PAY_0`). -2 no consumption, -1 paid in full, 0 revolving credit, 1–9 months delayed |
| `bill_amt1` … `bill_amt6` | bill statement amounts |
| `pay_amt1` … `pay_amt6` | previous payment amounts |
| `default` | target (UCI: `default.payment.next.month`) |

## Limitations

* Taiwanese card portfolio from 2005 — **not representative of Indian SME / NBFC lending**.
* It is not any lender's internal data (including Protium's).
* Files in this folder are git-ignored; do not commit the dataset or the generated SQLite DB.
