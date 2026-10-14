# Raw datasets

Files in this folder are kept exactly as downloaded. Scripts read them; nothing writes here.

| Dataset | File | Rows | Target column | Source | License |
|---|---|---:|---|---|---|
| IBM Telco Customer Churn (IBM Cognos Analytics 11.1.3 base sample) | `Telco_customer_churn.xlsx` | 7,043 | `Churn Value` (1 = churned; `Churn Label` is the same as Yes/No) | IBM Cognos Analytics 11.1.3 sample data; the download URL of this copy was not recorded in the repo | **IBM publishes no license for this file.** Used here for educational purposes only. |
| UCI Iranian Churn (#563) | `iranian_churn/Customer Churn.csv` (+ original zip) | 3,150 | `Churn` (1 = churn) | https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset, DOI 10.24432/C5JW3Z | CC BY 4.0 |
| Cell2Cell (Duke / Teradata) | not downloaded yet | — | — | see `docs/cleanup/inventory.md` §5 | to verify |

Notes:

- IBM Telco also contains `Churn Score`, `CLTV` and `Churn Reason`. These are produced by IBM's
  own analytics or recorded after the outcome, so they must not be used as features
  (measured in `docs/cleanup/leakage_audit.md`). The location columns (`CustomerID`, `Count`,
  `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`) are identifiers.
- `cleaned_telco.csv` is an output of the old ETL, not a raw file (to be removed in Phase 2).
- Row counts above are of the files as stored; duplicate rows are removed by the scripts
  before splitting (UCI Iranian has 300 exact duplicate rows).
