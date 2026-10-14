# UCI Iranian Churn (dataset #563)

- Source page: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- Downloaded file: https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip
- Downloaded on: 2026-10-14
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the UCI page
- DOI: 10.24432/C5JW3Z
- Citation: Iranian Churn [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5JW3Z

| File | Bytes | SHA256 |
|---|---:|---|
| `iranian+churn+dataset.zip` (as downloaded) | 41,943 | `696c3a1812267980751f30d19193a1b430ac4ad76172bb089e664c96813ead66` |
| `Customer Churn.csv` (extracted from the zip, unchanged) | 131,728 | `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1` |

Shape: 3,150 rows, 14 columns (13 features + `Churn`), all numeric, no missing values.
Target: `Churn` (1 = churn, 0 = non-churn).

From the UCI description: every attribute except `Churn` is aggregated over the first 9 months;
`Churn` is the customer's state at the end of month 12 (a 3-month planning gap). Column notes from UCI:
`Complains` 0/1; `Charge  Amount` ordinal 0 (lowest) to 9 (highest); `Age Group` ordinal 1–5;
`Tariff Plan` 1 = pay as you go, 2 = contractual; `Status` 1 = active, 2 = non-active;
`Customer Value` = "the calculated value of customer" (formula not published).
Some column names contain double spaces (`Call  Failure`, `Subscription  Length`, `Charge  Amount`);
they are kept as in the original file.
