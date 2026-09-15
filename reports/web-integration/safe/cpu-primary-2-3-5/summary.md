# Fixed CPU automatic decision: development revalidation

All values are percentages; brackets show95% asset-group bootstrap intervals. Each row uses the same100 development scenes and27 asset groups. No new neural inference.

## CPU app decision

| Views | Top-1 | Recall@2 | AP | MRR |
|---|---|---|---|---|
| 2 | 39.00 [30.09, 48.51] | 51.50 [42.48, 60.97] | 44.29 [35.85, 53.38] | 49.58 [40.44, 58.51] |
| 3 | 36.67 [27.67, 45.99] | 52.08 [44.61, 59.98] | 47.19 [39.72, 55.36] | 52.15 [45.29, 59.43] |
| 5 | 19.92 [12.21, 28.97] | 34.75 [25.41, 44.33] | 38.02 [31.12, 45.44] | 41.33 [34.28, 48.82] |

## Historical automatic baseline

| Views | Top-1 | Recall@2 | AP | MRR |
|---|---|---|---|---|
| 2 | 39.00 [30.09, 48.51] | 51.50 [42.48, 60.97] | 44.29 [35.85, 53.38] | 49.58 [40.44, 58.51] |
| 3 | 37.67 [29.08, 46.94] | 52.58 [45.02, 60.50] | 47.65 [40.22, 55.72] | 52.82 [46.08, 60.09] |
| 5 | 18.92 [11.00, 28.24] | 35.75 [26.81, 45.00] | 37.98 [31.27, 45.39] | 40.99 [34.13, 48.61] |

## Paired CPU minus historical differences

| Views | Top-1 difference (percentage points) | 95% interval |
|---|---|---|
| 2 | +0.00 | [0.00, 0.00] |
| 3 | -1.00 | [-2.53, 0.00] |
| 5 | +1.00 | [0.00, 3.09] |
