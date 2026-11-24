## Table 2: coverage (%), Mistral 7B, target 80%

| Category | Original | Adaptive | Group-wise | Boosted |
|---|---|---|---|---|
| Inventions | 85.60 ± 3.34 | 81.03 ± 6.23 | 80.10 ± 5.22 | 82.41 ± 4.72 |
| Biographies | 81.39 ± 3.74 | 79.53 ± 5.51 | 79.67 ± 5.55 | 80.47 ± 5.65 |
| Artworks | 75.46 ± 4.49 | 81.06 ± 5.77 | 80.49 ± 5.16 | 80.69 ± 6.25 |
| Books | 84.34 ± 3.70 | 80.88 ± 5.60 | 80.43 ± 5.29 | 81.98 ± 5.61 |
| Cities | 79.25 ± 3.87 | 79.87 ± 6.16 | 80.02 ± 4.90 | 81.90 ± 4.78 |
| Movies | 83.57 ± 3.27 | 79.14 ± 6.28 | 80.82 ± 4.68 | 81.38 ± 6.10 |
| Landmarks | 73.34 ± 3.60 | 79.04 ± 5.34 | 80.09 ± 4.75 | 82.33 ± 5.85 |
| Events | 78.04 ± 3.28 | 81.54 ± 4.77 | 80.39 ± 4.48 | 81.87 ± 5.09 |
| Cov. bias | 3.60 | 0.87 | 0.33 | 1.63 |

## Table 3: coverage (%), Gemma-3 12B, target 80%

| Category | Original | Adaptive | Group-wise | Boosted |
|---|---|---|---|---|
| Inventions | 86.07 ± 3.22 | 80.07 ± 6.14 | 79.32 ± 5.06 | 82.22 ± 4.83 |
| Biographies | 75.39 ± 4.07 | 79.76 ± 6.09 | 79.72 ± 4.73 | 82.54 ± 4.81 |
| Artworks | 65.11 ± 4.32 | 81.15 ± 5.81 | 80.00 ± 4.90 | 81.82 ± 6.15 |
| Books | 86.05 ± 3.06 | 80.98 ± 5.36 | 80.76 ± 4.42 | 82.11 ± 5.68 |
| Cities | 86.14 ± 3.31 | 81.44 ± 5.24 | 80.99 ± 4.95 | 82.15 ± 5.43 |
| Movies | 87.69 ± 3.45 | 80.03 ± 5.24 | 80.15 ± 4.66 | 81.17 ± 5.11 |
| Landmarks | 79.82 ± 3.50 | 79.33 ± 5.90 | 80.54 ± 4.85 | 82.17 ± 5.22 |
| Events | 74.04 ± 4.62 | 77.73 ± 6.49 | 79.42 ± 5.43 | 82.78 ± 5.35 |
| Cov. bias | 6.45 | 0.85 | 0.50 | 2.12 |

## Table 4: ablation, Mistral 7B, 50 splits, target 80%

| Ablation component | Adaptive wins | Mean bias reduction | Bias range |
|---|---|---|---|
| Number of PCA components | 4/4 | 74.3% | 0.52–1.49 |
| Embedding model | 4/4 | 80.9% | 0.52–0.84 |
| D_cal1/D_cal2 split | 5/5 | 81.9% | 0.52–0.74 |
| MLP configuration | 5/5 | 74.4% | 0.42–1.90 |

Original has a bias of 3.63 and the oracle Group-wise 0.36.

## Table 5: removed claims (%), Mistral 7B, target 80%

| Category | Original | Adaptive | Group-wise | Boosted |
|---|---|---|---|---|
| Inventions | 87.49 ± 0.89 | 83.19 ± 3.02 | 81.69 ± 1.60 | 83.68 ± 5.63 |
| Biographies | 82.46 ± 0.86 | 81.49 ± 2.35 | 81.39 ± 1.66 | 85.38 ± 5.88 |
| Artworks | 90.12 ± 0.64 | 92.05 ± 1.65 | 92.00 ± 0.94 | 91.81 ± 4.16 |
| Books | 85.17 ± 1.08 | 83.73 ± 2.49 | 82.21 ± 1.83 | 86.20 ± 5.84 |
| Cities | 82.50 ± 0.81 | 83.26 ± 2.29 | 83.00 ± 1.01 | 87.52 ± 4.76 |
| Movies | 83.90 ± 0.87 | 80.53 ± 3.24 | 82.21 ± 1.42 | 82.40 ± 8.04 |
| Landmarks | 82.01 ± 1.12 | 85.49 ± 2.32 | 86.58 ± 1.65 | 86.14 ± 6.76 |
| Events | 81.46 ± 1.18 | 83.84 ± 3.27 | 83.49 ± 1.88 | 83.15 ± 6.90 |

## Table 6: removed claims (%), Gemma-3 12B, target 80%

| Category | Original | Adaptive | Group-wise | Boosted |
|---|---|---|---|---|
| Inventions | 88.06 ± 0.93 | 82.98 ± 2.58 | 82.24 ± 2.56 | 89.03 ± 3.80 |
| Biographies | 79.67 ± 0.94 | 82.63 ± 2.88 | 81.43 ± 1.25 | 90.74 ± 4.63 |
| Artworks | 81.14 ± 0.87 | 88.61 ± 1.97 | 87.92 ± 1.03 | 93.31 ± 3.73 |
| Books | 84.05 ± 0.87 | 79.85 ± 2.89 | 79.92 ± 1.52 | 87.11 ± 5.31 |
| Cities | 86.91 ± 0.63 | 84.11 ± 1.90 | 83.56 ± 1.08 | 89.40 ± 4.79 |
| Movies | 85.06 ± 0.79 | 80.59 ± 2.73 | 80.25 ± 1.43 | 87.80 ± 5.41 |
| Landmarks | 80.66 ± 0.97 | 79.48 ± 2.96 | 81.13 ± 1.68 | 85.06 ± 7.21 |
| Events | 78.67 ± 1.20 | 80.56 ± 3.27 | 81.63 ± 1.64 | 86.11 ± 4.76 |

## Table 7: coverage (%), Mistral 7B, target 95%

| Category | Original | Adaptive | Group-wise |
|---|---|---|---|
| Inventions | 97.84 ± 1.40 | 96.37 ± 2.48 | 95.76 ± 2.65 |
| Biographies | 97.08 ± 1.50 | 94.23 ± 2.95 | 94.99 ± 2.80 |
| Artworks | 94.89 ± 2.23 | 95.61 ± 2.92 | 95.43 ± 2.77 |
| Books | 96.76 ± 1.71 | 95.35 ± 2.67 | 95.47 ± 2.58 |
| Cities | 94.44 ± 2.08 | 95.27 ± 3.23 | 95.25 ± 2.51 |
| Movies | 95.93 ± 1.86 | 94.28 ± 2.95 | 94.92 ± 2.68 |
| Landmarks | 90.60 ± 2.77 | 94.67 ± 2.49 | 94.82 ± 2.99 |
| Events | 92.69 ± 2.36 | 94.67 ± 2.75 | 95.64 ± 2.52 |
| Cov. bias | 1.87 | 0.59 | 0.35 |

## Appendix B.2: efficiency, target 80%

| Model | Method | Retained claims per prompt | Correct claims removed (%) |
|---|---|---|---|
| Mistral 7B | Original | 3.66 | 82.2 |
| Mistral 7B | Adaptive | 3.73 | 82.0 |
| Gemma-3 12B | Original | 4.64 | 81.3 |
| Gemma-3 12B | Adaptive | 4.82 | 80.7 |
