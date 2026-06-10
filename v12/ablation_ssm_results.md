# SSM Ablation Study Results

Steps per ablation: 3000

| Model | Params | Val BPC | Val Acc | ms/step |
|-------|--------|---------|---------|----------|
| A: Full V12.1 | 2,345,031 | 2.302 | 53.3% | 317 |
| B: Zero Context | 2,345,031 | 3.589 | 27.0% | 279 |
| C: SSM+MLP Only | 2,112,327 | 2.267 | 53.8% | 68 |
| D: No Sparsity | 2,345,031 | 2.275 | 53.4% | 279 |

## Deltas from baseline

- SSM context: +1.287 BPC
- Spectral components: -0.035 BPC
- Sparsity: -0.027 BPC
