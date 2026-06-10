# V14: Completing the Theory

## Status: Design Phase

V14 exists because V13 has the right geometry but the wrong dynamics. The constellation representation (tokens as sparse spectral dots, shared modes as connections, Parseval inner products) is mathematically sound. But the three mechanisms that give the theory its expressive power — gauge transport, Langevin settling, and proximal sparsity — are not implemented.

## Documents

| File | Contents |
|---|---|
| [v13_diagnosis.md](v13_diagnosis.md) | Why V13's loss plateaus at CE ~2.17 after ~500 steps |
| [mathematical_foundations.md](mathematical_foundations.md) | The five axioms, Fourier duality, spectral sparsity theorems, constellation geometry |
| [theory_vs_implementation_gap.md](theory_vs_implementation_gap.md) | The three missing load-bearing mechanisms and what each needs to look like |
| [scaling_analysis.md](scaling_analysis.md) | O(n) vs O(n^2) complexity, inference memory, parameter scaling, associative recall |
| [version_history.md](version_history.md) | What each version v1-v13 taught, the recurring pattern of losing content dependence |
| [v14_design_requirements.md](v14_design_requirements.md) | Concrete requirements, implementation order, success criteria, ablation plan |

## The Core Thesis

The theory prescribes three operations that together form the forward-reverse loop:

```
Sparse spectral (constellation)
    |
    v
Gauge transport (Wilson line: content-dependent phase rotation in fiber)
    |
    v
Field reconstruction (Parseval read: dense messages from spectral accumulation)
    |
    v
Langevin settling (Hopfield energy descent: iterative content-addressable retrieval)
    |
    v
Proximal sparsity (soft-thresholding: collapse back to few active modes)
    |
    v
Sparse spectral (updated constellation)
```

V13 implements only the middle step (field reconstruction via Parseval read). V14 must complete the loop.

## The Bar to Clear

The V12.2 ablation showed SSM+MLP alone matched the full spectral model. V14's spectral machinery must **measurably outperform** the SSM+MLP ablation — not just match it. If it can't, the geometry is decorative regardless of how elegant the math is.
