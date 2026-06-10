# Study Plan: The Geometry of Language

## Who This Is For
Someone with college-level calculus and linear algebra who wants to deeply understand every mathematical concept in the thesis "The Geometry of Language" and the SGST codebase — well enough to explain any concept without hesitation.

## How to Use This Plan
- Work through units **in order** (dependencies build on each other)
- Each unit has: **readings**, **concept summaries**, **worked examples with solutions**, and **comprehension questions**
- Do NOT skip the problems. The goal is to drill the math until it's automatic.
- Papers referenced as `papers/filename.pdf` are in the repo's `papers/` directory
- Thesis references point to `/thesis_constrained/` chapter files or the PDF at `Downloads/thesis_constrained-5.pdf`
- Topology deep-dives are in the repo's `topology/` directory

## Estimated Time
~150-200 hours total (10-15 hours per unit, 16 units)

## Dependency Graph

```
Unit 01 (Linear Algebra) ──────────────────────────────┐
Unit 02 (Calculus & Optimization) ─────────────────────┤
                                                        ├── Unit 05 (Diff. Geometry) ── Unit 06 (Fiber Bundles) ── Unit 07 (Finsler)
Unit 03 (Fourier Analysis) ───── Unit 08 (Compressed   │
                                  Sensing)              │
Unit 04 (Graph Theory) ────────────────────────────────┤
                                                        ├── Unit 10 (Neural Architectures)
Unit 09 (Hopfield Networks) ───────────────────────────┘
                                                        
Unit 11 (Spectral Methods in DL) ── requires 03, 04, 10
Unit 12 (SGST Architecture) ──────── requires 06, 08, 09, 10, 11
Unit 13 (CurvBias) ───────────────── requires 06, 10
Unit 14 (Spectral Sparsity Hypothesis) ── requires 03, 08, 12
Unit 15 (Full Synthesis) ─────────── requires ALL previous
```

## Unit Index

| Unit | Title | Core Math | Thesis Connection |
|------|-------|-----------|-------------------|
| 01 | Linear Algebra Foundations | Eigendecomposition, SVD, rank, complex vectors | Rank collapse, oversmoothing |
| 02 | Calculus & Optimization | Gradients, Hessians, convexity, gradient descent | Energy landscapes, training |
| 03 | Fourier Analysis | DFT, FFT/IFFT, frequency domain, Parseval's theorem | Spectral representations |
| 04 | Graph Theory & Spectral Methods | Adjacency, Laplacian, spectral gap, graph filters | GNN message passing |
| 05 | Differential Geometry | Manifolds, metrics, connections, curvature, geodesics | Manifold hypothesis |
| 06 | Fiber Bundles & Gauge Theory | Bundles, connections, parallel transport, holonomy, Wilson lines | Attention as gauge connection |
| 07 | Finsler Geometry | Asymmetric metrics, directional dependence | Causal structure of language |
| 08 | Compressed Sensing & Sparsity | Donoho-Stark, RIP, basis pursuit, soft-thresholding | Spectral sparsity |
| 09 | Hopfield Networks & Energy Models | Classical/modern Hopfield, exponential capacity, Fenchel-Young | Associative memory |
| 10 | Neural Architectures | Transformers, attention, GNNs, SSMs, parallel scan | Architecture design |
| 11 | Spectral Methods in Deep Learning | FNO, spectral convolution, reaction-diffusion | $V_{12}$ architecture |
| 12 | The SGST Architecture | Full architecture, evolution $V_5$--$V_{16}$, ablation analysis | Thesis Ch. 5-6 |
| 13 | CurvBias Position Encoding | Gauge theory hierarchy, geometric position encoding | Thesis Ch. 6.7 |
| 14 | Spectral Sparsity Hypothesis | Uncertainty principle, emergent sparsity, grid cells | Thesis Ch. 7.2 |
| 15 | Full Synthesis | Convergence of 6 research lines, open problems | Thesis Ch. 7-8, `topology/SYNTHESIS.md` |

## Key Free Resources Used Throughout

**Textbooks (all freely available online):**
- Strang, *Linear Algebra and Its Applications* (MIT OCW)
- Boyd & Vandenberghe, *Convex Optimization* (Stanford, free PDF)
- Bracewell, *The Fourier Transform and Its Applications* (classic, library)
- do Carmo, *Riemannian Geometry* (standard graduate text)
- Lee, *Introduction to Smooth Manifolds* (Springer GTM)
- Nakahara, *Geometry, Topology and Physics* (physics-friendly intro)
- Baez & Muniain, *Gauge Fields, Knots and Gravity* (accessible gauge theory)

**Video Lectures:**
- 3Blue1Brown: *Essence of Linear Algebra* (YouTube, free)
- 3Blue1Brown: *Essence of Calculus* (YouTube, free)
- Eigenchris: *Tensor Calculus* series (YouTube, free — outstanding for diff. geometry)
- Frederic Schuller: *Lectures on Geometrical Anatomy of Theoretical Physics* (YouTube, free)

**Repo Resources:**
- `topology/` — Deep-dive documents on gauge theory, holonomy, metrics, transport
- `topology/lit_review/` — Literature reviews covering gauge theory in NNs, Finsler geometry, SSMs, unitary RNNs, topological DL
- `topology/SYNTHESIS.md` — The convergence of 6 independent research directions
- `Architecture.md` — Formal mathematical reference for the Vega architecture
- `CLMWithArch.md` — How causal LM maps to geometric transport
- `papers/` — 44 reference papers
