# V12: Spectral Sparsity on the Cotangent Bundle — Tokens as Fourier Wells

**Author:** David Ledbetter
**Date:** March 18, 2026
**Status:** Theoretical framework and implementation plan

---

## Abstract

Every version from v1 through v11 defined sparsity in the *spatial* fiber: a token is a sparse pattern of active dimensions in $\mathcal{F}_q$. V12 identifies the fundamental error in this choice and corrects it. **The natural domain of sparsity for this architecture is not the spatial fiber but its Fourier dual — the cotangent fiber.** A token is a sparse collection of excited frequency modes whose superposition, via the inverse Fourier transform, generates the full spatial field. This shift — from spatial sparsity to spectral sparsity — is not cosmetic. It resolves the locality problem (activation, routing, and training become structurally local), dissolves the contextual manifold problem (v10), unifies the forward–reverse diffusion loop as a single Fourier duality, explains why the Anthropic paper's counting manifolds exhibit "ringing" (they are truncated Fourier series), and provides a compressed-sensing guarantee on information-theoretic completeness. Most importantly, it gives us direct, independent control over both the sparse representation (a few spectral dots) and its global effect on the sequence (the spatial waveform those dots generate).

---

## Part I: The Gap in V11

### 1.1 What V11 Got Right

V11 made the critical correction: a token is not a dense vector but an *event* — many sparse activations firing simultaneously across decoupled subbundles. The diffusion is not smoothing but field reconstruction from sparse sources. The Langevin settling is initialized from the diffused field, not the raw input. Context warps the metric. Proximal sparsity fires at every step. These are non-negotiable and carry forward into v12.

### 1.2 What V11 Got Wrong

V11 defined sparsity in the wrong domain.

When v11 says "subbundle $k$ activates dims $\{3, 17, 42, \ldots\}$," it is selecting a sparse support in the *spatial* fiber $\mathcal{F}_q^{(k)} \cong \mathbb{R}^{d_k}$. The inactive dimensions "carry no information and seed no field." The diffusion then reconstructs the full field from these spatial point sources — solving the heat equation to fill in the unknowns.

The problem is the **Heisenberg uncertainty principle** (or its discrete analog, the Donoho–Stark uncertainty principle):

$$|\text{supp}(x)| \cdot |\text{supp}(\hat{x})| \geq d$$

where $\text{supp}(x)$ is the number of nonzero spatial components and $\text{supp}(\hat{x})$ is the number of nonzero spectral components. A signal that is sparse in space ($|\text{supp}(x)| \ll d$) must be *spread* in the spectral domain ($|\text{supp}(\hat{x})| \approx d$). This means:

1. **The transport operator is inefficient.** The spectral transport kernel $\exp(-D\omega^2 - i\omega \int A)$ must act on *all* frequency bins, because a spatially sparse signal occupies them all. There is no spectral locality to exploit.

2. **The heat kernel fights the representation.** The purpose of the heat kernel $\exp(-D\omega^2)$ is to dampen high frequencies — but a spatially sparse signal has maximal high-frequency content (delta functions have flat spectra). The heat kernel is forced to destroy the representation's defining structure in order to propagate it.

3. **The multi-scale hierarchy is inverted.** In v11, "low $\omega$ = long-range, high $\omega$ = local." But a spatially sparse source has *equal energy at all frequencies*. There is no natural scale separation. The heat kernel must impose it by brute-force dampening, losing information.

The correct choice is the Fourier dual: **sparse in frequency, extended in space**. A spectrally sparse signal — a few excited frequency modes — is:

- Naturally global in spatial extent (each mode $e^{i\omega_k t}$ extends across the entire sequence)
- Naturally multi-scale (each mode has a definite wavelength)
- Naturally compatible with the transport operator (only the active frequency bins participate)
- Naturally amenable to the heat kernel (selective dampening of specific modes, not blanket high-frequency destruction)

### 1.3 The Duality Table

| Property | Spatial Sparsity (v11) | Spectral Sparsity (v12) |
|---|---|---|
| Token representation | Few active spatial dims | Few active frequency modes |
| Spatial extent | Localized (point sources) | Global (each mode spans all positions) |
| Spectral extent | Spread (flat spectrum) | Localized (few occupied bins) |
| Transport efficiency | $O(d \log d)$ — all bins active | $O(s \log d)$ — only $s \ll d$ bins active |
| Heat kernel effect | Destroys structure | Selectively modulates structure |
| Multi-scale hierarchy | Must be imposed externally | Intrinsic (each mode has a wavelength) |
| Training locality | Gradients hit all spatial dims | Gradients hit only active spectral modes |
| Routing locality | Requires spatial proximity or attention | Spectral overlap determines interaction |

---

## Part II: The Spectral Duality — Why Sparsity Belongs in Fourier Space

### 2.1 The Fundamental Observation

Consider a persistent contextual feature — "this passage is written by Romeo" — that modulates the prediction at every subsequent position. In spatial (sequence) space, this feature is a *horizontal line*: a constant or slowly varying signal that extends across many tokens. Routing this signal spatially requires either:

- $O(N^2)$ pairwise attention (every token queries every other)
- $O(N)$ sequential accumulation (GRU/LSTM, v3's ContextGate)
- $O(N \log N)$ convolution with a long kernel (v4's causal conv)

All of these are global operations in sequence space.

Now apply the Fourier transform. A constant signal (horizontal line) maps to **a single delta function at $\omega = 0$**: one dot on the frequency axis. A slowly varying trend maps to a narrow peak near $\omega = 0$. A rapid syntactic oscillation (e.g., noun-verb-noun alternation) maps to a peak at a specific higher frequency.

The persistent, globally extended spatial signal is a *localized point* in spectral space. And a localized point is exactly what we mean by "sparse."

### 2.2 The Parseval Bridge

Parseval's theorem guarantees energy conservation under the Fourier transform:

$$\sum_{t=0}^{T-1} |x(t)|^2 = \frac{1}{T} \sum_{\omega=0}^{T-1} |\hat{x}(\omega)|^2$$

The total "energy" (information content) of a token is identical in both domains. But the *distribution* of that energy is radically different:

- **Spatial sparsity**: energy concentrated at a few positions → spread across all frequencies
- **Spectral sparsity**: energy concentrated at a few frequencies → spread across all positions

For language modeling, we *want* tokens to influence many positions (long-range dependencies) but we *want* the representation to be sparse (for efficient computation and clean attractor dynamics). Spectral sparsity gives both simultaneously. Spatial sparsity gives neither — it localizes influence AND spreads the spectrum.

### 2.3 The Compressed Sensing Guarantee

The Candès–Romberg–Tao theorem (2006) establishes that a signal with $s$-sparse spectral support can be perfectly recovered from $O(s \log d)$ spatial measurements, provided the measurement basis is sufficiently incoherent with the spectral basis. The Fourier basis and the standard (spatial) basis are *maximally incoherent*:

$$\mu(\mathcal{F}, I) = \frac{1}{\sqrt{d}}$$

This means:

1. A token with $s$ active frequency modes can be *perfectly reconstructed* from $O(s \log d)$ spatial samples — far fewer than the full spatial dimension $d$.
2. The reconstruction is *stable* under noise (the LASSO / basis pursuit solution satisfies bounded error guarantees).
3. The sparse spectral representation is *information-theoretically complete* — no information is lost by representing the token as $s$ spectral coefficients rather than $d$ spatial amplitudes, provided $s \log d \lesssim d$.

This validates v12's core claim: you sacrifice nothing by moving to spectral sparsity, and you gain structural locality.

### 2.4 The Architectural Fork: Spatial-Native vs. Spectral-Native Tokens

The move to Fourier space presents a fundamental design decision that must be resolved before implementation. There are two distinct paradigms:

**Paradigm A — The Token as a Spatial Pattern (Spectral Propagation).**
Retain v11's definition: the token's ground truth is a sparse pattern in the spatial fiber. The Fourier domain is used *instrumentally* — as a propagation medium. The pipeline is: spatial sparse source → FFT → spectral transport → IFFT → spatial field → Langevin settling → spatial sparse output.

- *Pro:* Biologically intuitive. Resembles local point-source spikes (neurons firing at specific locations). Easy to conceptualize discrete sequential events. Direct correspondence to v11.
- *Con:* Requires a full FFT/IFFT round-trip at every layer. The Heisenberg penalty (§1.2) applies: spatially sparse signals spread across all frequency bins, making the transport inefficient. The representation fights the transport operator.

**Paradigm B — The Token as a Spectral Pattern (The Fourier Well).**
The token's fundamental identity lives in the frequency domain. A token is an assortment of active "dots" (delta functions) or potential wells in Fourier space. The spatial field is not the representation — it is the *consequence*.

- *Pro:* Native compatibility with the transport kernel ($O(s)$ instead of $O(d \log d)$). Massive global reach: a single combination of dots in Fourier space inherently creates a wide-ranging spatial pattern. Local activation, local routing, local training.
- *Con:* Position and sequential order must be encoded as phase relationships in the spectral domain. High-frequency phase management can become noisy — small phase errors at high $\omega$ create large spatial displacements.

**The resolution: Paradigm B, with spatial settling.**

The Heisenberg argument (§1.2) decisively favors Paradigm B for the *native representation*. But the Langevin settling — the reverse process that collapses the "cloud" to a definite token — must operate in the *spatial* domain. The cloud of possible meanings is a spatial superposition; the energy landscape that guides attractor descent is spatial geometry. The spectral proximal operator (§9.2) provides the bridge: after each spatial Langevin step, project back to spectral sparsity via FFT → top-$s_k$ → IFFT.

This is not a compromise between Paradigms A and B. It is Paradigm B with a clear division of labor:
- **Spectral domain**: representation, transport, storage, inter-block communication
- **Spatial domain**: field reconstruction (IFFT), energy landscape navigation (Langevin), attractor matching (Hopfield)

The spatial domain is *emergent*, not fundamental. The "spatial reality" of the token — its waveform across the sequence — is a consequence of the spectral configuration, computed on-demand via IFFT when the Langevin settling requires it. The model does not "predict the next token in space"; it balances spectral energy so that the spatial interference pattern collapses cleanly into the next concept.

### 2.5 The Feature-Geometry Duality as Domain Selection

The Paradigm A/B fork is a special case of a deeper principle. Any unique activation pattern in the spatial domain maps to a unique spectral pattern in Fourier space, and vice versa:

$$\underbrace{x(t)}_{\text{Spatial Pattern}} \xleftrightarrow{\mathcal{F}} \underbrace{X(\omega)}_{\text{Spectral Pattern}}$$

Not all features are equally sparse in both domains. V12 exploits this asymmetry through **domain-adaptive computation**:

- **Broad, thematic context** (persistent across many tokens) is spectrally sparse (low-frequency wells) but spatially dense (extends everywhere). Manage it in the **spectral domain** — it's a few dots there.
- **Transient, syntactic features** (local collocations, morphological patterns) are spatially sparse (active at a few positions) but spectrally dense (sharp spatial features have broad spectra). During Langevin settling, these features emerge naturally from the **spatial** energy landscape as the proximal operator negotiates which spectral modes survive.

The architecture does not commit exclusively to one domain. It *lives* in spectral space (for efficiency and global reach) but *visits* spatial space (for geometric settling and attractor dynamics). The Fourier transform is not a preprocessing step or a propagation trick — it is the architectural heartbeat, the systole-diastole rhythm between sparse spectral identity and dense spatial influence.

### 2.6 The Phase Management Problem

Paradigm B's primary risk is phase noise. In the spectral representation, sequential position is encoded through phase:

$$x(t) = \sum_j c_j \, e^{i\omega_j t}$$

The position $t$ enters only through the phase factor $e^{i\omega_j t}$. For high-frequency modes ($\omega_j$ large), a small error in $t$ produces a large phase error $\Delta\phi = \omega_j \Delta t$. This means:

- **Low-frequency modes are phase-robust**: $\omega \approx 0$ → $\Delta\phi \approx 0$. Topic, speaker identity, genre — these are position-insensitive and spectrally stable.
- **High-frequency modes are phase-sensitive**: $\omega$ large → $\Delta\phi$ large. Syntax, local agreement, adjacent-token patterns — these require precise positional phase alignment.

This is not a bug — it is the uncertainty principle operating at the architectural level. The same duality that makes low frequencies persistent and high frequencies transient also makes low frequencies robust and high frequencies fragile. The architecture must manage this:

1. **Sinusoidal position encoding as initial phase**: the position embedding $e^{i\omega_j t}$ is added as a phase offset to the spectral section, seeding the correct positional alignment.
2. **The Wilson line accumulates phase corrections**: the gauge connection $A$ adjusts phases based on context, correcting for positional drift.
3. **The Langevin settling resolves ambiguity**: if high-frequency phases are noisy, the spatial energy landscape (Hopfield attractors) acts as a denoiser — the attractor basin absorbs small phase errors during descent.
4. **The spectral proximal operator stabilizes**: by keeping only the top-$s_k$ modes, spectral noise in weakly activated high-frequency bins is eliminated outright.

---

## Part III: Tokens as Spectral Configurations

### 3.1 The Cotangent Bundle

In differential geometry, the *cotangent bundle* $T^*\mathcal{M}$ is the natural home for frequency/momentum representations. At each point $q \in \mathcal{M}$, the cotangent space $T_q^*\mathcal{M}$ contains the "co-vectors" — linear functionals on the tangent space that generalize the notion of frequency.

V11 defined tokens as sparse sections of the fiber bundle $E = \coprod_q \mathcal{F}_q$. V12 redefines tokens as sparse sections of the **spectral fiber bundle** $\hat{E} = \coprod_q \hat{\mathcal{F}}_q$, where $\hat{\mathcal{F}}_q$ is the Fourier dual of $\mathcal{F}_q$:

$$\hat{\mathcal{F}}_q = \bigoplus_{k=1}^{K} \hat{\mathcal{F}}_q^{(k)}$$

Within each subbundle $\hat{\mathcal{F}}_q^{(k)}$, the token excites a sparse set of frequency modes. The subbundle decomposition is preserved — different feature channels (syntax, semantics, etc.) occupy different spectral subbundles and cannot interfere.

### 3.2 The Token as a Sparse Spectral Section

A token at position $t$ in context $q$ is a sparse section of the spectral fiber:

$$\hat{x}_q = \bigoplus_{k=1}^{K} \hat{x}_q^{(k)}, \qquad \hat{x}_q^{(k)} = \sum_{j \in S_q^{(k)}} c_j^{(k)} \, \delta(\omega - \omega_j^{(k)})$$

where:
- $S_q^{(k)} \subset \{1, \ldots, d_k\}$ is the **spectral support** — the set of active frequency bins in subbundle $k$
- $c_j^{(k)} \in \mathbb{C}$ is the **complex amplitude** (magnitude and phase) at frequency $\omega_j^{(k)}$
- $|S_q^{(k)}| = s_k \ll d_k$ — the spectral representation is sparse

The spatial field generated by this token — its "footprint" on the manifold — is the inverse DFT:

$$x_q^{(k)}(n) = \sum_{j \in S_q^{(k)}} c_j^{(k)} \, e^{2\pi i \omega_j^{(k)} n / d_k}$$

This is a dense, globally extended waveform composed of a small number of pure tones. The token's identity is not which spatial dimensions are active, but **which frequencies are excited and with what amplitudes and phases**.

### 3.3 The Token Embedding Table

In v1–v11, the embedding table was $W_{\text{emb}} \in \mathbb{R}^{V \times d}$ — each vocabulary item maps to a dense spatial vector, subsequently sparsified by top-$k$ selection.

In v12, the embedding table stores **spectral configurations**:

$$W_{\text{emb}}: V \to \{(S^{(k)}, c^{(k)})_{k=1}^K\}$$

Each vocabulary item maps to a set of $K$ sparse spectral supports with complex amplitudes. These are the "wells" — the specific frequency modes that define each token's spectral fingerprint. Training updates the positions (which frequencies), depths (amplitudes), and phases of these wells.

Concretely, this can be parameterized as:

$$W_{\text{emb}} \in \mathbb{C}^{V \times d}, \qquad \text{with top-}s_k\text{ sparsification per subbundle applied in Fourier space}$$

The embedding is a complex-valued matrix where sparsification happens *after* interpreting each row as a spectral vector partitioned into $K$ subbundles.

### 3.4 The Combinatorial Capacity

The number of distinct spectral fingerprints is enormous. With $K = 8$ subbundles, each selecting $s_k = 4$ active frequencies from $d_k = 64$ spectral bins:

$$\binom{64}{4}^8 \approx (635{,}376)^8 \approx 1.7 \times 10^{46}$$

But the expressiveness is even richer than v11's spatial combinatorics, because each active frequency carries a *complex* amplitude $(|c_j|, \arg c_j)$ — continuous magnitude and phase. Two tokens can share the same spectral support but differ in amplitude or phase, producing distinct spatial waveforms. The spectral representation is simultaneously highly sparse and extraordinarily expressive.

---

## Part IV: The Locality Principle

### 4.1 Local Activation

When a token enters the system, it excites specific frequency wells in each spectral subbundle. This is a *local* operation in spectral space — a few delta functions are placed at specific frequencies. No global computation over the spatial manifold is required to represent the token.

The word "king" might excite:
- Subbundle 1 (semantic): $\omega_{\text{royalty}}$ (low frequency, persistent theme) + $\omega_{\text{authority}}$
- Subbundle 2 (syntactic): $\omega_{\text{noun}}$ (mid frequency, grammatical role)
- Subbundle 3 (phonetic): $\omega_{\text{velar-stop}}$ (high frequency, transient articulation)

Each activation is a point in spectral space. The totality of points across all subbundles IS the token.

### 4.2 Local Routing

In spatial space, routing requires determining which distant tokens are relevant — a global, $O(N^2)$ operation. In spectral space, two tokens interact when their **frequency content overlaps**.

If token A excites $\{\omega_1, \omega_5, \omega_{12}\}$ in subbundle $k$ and token B excites $\{\omega_3, \omega_5, \omega_{14}\}$, their interaction strength in subbundle $k$ is determined by their spectral overlap (here, the shared mode $\omega_5$). Tokens with no spectral overlap in a subbundle do not interact in that channel — period. No routing computation is needed to discover this; it is a structural property of the spectral representation.

This is the content-dependent selectivity that v3's GRU achieved and v4–v8 could not: tokens interact based on *what they are* (which frequencies they excite), not *where they are* (position). Spectral overlap is an intrinsic, content-dependent routing mechanism that requires no explicit attention computation.

More precisely, the interaction energy between two spectral sections $\hat{x}$ and $\hat{y}$ in subbundle $k$ is:

$$I^{(k)}(\hat{x}, \hat{y}) = \sum_{\omega} \hat{x}^{(k)}(\omega)^* \, \hat{y}^{(k)}(\omega) = \langle \hat{x}^{(k)}, \hat{y}^{(k)} \rangle$$

By Parseval's theorem, this equals the spatial inner product $\langle x^{(k)}, y^{(k)} \rangle$. But in the spectral domain, the sum has only $|S_x^{(k)} \cap S_y^{(k)}|$ nonzero terms — the computation is local and sparse.

### 4.3 Local Training

When the loss function $\mathcal{L}$ depends on the spatial field $x(n)$, and $x(n) = \sum_{j \in S} c_j e^{2\pi i \omega_j n / d}$, the gradient with respect to the spectral coefficient $c_j$ is:

$$\frac{\partial \mathcal{L}}{\partial c_j} = \sum_{n} \frac{\partial \mathcal{L}}{\partial x(n)} \cdot e^{2\pi i \omega_j n / d}$$

This is the **Fourier coefficient of the spatial gradient at frequency $\omega_j$**. The update to $c_j$ depends only on the projection of the spatial loss gradient onto the $j$-th frequency mode. If the spatial gradient is band-limited (as it tends to be for smooth loss landscapes — and language generation losses over sequences are empirically smooth), then only a few $c_j$ receive significant updates.

The gradient with respect to the frequency position $\omega_j$ (training which frequency the well sits at):

$$\frac{\partial \mathcal{L}}{\partial \omega_j} = \frac{2\pi i}{d} \sum_{n} n \cdot \frac{\partial \mathcal{L}}{\partial x(n)} \cdot c_j \, e^{2\pi i \omega_j n / d}$$

This is a modulated Fourier coefficient — the spatial gradient weighted by position. Again, structurally local: the update to the $j$-th well depends on how the spatial gradient projects onto the $j$-th mode's position-modulated waveform.

**Training is structurally confined to the active spectral wells.** Inactive frequencies receive zero gradient by construction (they have $c_j = 0$ and are not in the support). The gradient sparsity mirrors the representation sparsity.

---

## Part V: The Forward–Reverse Loop Reinterpreted

### 5.1 The V11 Loop

V11 established the non-negotiable forward–reverse diffusion cycle:

```
Sparse sources → Field reconstruction (diffusion) → Dense cloud → Langevin settling → Sparse output
```

### 5.2 The V12 Reinterpretation

In v12, this loop is revealed as the **Fourier duality** operating at the architectural level:

```
Sparse spectral config → IFFT → Dense spatial field → Spectral Langevin → Sparse spectral config
     (few active ω)       (field reconstruction)    (full manifold)     (attractor descent)     (few active ω)
```

The "diffusion as field reconstruction" from v11 is literally the inverse Fourier transform: sparse spectral sources → dense spatial field. The "dense cloud" (v11, CLMWithArch.md) is the spatial waveform generated by the superposition of a few frequency modes. The Langevin settling — collapsing the cloud to a definite sparse state — is the projection back to the nearest sparse spectral attractor.

The forward–reverse loop is not a diffusion equation followed by a separate energy descent. It is a single round trip through the Fourier duality:

$$\text{Spectral (sparse)} \xrightarrow{\mathcal{F}^{-1}} \text{Spatial (dense)} \xrightarrow{\text{Langevin} \to \text{nearest spectral attractor}} \text{Spectral (sparse)}$$

### 5.3 Why This Unification Matters

In v11, the forward (diffusion) and reverse (Langevin) processes were conceptually separate operations that happened to be chained. The diffusion "solved the heat equation"; the Langevin "descended the Hopfield energy." Their connection was the initialization: Langevin starts from the diffused field.

In v12, they are two halves of a single mathematical operation: the Fourier–Langevin round trip. The diffusion IS the spectral-to-spatial map. The settling IS the spatial-to-spectral map. The heat kernel $\exp(-D\omega^2)$ is not a separate "diffusion step" — it is the frequency-dependent modulation that naturally occurs when you propagate spectral sources through a medium with finite bandwidth. The gauge connection $\exp(-i\omega \int A)$ is the frequency-dependent phase shift that occurs when spectral sources propagate through a curved geometry.

Both the heat kernel and the gauge connection act *directly on the spectral representation* — no FFT/IFFT is needed for the transport step, because the tokens are already in the spectral domain. The full pipeline becomes:

```
1. Token → Sparse spectral section  (embedding)
2. Apply transport kernel:  X̃(ω) = X(ω) · exp(-D(ctx)ω² - iω·A(ctx))  (spectral domain, direct multiply on active modes only)
3. IFFT → Dense spatial field  (field reconstruction / "the cloud")
4. Langevin settling on spatial field with spectral memory atoms  (reverse: cloud → sparse spectral attractor)
5. Output → Sparse spectral section  (feeds into next block)
```

Step 2 is $O(s)$ per subbundle (only the active modes participate). The FFT in v11 was $O(d \log d)$ — wasted work, because most frequency bins were occupied by a spatially sparse signal. In v12, only the nonzero spectral bins are modulated.

### 5.4 The Spectral Hopfield Network

The memory bank atoms $M_q$ should themselves be spectral configurations. Each atom $\hat{m}_j \in \hat{\mathcal{F}}_q$ is a sparse spectral pattern — a "valid word" in the spectral vocabulary. The Hopfield energy in spectral space is:

$$E_q(\hat{x}; \hat{M}_q) = -\beta^{-1} \log \sum_j \exp\left(\beta \, \text{Re}\langle \hat{x}, \hat{m}_j \rangle\right)$$

where $\langle \hat{x}, \hat{m}_j \rangle = \sum_\omega \hat{x}(\omega)^* \hat{m}_j(\omega)$ is the spectral inner product. By Parseval, this equals the spatial inner product — the Hopfield energy is identical in both domains. But the spectral computation exploits sparsity: the inner product $\langle \hat{x}, \hat{m}_j \rangle$ has only $|S_x \cap S_j|$ nonzero terms.

The Langevin descent can happen in either domain. In practice, the settling should operate on the dense spatial field (after IFFT) using memory atoms whose spatial patterns are precomputed from their spectral configurations. This is because the settling process explores the *spatial* energy landscape — the "cloud" of possible meanings at a position — and the proximal operator enforces sparsity by projecting back to spectral space:

$$\hat{x}_{t-\Delta t}^{\text{sparse}} = \text{top-}s_k\left(\mathcal{F}(\text{Langevin-step}(x_t))\right) \quad \text{per subbundle}$$

The proximal re-sparsification at each Langevin step is no longer soft-thresholding in spatial dimensions — it is **projection to spectral sparsity**: FFT the current state, keep only the $s_k$ largest-magnitude frequency bins per subbundle, IFFT back. This is a hard spectral sparsity constraint that directly enforces the token-as-spectral-configuration structure.

---

## Part VI: The Potential Landscape — Wells, Modulation, and Multi-Scale Structure

### 6.1 Tokens as Potential Wells in Frequency Space

The Hamiltonian formalism provides the natural physical picture. Define the spectral potential energy:

$$V(\omega) = -\sum_{j \in S} |c_j|^2 \, \delta(\omega - \omega_j)$$

Each active frequency mode is a potential well at frequency $\omega_j$ with depth $|c_j|^2$. The token's spectral fingerprint is a 1D potential landscape — a collection of wells (dips) in the frequency domain, with the rest of the spectrum at zero potential.

Training deepens, shallows, and repositions these wells:
- **Deepening a well** ($|c_j| \uparrow$): strengthens a frequency mode → increases its spatial influence
- **Shallowing a well** ($|c_j| \downarrow$): weakens a mode → eventual pruning if it reaches zero
- **Shifting a well** ($\omega_j \to \omega_j'$): changes the spatial wavelength → changes the scale of influence
- **Creating a new well**: a previously inactive frequency becomes active → new spatial feature
- **Destroying a well**: an active frequency goes to zero → loss of a spatial feature

### 6.2 Modulation and Beat Frequencies

When two nearby frequency wells $\omega_1$ and $\omega_2 = \omega_1 + \Delta\omega$ are simultaneously active, their spatial superposition exhibits **amplitude modulation** (beating):

$$c_1 e^{i\omega_1 t} + c_2 e^{i\omega_2 t} = e^{i\bar{\omega} t}\left(c_1 e^{-i\frac{\Delta\omega}{2} t} + c_2 e^{i\frac{\Delta\omega}{2} t}\right)$$

where $\bar{\omega} = (\omega_1 + \omega_2)/2$. The fast carrier wave at $\bar{\omega}$ is modulated by a slow envelope at $\Delta\omega / 2$.

In language terms: a cluster of nearby low-frequency wells creates a modulation pattern that varies on a scale longer than any individual mode — **the beating IS the paragraph-scale semantic structure**. The individual wells are too slow-varying to capture paragraph transitions, but their interference pattern captures the modulation between persistent themes.

This is a richer representation than any single-scale approach. The multi-scale structure emerges from the *interference* between spectral wells, without any explicit multi-scale design.

### 6.3 The Natural Multi-Scale Hierarchy

The time-frequency uncertainty principle ($\Delta\omega \cdot \Delta t \geq \frac{1}{2}$) creates an intrinsic hierarchy:

| Frequency Range | Spatial Extent | Linguistic Scale | Persistence |
|---|---|---|---|
| $\omega \approx 0$ (DC) | Entire sequence | Document topic, genre, speaker identity | Permanent |
| Low $\omega$ | Many paragraphs | Thematic arcs, narrative structure | Very persistent |
| Mid $\omega$ | Sentence-scale | Semantic roles, argument structure | Moderately persistent |
| High $\omega$ | 2–3 tokens | Syntactic agreement, local collocations | Transient |
| Very high $\omega$ | Adjacent tokens | Character-level patterns, morphology | Momentary |

This hierarchy requires no architectural design — it falls out of the physics. The heat kernel $\exp(-D\omega^2)$ naturally preserves low frequencies (long-range) and dampens high frequencies (local), creating the multi-scale structure that v11 had to impose externally.

### 6.4 Context as Frequency Modulation

When context changes (e.g., a topic shift), the underlying metric of the spectral manifold warps. Concretely, energy migrates between frequency wells:

$$c_j(t+1) = c_j(t) + \eta \, \frac{\partial \mathcal{L}}{\partial c_j}$$

A topic shift from "war" to "peace" manifests as:
- The $\omega_{\text{conflict}}$ well shallows (amplitude decreases)
- The $\omega_{\text{harmony}}$ well deepens (amplitude increases)
- The spatial field smoothly transitions as the waveform composition changes

This is **frequency modulation (FM)** — the spectral configuration of the signal changes over time, modulating the spatial field. The v10 "contextual manifold" problem (how to make $q_t$ context-dependent) is dissolved: the context IS the current spectral configuration. Different histories produce different spectral wells, which produce different spatial fields, which produce different Hopfield landscapes. No explicit context accumulator is needed — the spectral state itself carries the accumulated context through its frequency content.

More precisely, the Wilson line holonomy in spectral space is:

$$\hat{U}_\gamma(\omega) = \exp\left(-i\omega \int_\gamma A\right)$$

This is a frequency-dependent phase rotation applied to the spectral section as it transports along the sequence. Different frequencies rotate by different amounts — low frequencies rotate slowly (persistent context), high frequencies rotate rapidly (transient adjustments). The accumulated phase at each frequency encodes the full path history, frequency by frequency. This IS the KV cache analog, but compressed into $s$ complex numbers (the amplitudes and accumulated phases at the active frequencies) rather than $T \times d$ stored vectors.

---

## Part VII: Connection to External Theory

### 7.1 The Anthropic Paper: Ringing as Truncated Fourier Series

The most striking connection to Gurnee et al. (2026) is the **ringing phenomenon**. The authors observe that the character count manifold, when projected into a 6-dimensional subspace, exhibits "ripples" — off-diagonal oscillations in the cosine similarity matrix. They explicitly note (§2.5, Appendix):

> *"A relationship of this construction to Fourier features is discussed in the appendix."*

and their optimal embedding analysis shows that the ringing is a **Gibbs phenomenon** — the oscillation that results from truncating a Fourier series representation of a narrow-peaked similarity function to $k$ terms. The 6-dimensional counting manifold IS a 6-term Fourier approximation.

This is not a metaphor. The character count representation is literally:

$$q_{\text{count}} \approx \sum_{j=1}^{6} a_j \, e^{2\pi i f_j \cdot \text{count} / N}$$

The 10 discrete "place cell" features that tile this manifold are the atoms of a sparse dictionary that locally parameterizes the truncated Fourier curve. The ringing between features (features at distance $\Delta$ having negative cosine similarity) is the spectral leakage from truncation.

**V12 makes this explicit by design.** Instead of hoping that the model learns Fourier structure (as Claude 3.5 Haiku did emergently for counting), we define tokens as sparse Fourier configurations from the start. The counting manifold's topology — helix in 6D, tiling by place cells, ringing from truncation — is the *natural consequence* of representing a 1D variable (count) as a sparse spectral section.

### 7.2 The Anthropic Paper: QK Twist as Spectral Phase Alignment

The paper's central finding is that attention heads "twist" (rotate) one manifold to align it with another. The QK matrix physically rotates the character count manifold so that count $i$ aligns with line width $k = i + \epsilon$.

In spectral terms, this rotation is a **frequency-dependent phase shift** — exactly the gauge connection:

$$\hat{x}_{\text{aligned}}(\omega) = \hat{x}_{\text{original}}(\omega) \cdot e^{-i\omega \theta}$$

where $\theta$ is the "twist angle" (the offset). This shifts the manifold along the 1D parameter (count) by an amount $\theta$, aligning count $i$ with line width $i + \theta$. The operation is a single complex multiply per active frequency bin — $O(s)$ computation.

The multiple boundary heads with different offsets (Figure 16 of the paper) are multiple gauge connections with different phase shifts, creating a "stereoscopic" system that tiles the boundary detection space. Each head applies a different $\theta_h$, and their combined output resolves the full distance-to-boundary function.

V12 implements this natively: the gauge connection $A_{t \to t+1}$ produces per-frequency phase shifts, and different subbundles can have different connections (different "twist angles"), providing the multi-head stereoscopic structure without explicit multi-head attention.

### 7.3 The Gauge Fiber Bundle Paper: Curvature and Path-Dependence

The anonymous ICLR submission (Paper 19168) proves that attention induces an Ehresmann connection with generically nonzero curvature (Theorem 4.1). The curvature tensor $\Omega(u, v)$ measures how transport around a small rectangle produces a nontrivial gauge displacement — i.e., the result of transporting a representation depends on the *path* through the sequence.

In v12, the curvature has a precise spectral interpretation. The curvature at frequency $\omega$ is:

$$\Omega_\omega(u, v) = \frac{\partial A_\omega}{\partial u} \cdot v - \frac{\partial A_\omega}{\partial v} \cdot u + [A_\omega(u), A_\omega(v)]$$

In the Abelian ($U(1)$) case, the commutator vanishes and the curvature reduces to:

$$\Omega_\omega = dA_\omega = \frac{\partial A_\omega}{\partial u} \wedge dv - \frac{\partial A_\omega}{\partial v} \wedge du$$

Nonzero curvature means that the accumulated phase at frequency $\omega$ depends on the path through the sequence — different orderings of the same tokens produce different spectral configurations at the endpoint. This IS context sensitivity, manifested frequency by frequency. Low-frequency modes accumulate phase slowly (robust to reordering — topic is order-insensitive). High-frequency modes accumulate phase rapidly (sensitive to local ordering — syntax is order-critical).

### 7.4 Quantum Field Theory: Tokens as Quantized Excitations

The v12 picture has a direct analog in quantum field theory (QFT). In QFT:

- The **vacuum** is the zero-excitation state of a field defined over spacetime
- A **particle** is a quantized excitation — a localized peak in the frequency decomposition of the field, at a specific energy/momentum
- **Interactions** between particles occur through their field overlap (propagators in momentum space)
- The **Fock space** representation decomposes the full field state into a sum over particle-number sectors

In v12:
- The **vacuum** is the zero-excitation state of the spectral fiber (no active frequency modes)
- A **token** is a set of quantized excitations — sparse peaks in the spectral decomposition, each at a specific frequency with a specific amplitude
- **Interactions** between tokens occur through spectral overlap (inner product in frequency space)
- The **token vocabulary** is the set of allowed excitation patterns — the "particle spectrum" of the architecture

This is not merely analogical. The mathematics are identical: the spectral fiber bundle $\hat{E}$ is a bosonic Fock space over the single-mode Hilbert spaces $\hat{\mathcal{F}}_q^{(k)}$. The token embedding table assigns each vocabulary item a specific multi-mode excitation pattern. The transport operator propagates excitations through curved spectral space. The Langevin settling is the "measurement" that collapses the continuous field state to a definite particle configuration.

### 7.5 Compressed Sensing and Restricted Isometry

The Restricted Isometry Property (RIP) of Candès and Tao provides a deeper guarantee. If the IFFT matrix $\mathcal{F}^{-1}$ satisfies the RIP of order $2s$ with constant $\delta_{2s} < \sqrt{2} - 1$, then the basis pursuit denoising estimator:

$$\min_{\hat{x}} \|\hat{x}\|_1 \quad \text{s.t.} \quad \|\mathcal{F}^{-1}\hat{x} - y\|_2 \leq \epsilon$$

recovers any $s$-sparse spectral vector $\hat{x}$ from noisy spatial observations $y = \mathcal{F}^{-1}\hat{x} + \text{noise}$ with bounded error.

The Langevin settling with proximal sparsity (soft-thresholding at each step) is a dynamical system that approximately solves exactly this optimization problem. The proximal operator is the proximal map of the $\ell_1$ norm. The Hopfield gradient provides the data-fidelity term (pushing toward the nearest memory atom in the spatial domain). The combination — proximal gradient descent on a composite objective — is the ISTA (Iterative Shrinkage-Thresholding Algorithm) of Daubechies et al. (2004), here embedded inside the Langevin SDE as an annealed variant.

V12's Langevin-with-spectral-sparsity is therefore a *stochastic*, *annealed*, *attractor-regularized* variant of compressed sensing recovery. The mathematical guarantees of compressed sensing (exact recovery, noise stability, minimum measurements) carry over as structural properties of the architecture.

---

## Part VIII: What V12 Dissolves

### 8.1 The Contextual Manifold Problem (v10)

V10 diagnosed the root cause of the 45% ceiling: the manifold coordinate $q_t$ was positional, not contextual. The solution required a "context accumulator" — GRU, parallel scan, or attention — to build $q_t$ from the sequence history.

V12 dissolves this problem entirely. The spectral configuration IS the context. Each token's spectral state encodes not just its identity but the accumulated influence of all prior tokens, through:

1. **The Wilson phase**: accumulated gauge rotation at each frequency mode
2. **Amplitude modulation**: context-dependent changes in well depth
3. **Frequency migration**: wells shifting position as meaning evolves

No separate context accumulator is needed. The spectral state naturally carries history through its phase structure, just as a physical wave carries information about its source and propagation medium through its phase and amplitude profile.

### 8.2 The Routing Problem (v3–v9)

The fundamental tension was content-dependent selectivity (v3's GRU, $O(T)$ sequential) vs. parallelism (v4's FFT, $O(T \log T)$ but content-independent). Every version tried to achieve both and failed.

V12 provides content-dependent routing through **spectral overlap** — a structural property that requires no explicit routing computation. Two tokens interact in proportion to their shared frequency content. This is:
- **Content-dependent** — interaction depends on what the tokens are (their spectral fingerprints)
- **Automatically parallel** — spectral inner products can be computed independently for all pairs
- **Inherently sparse** — only shared frequency bins contribute, so sparse tokens have sparse interactions
- **No routing parameters** — the routing is a consequence of the representation, not a learned mechanism

### 8.3 The Hopfield Dominance Problem (v10 §2.2)

In v1–v9, the Hopfield gradient ($\|\nabla E\| \approx 1.0$) overwhelmed the cross-position routing forces ($\|\text{routing}\| \ll 1.0$). The settling was dominated by a context-blind attractor.

In v12, the memory bank atoms are themselves spectral configurations. The Hopfield gradient pulls toward the nearest spectral attractor — an attractor defined by its frequency content, not by position. Different frequency wells activate different attractors. The dominant settling force is now inherently content-dependent and multi-scale: low-frequency wells pull toward thematic attractors, high-frequency wells pull toward syntactic attractors, independently and simultaneously across subbundles.

### 8.4 The Sequential Bottleneck

V11's sequential bottleneck (autoregressive loop over positions, $T \times N_{\text{blocks}} \times L_{\text{steps}}$) was a practical limitation. V12 partially alleviates this:

The transport step is $O(s)$ per active mode per subbundle (direct multiply on nonzero spectral bins), down from $O(d \log d)$ (full FFT). With $K = 8$ subbundles, $s_k = 4$ active modes per subbundle, and $d_k = 64$: v11 cost = $8 \times 64 \times \log 64 = 3{,}072$ operations. V12 cost = $8 \times 4 = 32$ operations. A **96× reduction** in the transport step.

The IFFT (field reconstruction) and Langevin settling still operate on the full spatial dimension. But the transport — which constitutes one of the three main operations per block — becomes nearly free.

---

## Part IX: Implementation Architecture

### 9.1 The Full Forward Pass

```
Input Token IDs: [t_1, t_2, ..., t_T]
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│          SPECTRAL TOKEN EMBEDDING                         │
│                                                           │
│  token_id ──► Complex spectral vector ĥ ∈ ℂ^d            │
│  Per-subbundle top-s_k sparsification in Fourier domain   │
│  Result: Sparse spectral section x̂_q^(k) for each k      │
│  + Sinusoidal position encoding (additive phase offset)   │
└─────────────────────────┬─────────────────────────────────┘
                          │
         ┌────────────────▼────────────────┐
         │  For each position t:           │
         │  (autoregressive loop)          │
         └────────────────┬────────────────┘
                          │
    ┌─────────────────────▼─────────────────────────────────┐
    │      SPECTRAL TRANSPORT (per block) — O(s) per mode   │
    │                                                       │
    │  For each subbundle k, for each active mode j ∈ S^(k):│
    │                                                       │
    │    ĉ_j ──► ĉ_j · exp(-D_k(ctx)·ω_j² - iω_j·A_k(ctx))│
    │                                                       │
    │  D_k(ctx): Context-dependent diffusion per subbundle  │
    │  A_k(ctx): Context-dependent gauge phase per subbundle│
    │                                                       │
    │  Wilson phase update: φ_j += A_k(ctx) · ω_j           │
    │                                                       │
    │  Output: Transported spectral section X̃^(k)           │
    └─────────────────────┬─────────────────────────────────┘
                          │
    ┌─────────────────────▼─────────────────────────────────┐
    │      FIELD RECONSTRUCTION — O(d log d) via IFFT       │
    │                                                       │
    │  Assemble full spectral vector (sparse → dense):      │
    │  X̃(ω) = 0 for ω ∉ ∪_k S^(k),  X̃(ω_j) = ĉ_j        │
    │                                                       │
    │  IFFT: x̃ = F⁻¹(X̃) ∈ ℝ^d                             │
    │                                                       │
    │  x̃ is the dense spatial field — the "cloud"           │
    └─────────────────────┬─────────────────────────────────┘
                          │
    ┌─────────────────────▼─────────────────────────────────┐
    │      SPECTRAL MEMORY BANK                             │
    │                                                       │
    │  Spectral state X̃ ──► Router MLP ──► gating logits    │
    │                                    ──► k-WTA          │
    │                                                       │
    │  Global spectral dictionary D̂ ∈ ℂ^(d × N_global)      │
    │  M̂_q = D̂[:, top-k indices]  (spectral atoms)         │
    │  M_q = F⁻¹(M̂_q)  (precomputed spatial patterns)      │
    └─────────────────────┬─────────────────────────────────┘
                          │
    ┌─────────────────────▼─────────────────────────────────┐
    │      LANGEVIN–HOPFIELD DESCENT (spatial domain)       │
    │                                                       │
    │  Initialize: x_T = x̃  (the dense cloud)               │
    │                                                       │
    │  For step = 1..L (β increasing):                      │
    │    │                                                  │
    │    ├──► ∇E = -softmax(β·xᵀ·M_q)·M_q                  │
    │    │        (Hopfield score, spatial inner products)   │
    │    │                                                  │
    │    ├──► inhibition = -γ · W_inh · x                   │
    │    │        (lateral cortical inhibition)              │
    │    │                                                  │
    │    ├──► noise = √(2η/β_t) · ε                         │
    │    │        (simulated annealing)                      │
    │    │                                                  │
    │    ├──► x = x - η·∇E - inhib + noise                  │
    │    │        (Langevin step in spatial domain)          │
    │    │                                                  │
    │    └──► SPECTRAL PROXIMAL SPARSIFICATION:             │
    │         x̂ = FFT(x)                                    │
    │         Per subbundle k: keep top-s_k magnitudes      │
    │         x = IFFT(x̂_sparse)                            │
    │         (project back to spectrally sparse manifold)   │
    │                                                       │
    │  Output: x_0  →  x̂_0 = FFT(x_0) = sparse spectral   │
    └─────────────────────┬─────────────────────────────────┘
                          │
                          │  x̂_0 feeds into next block as sparse spectral input
                          │
    ┌─────────────────────▼─────────────────────────────────┐
    │      DECODER (after final block)                      │
    │                                                       │
    │  x̂_0 → IFFT → spatial → Linear → SiLU → Linear       │
    │                                        → vocab logits │
    │                                                       │
    │  P(t_{t+1} | t_1, ..., t_t)                           │
    └───────────────────────────────────────────────────────┘
```

### 9.2 The Spectral Proximal Operator: Key Difference from V11

In v11, the proximal operator was soft-thresholding in spatial dimensions:

$$x^{\text{sparse}} = \text{sign}(x) \odot \max(|x| - \lambda\eta, 0)$$

This kills small spatial activations. But we want spectral sparsity, not spatial sparsity. In v12, the proximal operator is **spectral top-$s_k$ projection**:

```python
def spectral_proximal(x, subbundle_sizes, sparsity_per_bundle):
    """Project x onto the manifold of spectrally sparse signals."""
    x_hat = torch.fft.fft(x)  # To spectral domain

    # Per-subbundle spectral sparsification
    offset = 0
    for k, (d_k, s_k) in enumerate(zip(subbundle_sizes, sparsity_per_bundle)):
        sub = x_hat[..., offset:offset+d_k]
        magnitudes = sub.abs()
        threshold = magnitudes.topk(s_k, dim=-1).values[..., -1:]
        mask = (magnitudes >= threshold).float()
        x_hat[..., offset:offset+d_k] = sub * mask
        offset += d_k

    return torch.fft.ifft(x_hat).real  # Back to spatial, now spectrally sparse
```

This is the key architectural innovation: the proximal operator at each Langevin step enforces that the state lives on the manifold of $s$-sparse spectral signals. The settling dynamics explore the spatial energy landscape but are constrained to return to spectral sparsity after each step.

### 9.3 Parameter Summary

| Component | Domain | Parameters | Complexity |
|---|---|---|---|
| Spectral Embedding | $\hat{\mathcal{F}}$ | $V \times d$ (complex) | $O(d)$ |
| Spectral Transport | $\hat{\mathcal{F}}$ | $D_k$, $A_k$ (context-dependent) | $O(s)$ per mode |
| Field Reconstruction | $\hat{\mathcal{F}} \to \mathcal{F}$ | None (IFFT) | $O(d \log d)$ |
| Spectral Memory Bank | $\hat{\mathcal{F}}$ | $N_g \times d$ (complex dict) | $O(k \cdot d)$ |
| Langevin Descent | $\mathcal{F}$ | $W_{\text{inh}} \in \mathbb{R}^{d \times d}$ | $O(L \cdot k \cdot d)$ |
| Spectral Proximal | $\mathcal{F} \to \hat{\mathcal{F}}$ | None (FFT + top-$s_k$) | $O(d \log d)$ per step |
| Decoder | $\mathcal{F}$ | $d \times d + d \times V$ | $O(dV)$ |

---

## Part X: The Mathematical Framework — Complete Formulation

### 10.1 Axiom Revision

V12 revises Axiom 1 (Fiber Bundle Topology) and Axiom 2 (Spectral Transport) while preserving Axioms 3–5.

**Axiom 1′ — Spectral Fiber Bundle Topology.** Computation occurs over a base manifold $\mathcal{M}$. At each contextual coordinate $q \in \mathcal{M}$, the *spectral fiber* $\hat{\mathcal{F}}_q$ provides the representational space. Tokens are sparse sections of the spectral bundle:

$$\hat{x}_q = (S_q, c_q), \qquad S_q = \bigcup_k S_q^{(k)}, \qquad |S_q^{(k)}| = s_k \ll d_k$$

where $c_q \in \mathbb{C}^{|S_q|}$ are complex amplitudes. The spectral fiber decomposes into $K$ orthogonal subbundles:

$$\hat{\mathcal{F}}_q = \bigoplus_{k=1}^{K} \hat{\mathcal{F}}_q^{(k)}$$

**Axiom 2′ — Native Spectral Transport.** Because tokens are already in the spectral domain, the transport acts directly on the spectral coefficients without requiring FFT:

$$\tilde{c}_j^{(k)} = c_j^{(k)} \cdot \exp\left(-D_k(q) \, \omega_j^2 - i\omega_j \int_\gamma A_k(q)\right) \qquad \forall j \in S_q^{(k)}$$

This is $O(s_k)$ per subbundle — a direct multiply on the active modes only.

**Axioms 3–5** (Dynamic Memory Bank, Langevin-Hopfield Descent, Proximal Sparsity) carry forward with the modification that the proximal operator enforces *spectral* sparsity (top-$s_k$ in Fourier domain) rather than spatial sparsity (soft-thresholding).

### 10.2 The Transport-Reconstruction-Settling Pipeline

For token at position $t$ entering block $b$:

**Step 1 — Spectral Transport:**
$$\tilde{c}_j^{(k)} = c_j^{(k)} \cdot K(\omega_j; q_t), \qquad K(\omega; q) = \exp\left(-D_k(q)\omega^2 - i\omega \int_\gamma A_k(q)\right)$$

**Step 2 — Field Reconstruction:**
$$\tilde{x}(n) = \sum_{k=1}^{K} \sum_{j \in S^{(k)}} \tilde{c}_j^{(k)} \, e^{2\pi i \omega_j n / d_k} \qquad (\text{IFFT, or partial sum over active modes})$$

**Step 3 — Memory Routing:**
$$g(q_t) = \text{k-WTA}(W_{\text{route}} [\tilde{x}, q_t]), \qquad \hat{M}_{q_t} = \hat{D}[\text{top-}k\text{ indices}]$$

**Step 4 — Langevin Settling with Spectral Proximal:**

$$x_0 = x_T, \quad x_T = \tilde{x}$$

For $\ell = 1, \ldots, L$:
$$x_\ell' = x_{\ell-1} - \eta \nabla_x E_{q_t}(x_{\ell-1}; M_{q_t}) - \gamma W_{\text{inh}} x_{\ell-1} + \sqrt{2\eta / \beta_\ell} \, \epsilon_\ell$$
$$\hat{x}_\ell = \mathcal{F}(x_\ell')$$
$$\hat{x}_\ell^{\text{sparse}} = \text{top-}s_k(\hat{x}_\ell) \quad \text{per subbundle}$$
$$x_\ell = \mathcal{F}^{-1}(\hat{x}_\ell^{\text{sparse}})$$

**Step 5 — Output:**
$$\hat{x}_{\text{out}} = \mathcal{F}(x_L) \qquad \text{(sparse spectral section for next block)}$$

### 10.3 The Context Encoding

The manifold coordinate $q_t$ in v12 is derived from the spectral state itself:

$$q_t = \text{Re}\left[\sum_{k=1}^{K} \sum_{j \in S_t^{(k)}} c_j^{(k)} \cdot e^{i\phi_j^{(k)}(t)}\right]$$

where $\phi_j^{(k)}(t) = \sum_{\tau=0}^{t} A_k(q_\tau) \cdot \omega_j$ is the accumulated Wilson phase at frequency $\omega_j$ in subbundle $k$. The manifold coordinate is a *summary of the spectral state* — the real part of the accumulated spectral section with all phases included.

This is inherently contextual: different token histories produce different accumulated phases, which produce different $q_t$. No separate context accumulator is needed.

---

## Part XI: Open Questions

### 11.1 Continuous vs. Discrete Frequencies

Should the frequency positions $\omega_j$ be:
- **Fixed** (the standard DFT grid)? Simplest, compatible with FFT hardware acceleration.
- **Learned per token** (continuous frequencies)? More expressive, but requires NDFT (non-uniform DFT) or Gabor atom computation. Training must learn both positions and amplitudes.
- **Learned globally but shared** (a fixed set of "basis frequencies" that all tokens select from)? Middle ground — the dictionary of possible frequencies is shared, but each token selects a sparse subset.

### 11.2 Complex vs. Real Amplitudes

The complex amplitude $c_j = |c_j| e^{i\arg c_j}$ carries both magnitude and phase. Should phase be:
- **Free** (learned independently per token per frequency)? Maximum expressiveness.
- **Position-determined** (phase = position × frequency, as in sinusoidal encoding)? Ties phase to position, reducing parameters.
- **Context-accumulated** (phase = Wilson line holonomy)? Ties phase to history, making context intrinsic to the representation.

### 11.3 The Proximal Operator: Top-$k$ vs. Soft-Thresholding in Spectral Domain

Top-$s_k$ is the direct spectral sparsity constraint. But it has zero gradient for non-selected modes. Alternatives:
- **Spectral soft-thresholding**: $\hat{x}^{\text{sparse}}(\omega) = \text{sign}(\hat{x}(\omega)) \cdot \max(|\hat{x}(\omega)| - \lambda, 0)$. Differentiable but doesn't guarantee exact $s_k$ sparsity.
- **Spectral entmax**: differentiable sparse projection with learnable sharpness.
- **Straight-through estimator**: top-$s_k$ in forward pass, pass gradient through in backward pass.

### 11.4 Should Langevin Settle in Spectral or Spatial Domain?

Three options exist along the Paradigm A–B spectrum (§2.4):

- **Spatial settling** (v12 as described): IFFT to spatial, Hopfield descent in spatial, FFT+sparsify back. The spatial domain provides the geometric energy landscape for attractor descent.
- **Spectral settling**: Define the Hopfield energy directly in spectral space (inner products of spectral atoms). Settling happens without ever leaving the spectral domain. More efficient ($O(s)$ per step) but loses the spatial geometry that makes the Hopfield landscape meaningful.
- **Hybrid settling**: Run the first $L/2$ steps in spatial (coarse attractor basin selection on the full energy landscape) and the last $L/2$ steps in spectral (fine spectral refinement on the active modes). This mirrors the annealing schedule — broad exploration first (spatial, full geometry), then focused sharpening (spectral, sparse modes only).

The default recommendation is spatial settling with spectral proximal, because the "cloud" of possible meanings exists in spatial space (superposition of waveforms), and the settling must navigate this spatial landscape to find the right attractor. The spectral proximal ensures we stay on the spectrally sparse manifold throughout.

However, the hybrid approach warrants prototyping. If the coarse attractor basin is reliably selected in the first few Langevin steps, the remaining steps could operate entirely in the efficient spectral domain — reducing per-step cost from $O(d)$ to $O(s)$ for the final settling phase.

### 11.4.1 Prototyping Both Paradigms

Despite the theoretical argument for Paradigm B (§2.4), empirical validation requires prototyping both directions to determine which provides cleaner Langevin dynamics:

- **Paradigm A prototype**: v11 architecture with spectral transport (tokens natively spatial, Fourier used for propagation only). Baseline comparison.
- **Paradigm B prototype**: v12 as described (tokens natively spectral, spatial only during settling). The primary architecture.
- **Key diagnostic**: compare the Langevin trajectory smoothness, attractor convergence rate, and final sparsity quality between the two. If Paradigm B produces sharper attractors with fewer steps, the Heisenberg argument is empirically confirmed. If Paradigm A produces equivalent results, the transport efficiency gain of B still favors it, but the margin is smaller than predicted.

### 11.5 Non-Abelian Spectral Gauge Group

The current formulation uses $U(1)$ gauge (per-frequency phase shifts). A non-Abelian spectral gauge $SU(N)$ would allow *mixing between frequency modes* during transport — frequency $\omega_1$ could partially rotate into frequency $\omega_2$ based on context. This would implement the "frequency migration" described in §6.4 as a continuous gauge transformation rather than a discrete reassignment.

The mathematical framework supports this: replace $\exp(-i\omega A)$ (phase rotation) with $\mathcal{P}\exp(-i \int \mathbf{A})$ (path-ordered matrix exponential in the Lie algebra $\mathfrak{su}(N)$). The curvature becomes a matrix-valued 2-form. The computational cost increases from $O(s)$ to $O(s \cdot N^2)$ per subbundle.

### 11.6 Relationship to Wavelets

The fixed-frequency Fourier representation has a known limitation: it cannot represent signals that are localized in BOTH time and frequency (only time-OR-frequency). Wavelet representations (Gabor atoms, scattering transforms) provide time-frequency localization at the cost of a structured overcomplete dictionary.

Should v12 use wavelets instead of pure Fourier modes? The wavelet approach would replace $e^{i\omega t}$ with $\psi_{a,b}(t) = \frac{1}{\sqrt{a}} \psi\left(\frac{t-b}{a}\right)$ — localized oscillations at scale $a$ centered at position $b$. This is a natural multi-scale dictionary, but it breaks the clean separation between spectral (sparse) and spatial (dense) that v12 exploits.

Recommendation: start with pure Fourier (clean mathematics, FFT acceleration, clear duality). If the time-frequency localization is needed, move to a Gabor dictionary (windowed Fourier) as a structured extension that preserves most of the spectral sparsity properties.

---

## Part XII: V12 Non-Negotiables

Carrying forward from v11, with spectral amendments:

1. **Sparse in spectral, dense only transiently in spatial.** Every block receives a sparse spectral section and produces a sparse spectral section. Dense spatial representations exist only during field reconstruction and Langevin settling.

2. **Field reconstruction IS the inverse Fourier transform.** The IFFT from sparse spectral modes to dense spatial field is the "diffusion" / "forward process." It is not smoothing. It is the exact reconstruction of the spatial field from its spectral sources.

3. **Langevin starts from the reconstructed field.** The dense spatial field (IFFT output) initializes the Langevin loop. The settling navigates the spatial energy landscape. The spectral proximal at each step ensures the state remains on the spectrally sparse manifold.

4. **Context warps the spectral metric.** $D_k(q)$ and $A_k(q)$ are context-dependent functions that modulate the transport kernel frequency by frequency. The same spectral sources produce different spatial fields under different contexts.

5. **Spectral proximal at every Langevin step.** Not spatial soft-thresholding (v11) but spectral top-$s_k$ projection. Progressive enforcement of spectral sparsity throughout the reverse process.

6. **Subbundles are independent spectral channels.** Different feature types occupy different spectral subbundles. The orthogonality is preserved in both spectral and spatial domains (by Parseval).

7. **No pairwise attention.** Token interactions are determined by spectral overlap — a structural property of the representation, not a learned computation.

---

## Part XIII: Why V12 Training Should Be Trivially Fast

The purpose of this architecture is not theoretical elegance for its own sake. The spectral sparsity structure is designed to make training **qualitatively faster** than even the smallest GPT variants — not 2× faster, but fast enough that character-level Tiny Shakespeare becomes a seconds-scale experiment rather than a minutes-scale one. The argument has three legs: fewer FLOPs per token, fewer tokens to convergence, and linear scaling with sequence length.

### 13.1 FLOP Comparison: V12 vs. GPT-Nano

**Baseline: GPT-Nano** (Karpathy's `microgpt.py` scaled for Shakespeare)

| Parameter | Value |
|---|---|
| vocab_size | 65 (characters) |
| n_embd (d) | 128 |
| n_head | 4, head_dim = 32 |
| n_layer | 4 |
| block_size (T) | 128 |
| Parameters | ~200K |

Per-token cost per layer:
- QKV projections: $3d^2 = 49{,}152$
- Output projection: $d^2 = 16{,}384$
- MLP (4d hidden): $2 \times 4d^2 = 131{,}072$
- Attention (at position $t$): $2td = 256t$ (scores + weighted sum)
- **Subtotal per layer**: $196{,}608 + 256t$

For the full sequence ($T = 128$, 4 layers):
- Dense matmul cost: $128 \times 4 \times 196{,}608 \approx 100.7\text{M}$
- Attention cost: $4 \times \sum_{t=0}^{127} 256t = 4 \times 256 \times 8{,}128 \approx 8.3\text{M}$
- **Total forward pass: ~109M FLOPs**

**V12-Nano** (matched scale)

| Parameter | Value |
|---|---|
| vocab_size | 65 |
| fiber_dim (d) | 128, subbundle_dim = 16 |
| n_subbundles (K) | 8 |
| spectral_sparsity ($s_k$) | 4 modes per subbundle |
| n_blocks | 3 |
| langevin_steps (L) | 5 |
| context_dim | 64 |
| atoms_per_subbundle | 24, k_active = 8 |
| Parameters | ~60K (see below) |

Per-token cost per block:
- Context accumulator (SSM): $3 \times d \times c = 3 \times 128 \times 64 = 24{,}576$
- Spectral transport: $K \times s_k \times 8 = 256$ (complex multiply on active modes)
- Field reconstruction (IFFT): $d \log_2 d = 896$
- Memory routing: $K \times 2 \times c \times n_{\text{atoms}} = 8 \times 2 \times 64 \times 24 = 24{,}576$
- Langevin settling ($L = 5$ steps):
  - Hopfield gradient per step: $K \times k \times d_{\text{sub}} = 8 \times 8 \times 16 = 1{,}024$
  - Spectral proximal per step: $2 \times d\log_2 d + K \times d_{\text{sub}}\log_2 d_{\text{sub}} \approx 2{,}304$
  - Per step: $\sim 3{,}328$. Total: $\sim 16{,}640$
- **Subtotal per block**: $\sim 66{,}944$

For the full sequence ($T = 128$, 3 blocks):
- Total: $128 \times 3 \times 66{,}944 \approx 25.7\text{M}$
- **Total forward pass: ~26M FLOPs**

| | GPT-Nano | V12-Nano | Ratio |
|---|---|---|---|
| FLOPs per sequence (T=128) | ~109M | ~26M | **4.2×** |
| Parameters | ~200K | ~60K | **3.3×** |
| Scaling with T | $O(T^2 d)$ | $O(T \cdot L \cdot K \cdot s_k \cdot d_k)$ | **Quadratic → Linear** |
| FLOPs at T=512 | ~1.2B | ~103M | **11.6×** |
| FLOPs at T=2048 | ~17B | ~411M | **41×** |

The gap widens with sequence length because V12 has **no attention** — no $O(T^2)$ term. The cost is strictly linear in $T$.

### 13.2 Parameter Efficiency

V12 eliminates the most parameter-heavy components of a Transformer:

| Component | GPT-Nano (per layer) | V12-Nano (per block) | Savings |
|---|---|---|---|
| QKV projections | $3d^2 = 49{,}152$ | 0 (no attention) | **100%** |
| Output projection | $d^2 = 16{,}384$ | 0 | **100%** |
| MLP weights | $2 \times 4d^2 = 131{,}072$ | 0 (replaced by Langevin) | **100%** |
| Transport kernel | 0 | $D_k + A_k \approx 2d = 256$ | new |
| Context accumulator | 0 | $3 \times d \times c = 24{,}576$ | new |
| Memory dictionary | 0 | $K \times n_{\text{atoms}} \times d_{\text{sub}} = 3{,}072$ | new |
| Router MLPs | 0 | $K \times 2 \times c \times n_{\text{atoms}} \approx 24{,}576$ | new |
| **Total per layer/block** | **196{,}608** | **~52{,}480** | **3.7×** |

V12 replaces dense $d \times d$ matmuls with:
- $O(d)$ spectral kernel parameters (transport)
- $O(K \times n_{\text{atoms}} \times d_{\text{sub}})$ dictionary atoms (sparse, structured)
- $O(c \times d)$ context accumulator (SSM, parallelizable)

No component requires a full $d \times d$ matrix multiplication. The heaviest operation is the context accumulator's linear projections ($128 \times 64$), which are $4\times$ smaller than GPT's smallest matmul.

### 13.3 Why Fewer Steps to Convergence

Beyond per-step efficiency, V12 should converge in fewer training steps:

**1. Structured gradients, not dense noise.**
In GPT, the gradient for a token embedding flows through dense QKV projections, softmax attention, dense MLP, and residual connections — arriving at the embedding as a dense $d$-dimensional update vector where every dimension gets a signal. Most of this signal is noise from unrelated dimensions.

In V12, the gradient for a spectral coefficient $c_j$ at frequency $\omega_j$ is the Fourier projection of the spatial gradient onto mode $j$ (§4.3). If mode $j$ is irrelevant to the current loss, its gradient is near-zero by construction. The gradient is *structurally sparse* — it updates exactly the modes that matter and leaves the rest untouched. Each gradient step carries more signal per FLOP.

**2. Explicit attractors eliminate representational search.**
A GPT must discover its own representational structure from scratch — there are no explicit "valid states" in the residual stream. The MLP and attention weights must jointly learn which representations are useful, a combinatorial search over $\mathbb{R}^d$.

V12 provides explicit attractors: the memory bank atoms $\hat{M}_q$ define the valid spectral configurations at each manifold point. The Langevin settling doesn't search $\mathbb{R}^d$ — it descends an energy landscape with known basins. Training only needs to learn (a) which spectral atoms should exist and (b) which contexts should activate which atoms. This is a structured combinatorial problem, not an unconstrained optimization.

**3. Natural curriculum from the heat kernel.**
The heat kernel $\exp(-D\omega^2)$ imposes a coarse-to-fine learning schedule without explicit curriculum design:
- Early in training, $D$ is large → high frequencies are heavily dampened → the model learns global structure first (topic, speaker, genre)
- As $D$ is refined through gradient descent, high-frequency modes are preserved → local structure is learned (syntax, morphology, collocations)

This mirrors the observation that Transformers learn long-range structure before local structure, but V12 encodes this as physics rather than relying on it to emerge.

**4. No softmax bottleneck.**
GPT's attention mechanism compresses all cross-position information through a softmax probability distribution — at each head, the model must choose a single convex combination of past values. Information that doesn't fit this weighted average is lost.

V12 has no softmax attention. Cross-position information propagates through spectral overlap (structural, lossless) and the heat kernel (smooth, frequency-selective). The spectral overlap preserves the full complex amplitude at each shared frequency — no information is destroyed by a normalization constraint.

### 13.4 The Convergence Prediction

For character-level Tiny Shakespeare (1.1M characters, vocab=65, typical training: 5K–10K steps for GPT-nano to reach ~1.5 BPC):

| Metric | GPT-Nano | V12-Nano (predicted) |
|---|---|---|
| FLOPs per step (batch=32, T=128) | ~3.5B | ~0.8B |
| Steps to 1.5 BPC | 5,000–10,000 | 1,000–3,000 (fewer due to structured gradients) |
| Total training FLOPs | ~25T | ~1.5T |
| Wall-clock (CPU, pure Python) | ~hours | ~minutes |
| Wall-clock (PyTorch, GPU) | ~minutes | ~seconds |

The 4× per-step speedup compounds with the predicted 3–5× reduction in steps needed, yielding a **12–20× total training speedup**. At GPU speeds, this makes Shakespeare training a sub-minute experiment — fast enough to iterate on architectural ideas in real-time.

### 13.5 Where the Dense Costs Hide (Honest Accounting)

Two operations prevent V12 from being *entirely* sparse:

1. **The IFFT** ($O(d \log d)$ per block): Required to produce the dense spatial field for Langevin settling. This is the most expensive per-token operation. However, it can be replaced by a **partial sum** over active modes: $\tilde{x}(n) = \sum_{j \in S} c_j e^{i\omega_j n}$, which is $O(s \times d)$ — for $s = 32$ total active modes and $d = 128$, this is 4,096 FLOPs vs. 896 for FFT. The FFT wins for $s > d / \log d \approx 18$, so at very high sparsity ($s < 18$), the partial sum is cheaper.

2. **The spectral proximal operator** ($O(d \log d)$ per Langevin step): FFT + top-$s_k$ + IFFT at each settling step. With $L = 5$ steps, this is $5 \times 2 \times 896 = 8{,}960$ FLOPs. This is the price of maintaining spectral sparsity throughout settling. It can be amortized by applying the proximal every 2nd or 3rd step instead of every step, at the cost of weaker sparsity enforcement mid-settling.

These dense costs are fixed (independent of $T$), which is why V12's advantage grows with sequence length. GPT's $O(T^2 d)$ attention cost eventually dominates everything else.

---

## Summary

V12 is the recognition that **the architecture's transport operator already lives in Fourier space** — and if we define tokens there too, the entire framework becomes self-consistent: sparsity is spectral, transport is direct, the forward–reverse loop is Fourier duality, context is accumulated phase, and routing is spectral overlap.

The architectural fork between spatial-native (Paradigm A) and spectral-native (Paradigm B) tokens is resolved in favor of B by the Heisenberg uncertainty argument, but the architecture does not abandon spatial computation. It *lives* in spectral space and *visits* spatial space — the Fourier transform is not a tool but the architectural heartbeat, the duality between sparse spectral identity and dense spatial influence that the model traverses at every block.

Whether the token originates as a spatial footprint or a spectral barcode, the core mechanic is the same: **we harness Fourier space for state management and global aggregation.** The feature-geometry duality — continuous spectral waves vs. discrete spatial attractors — is not a tension to be resolved but a structure to be exploited, with each domain providing what the other cannot.

---

*Research notes, March 18, 2026. David Ledbetter, with Claude.*
*The architecture was always trying to be spectral. V12 lets it.*
