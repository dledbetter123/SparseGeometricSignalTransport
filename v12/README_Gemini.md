# V12 Architecture: Dual Sparsity — Spatial vs. Spectral Routing

## 1. The Core Insight: Tokens, Patterns, and the Fourier Domain

In V11, we defined a token as a sparse event—a set of point-source activations on a context-warped fiber bundle in the spatial domain. 
In V12, we introduce the **Fourier dual space** into this formulation. The fundamental insight is mathematical equivalence with profound computational implications: **any unique activation pattern in the spatial domain maps to a unique spectral pattern in Fourier space.** 

The critical advantage of the Fourier space is management and global reach. Combining signals from patterns representing tokens in the frequency domain natively superimposes their global, sequence-wide effects. We no longer have to brute-force route spatial signals across vast distances using attention; instead, intersecting Fourier components algebraically synthesizes global coherence.

However, this raises a fundamental architectural fork in how we define a token's intrinsic representation:
Are tokens natively objects in **Spatial Space** or **Fourier Space**?

---

## 2. Paradigm A: The Token as a Spatial Pattern (Spectral Propagation)

In this direction, we retain the V11 definition where the token's ground truth is a **sparse pattern of values in the normal spatial domain**.

**The Mechanics:**
1. **Local Activation:** A token is a set of sparse fiber-bundle activations at position $t$.
2. **Spectral Projection:** To manage context and route information globally, we project these spatial agitations into the Fourier domain. 
3. **Global Combination:** In the Fourier manifold, these patterns are combined. Since low-frequency components stretch across the sequence, mixing them here is computationally cheap and globally effective.
4. **Pullback:** The combined spectral field is Inverse Fourier Transformed back to the spatial domain to determine the resulting local potential landscape.

**Pros & Cons:**
- *Pro:* Biologically intuitive (resembles local point-source spikes). Easy to conceptualize discrete sequential events.
- *Con:* Requires a continuous forward/inverse spatial-to-spectral projection at every layer or step.

---

## 3. Paradigm B: The Token as a Spectral Pattern (The Fourier Well)

In this direction, the fundamental identity of a token lives entirely in the frequency domain. A token is defined a priori as **an assortment of active "dots" (delta functions) or potential wells in Fourier space**. 

**The Mechanics:**
1. **Spectral Activation:** Every token in the vocabulary is represented by a unique set of excited frequencies—a sparse spectral fingerprint $X(\omega)$. 
2. **Local Routing & Training:** Because the vocabulary lives in the frequency domain, routing and training become strictly *local* within Fourier space. We update the frequency peaks and connections directly.
3. **Emergent Spatial Reality:** The "spatial domain" is simply the consequent continuous wave—a global envelope caused by the interference of these Fourier patterns. The model doesn't "predict the next token in space"; it balances the spectral energy so that the spatial interference pattern collapses cleanly into the next concept.

**Pros & Cons:**
- *Pro:* Massive global reach. A single combination of dots in Fourier space inherently creates a wide-ranging spatial pattern, making it vastly easier to manage long-term dependencies.
- *Con:* The concept of strict "position and order" in a sequence requires encoding precise phase relationships in the Fourier domain. High-frequency phase management can become noisy.

---

## 4. Unifying the Dual Approaches

Whether the token originates as a spatial footprint or a spectral barcode, the core mechanic of V12 is the same: **We harness Fourier space for easier state management and global aggregation.** 

Instead of operating exclusively on one end of the transform, V12 treats the token representations as a **Feature-Geometry Duality**:

$$ \underbrace{x(t)}_{\text{Spatial Sparse Pattern}} \xleftrightarrow{\mathcal{F}} \underbrace{X(\omega)}_{\text{Spectral Sparse Pattern}} $$

By actively training the network to modulate the signal in whichever domain provides the sparsest, cleanest gradients:
- Transient, syntactical features can be managed dynamically in the spatial domain.
- Broad, thematic context can be managed efficiently in the Fourier domain.

**Next Steps for V12:** We must prototype both directions to see which provides cleaner Riemannian Langevin dynamics upon settling. The decision hinges on whether it is mathematically cleaner to pull a global spectral wave back into a sharp spatial point, or to collapse a spatial sequence into a sharp unified spectral eigenstate.
