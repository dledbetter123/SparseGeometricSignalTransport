# Finding: The irfft Round-Trip May Be Unnecessary

**Date**: 2026-04-01

## The Question

Why do we convert spectral → spatial (irfft) → process with FFN → spatial → spectral (rfft) every block? Can we stay in spectral space?

## Analysis

### What irfft actually computes

`irfft(spectral_modes)` per position computes:
```
spatial[d] = sum over k: mag[k] * cos(2π·k·d/N + phase[k])
```

This is a matrix multiply by the inverse DFT matrix — a fixed orthogonal linear transform. Every position gets the same matrix applied to its own modes. It converts frequency-localized features to spatially-localized features.

### What we thought we were getting

1. Parseval energy conservation: ||spatial||² = ||spectral||²
2. "Multiplication in spectral = convolution in spatial" (the mixing argument)
3. A natural basis for the FFN

### What we're actually getting

The convolution theorem argument (point 2) only applies to the position-axis FFT, which we had to remove because it's non-causal. The mode-axis irfft is NOT mixing tokens — it's transforming features within a single position. It's a basis change, not a mixing operation.

The FFN doesn't care which basis its input is in. Its first Linear layer can learn any basis transformation. The DFT matrix is just one particular orthogonal matrix among many — the FFN could learn whatever rotation it needs from raw (mag, phase) features.

### The Parseval energy guarantee

Parseval says: ||spatial||² = ||spectral||². If you control energy in one domain, it's controlled in both.

CloudNorm already does this:
```python
mag_rms = (c.mag ** 2).mean(dim=-1, keepdim=True).sqrt()
normalized_mag = c.mag / mag_rms * learned_scale
```

This IS spectral energy normalization. By Parseval, spatial energy is also controlled — whether or not we compute the spatial representation. The guarantee holds even without the irfft.

Combined with the Parseval filter constraint (|W| ≤ 1 → energy can only decrease), the full energy bound is:
```
E_output ≤ E_input (filter)
E_input is controlled (CloudNorm)
∴ E_output is bounded
```

This holds entirely in spectral space. The irfft makes it explicit but doesn't add anything.

### Positional encoding in spectral space

Position is already encoded in spectral space via phase shift: `phase += pos * freqs`. This is mathematically equivalent to RoPE (rotary position embeddings). The phases carry positional information through the fiber and filter.

The irfft was the only reason we needed spatial positional encoding (nn.Embedding) — because after converting to spatial domain, the phase-based positional information gets mixed into the DFT basis and the FFN can't distinguish positions. Without irfft, the phases remain explicit and carry position throughout.

## Conclusion

The irfft/rfft round-trip is a fixed linear transform (DFT basis change) applied per-position per-block. It provides:
- A specific basis for FFN processing (but FFN can learn its own)
- Explicit Parseval energy equivalence (but CloudNorm already controls this)
- No cross-position mixing (that's only from the position-axis FFT, which is non-causal)

Removing it saves 2 FFT operations per block (irfft for constellation, irfft for filtered, rfft to convert back) and lets the FFN operate directly on spectral features where positional encoding (phase) is explicit.

## Implementation: V16c

Stay entirely in spectral space. No irfft/rfft per block. The block operates on (mag, phase, log_var) throughout:

```
CloudNorm(constellation)
├── Wilson fiber (causal EMA on complex spectral coefficients)
├── Parseval filter (spectral gating + cross-mode)
├── Local conv on spectral features (causal)
└── FFN on spectral features (mag, phase, log_var, filtered)
→ Updated constellation
```

Energy guarantees preserved via CloudNorm + Parseval constraint. No basis change needed.
