"""Modulation-signature KNOWLEDGE BASE for RML2018.01A (24 classes).

Authoritative, textbook AMC signatures (family, envelope, constellation/phase/amplitude
structure, frequency-modulation, higher-order-cumulant cues, spectral shape). Injected
to the LLM via RAG so a small model that lacks baked-in RF domain knowledge can still
reason signal-description -> modulation, INCLUDING classes it has no labeled examples for.
Each entry also has a coarse feature-template (for the deterministic KB-matcher baseline):
template = [envelope_var, freq_mod, phase_states_norm, amp_levels_norm, c40, c42, analog].
"""

# textual signatures (knowledge for the LLM)
KB_TEXT = {
    "OOK": "On-Off Keying: amplitude-shift keying with 2 amplitude levels (on/off); strong amplitude variation; not constant envelope; no phase information; digital, narrowband.",
    "4ASK": "4-level Amplitude-Shift Keying: 4 amplitude levels on one carrier phase; strong amplitude variation; constant phase; digital.",
    "8ASK": "8-level ASK: 8 amplitude levels, real-valued; very strong amplitude variation; single phase; digital.",
    "BPSK": "Binary PSK: 2 phase states (0, pi); constant envelope; no amplitude info; large |C40|; digital.",
    "QPSK": "Quadrature PSK: 4 equally-spaced phase states; constant envelope; no amplitude levels; moderate |C40|; digital.",
    "8PSK": "8-PSK: 8 equally-spaced phase states; constant envelope; small |C40|; finer phase discreteness than QPSK.",
    "16PSK": "16-PSK: 16 phase states on a circle; constant envelope; very fine phase discreteness; |C40| near zero.",
    "32PSK": "32-PSK: 32 phase states; constant envelope; extremely fine phase discreteness; near-continuous phase ring.",
    "16APSK": "16-APSK: amplitude-and-phase shift keying, 2 amplitude rings of phase points; mild amplitude variation + multiple phases; digital satellite waveform.",
    "32APSK": "32-APSK: 3 amplitude rings of phase points; moderate amplitude variation + many phases; digital satellite.",
    "64APSK": "64-APSK: several amplitude rings, many phases; moderate-to-strong amplitude variation; dense.",
    "128APSK": "128-APSK: many amplitude rings and phases; strong amplitude+phase structure; very dense constellation.",
    "16QAM": "16-QAM: 4x4 rectangular grid, 3 amplitude levels; clear amplitude variation; |C40| near zero (distinguishes from PSK); digital.",
    "32QAM": "32-QAM: cross/grid constellation, more amplitude levels; clear amplitude variation; |C40| near zero.",
    "64QAM": "64-QAM: 8x8 grid, many amplitude levels; strong amplitude variation; |C40|~0; dense.",
    "128QAM": "128-QAM: large cross constellation; strong amplitude variation; very dense; |C40|~0.",
    "256QAM": "256-QAM: 16x16 grid; very dense, strong amplitude variation; |C40|~0; highest-order common QAM.",
    "AM-SSB-WC": "Analog AM single-sideband with carrier: asymmetric spectrum (one sideband), amplitude-varying, analog (non-discrete), residual carrier tone.",
    "AM-SSB-SC": "Analog AM single-sideband suppressed-carrier: asymmetric one-sided spectrum, amplitude-varying, analog, no carrier tone.",
    "AM-DSB-WC": "Analog AM double-sideband with carrier: symmetric spectrum about carrier, strong amplitude variation tracking message, analog, carrier tone present.",
    "AM-DSB-SC": "Analog AM double-sideband suppressed-carrier: symmetric two-sided spectrum, amplitude-varying, analog, no carrier tone.",
    "FM": "Analog Frequency Modulation: constant envelope, instantaneous frequency varies continuously with the message; broadband; analog (non-discrete phase/amplitude).",
    "GMSK": "Gaussian Minimum-Shift Keying: constant envelope, continuous-phase frequency modulation (CPM) with Gaussian pulse shaping; no amplitude info; compact spectrum; digital.",
    "OQPSK": "Offset QPSK: 4 phase states with half-symbol I/Q offset (avoids zero crossings); near-constant envelope; no amplitude levels; digital.",
}

# coarse normalized templates for the deterministic KB-matcher baseline:
# [envelope_variation(0-1), freq_mod(0-1), phase_states(norm), amp_levels(norm), c40_large(0-1), analog(0-1)]
KB_TMPL = {
    "OOK":      [0.9, 0.0, 0.0, 0.2, 0.0, 0.0],
    "4ASK":     [0.8, 0.0, 0.0, 0.4, 0.0, 0.0],
    "8ASK":     [0.9, 0.0, 0.0, 0.8, 0.0, 0.0],
    "BPSK":     [0.05, 0.0, 0.1, 0.0, 1.0, 0.0],
    "QPSK":     [0.05, 0.0, 0.2, 0.0, 0.6, 0.0],
    "8PSK":     [0.05, 0.0, 0.4, 0.0, 0.3, 0.0],
    "16PSK":    [0.05, 0.0, 0.7, 0.0, 0.1, 0.0],
    "32PSK":    [0.05, 0.0, 0.9, 0.0, 0.05, 0.0],
    "16APSK":   [0.4, 0.0, 0.4, 0.4, 0.1, 0.0],
    "32APSK":   [0.5, 0.0, 0.6, 0.5, 0.05, 0.0],
    "64APSK":   [0.6, 0.0, 0.7, 0.6, 0.05, 0.0],
    "128APSK":  [0.7, 0.0, 0.8, 0.7, 0.05, 0.0],
    "16QAM":    [0.55, 0.0, 0.5, 0.5, 0.05, 0.0],
    "32QAM":    [0.6, 0.0, 0.6, 0.6, 0.05, 0.0],
    "64QAM":    [0.7, 0.0, 0.7, 0.7, 0.05, 0.0],
    "128QAM":   [0.75, 0.0, 0.8, 0.8, 0.05, 0.0],
    "256QAM":   [0.8, 0.0, 0.9, 0.9, 0.05, 0.0],
    "AM-SSB-WC":[0.6, 0.2, 0.0, 0.5, 0.0, 1.0],
    "AM-SSB-SC":[0.6, 0.2, 0.0, 0.5, 0.0, 1.0],
    "AM-DSB-WC":[0.8, 0.2, 0.0, 0.6, 0.0, 1.0],
    "AM-DSB-SC":[0.8, 0.2, 0.0, 0.6, 0.0, 1.0],
    "FM":       [0.05, 1.0, 0.0, 0.0, 0.0, 1.0],
    "GMSK":     [0.05, 0.8, 0.0, 0.0, 0.0, 0.0],
    "OQPSK":    [0.15, 0.0, 0.2, 0.0, 0.5, 0.0],
}
