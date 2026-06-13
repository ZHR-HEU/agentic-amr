"""Range-based spectrum-allocation knowledge base (ITU/FCC-style) for open-world AMR.

Real frequency-allocation tables are organized by RANGE, not by point frequency.
A small model with RAG over this KB can therefore handle a carrier frequency that
no point-table lists, by retrieving the allocation RANGE that contains it. This is
the deployable, reproducible alternative to a frontier model's baked-in knowledge.

Each entry: (lo_MHz, hi_MHz, service description, candidate modulations, primary).
'candidate' = modulations realistically found in that band; 'primary' = the single
most common (used by the deterministic range_det baseline). Names match RML2018.01A.

This KB is used by BOTH:
  - the deterministic baselines (range_det, range_sig) -- KB knowledge, NO LLM, the
    fair baseline an LLM must beat to justify itself; and
  - the RAG-LLM (retrieve the containing range's text, reason, fuse signal).
"""

# (lo, hi, service, candidates, primary)
SPECTRUM_KB = [
    (88.0, 108.0, "FM broadcast radio", ["FM"], "FM"),
    (108.0, 137.0, "aeronautical VHF voice (airband)", ["AM-DSB-WC"], "AM-DSB-WC"),
    (137.0, 138.0, "meteorological satellite APT downlink (e.g. NOAA POES)", ["FM"], "FM"),
    (156.0, 162.05, "maritime VHF (voice channels + AIS transponders)", ["FM", "GMSK"], "FM"),
    (225.0, 400.0, "UHF military/SATCOM voice", ["FM", "AM-DSB-WC"], "FM"),
    (824.0, 960.0, "cellular GSM up/downlink", ["GMSK"], "GMSK"),
    (1215.0, 1240.0, "GNSS L2 navigation (GPS L2C)", ["BPSK"], "BPSK"),
    (1525.0, 1660.5, "L-band mobile-satellite service", ["QPSK", "OQPSK"], "QPSK"),
    (2400.0, 2500.0, "2.4 GHz ISM (WiFi / Bluetooth / Zigbee)", ["OQPSK", "QPSK", "GMSK"], "OQPSK"),
    (5725.0, 5875.0, "5 GHz ISM / U-NII WLAN", ["64QAM", "QPSK"], "64QAM"),
    (10700.0, 12750.0, "Ku-band FSS satellite downlink (DVB-S2)", ["QPSK", "8PSK", "16APSK", "32APSK"], "QPSK"),
    (14000.0, 14500.0, "Ku-band FSS satellite uplink (DVB-S2)", ["QPSK", "8PSK", "16APSK"], "QPSK"),
]


def retrieve(freq_mhz):
    """Return the allocation entry whose RANGE contains freq, else None.
    This is the RAG retrieval step (range match), the key to handling point
    frequencies absent from any point-table."""
    for lo, hi, svc, cands, prim in SPECTRUM_KB:
        if lo <= freq_mhz <= hi:
            return {"lo": lo, "hi": hi, "service": svc, "candidates": cands, "primary": prim}
    return None


def retrieve_text(freq_mhz):
    """Textual KB snippet for the RAG-LLM prompt (the 'retrieved document')."""
    e = retrieve(freq_mhz)
    if e is None:
        return "No matching spectrum-allocation entry was retrieved for this frequency."
    return (f"Retrieved spectrum-allocation entry: {e['lo']:g}-{e['hi']:g} MHz is allocated to "
            f"{e['service']}; modulations typically used in this band: {', '.join(e['candidates'])}.")


def range_det_pred(freq_mhz):
    """Deterministic baseline #1: predict the retrieved range's PRIMARY modulation
    (range lookup only, NO signal). Fails when the band hosts several modulations."""
    e = retrieve(freq_mhz)
    return e["primary"] if e else "UNKNOWN"


def range_sig_pred(freq_mhz, probs, mods):
    """Deterministic baseline #2 (the critical fair baseline): retrieve the range's
    candidate set, then pick the candidate with the highest CNN probability
    (KB-restricted signal argmax). Same KB knowledge as the RAG-LLM, plus signal
    fusion, but NO LLM reasoning. If an LLM cannot beat THIS, the value is the
    knowledge+signal, not the LLM."""
    e = retrieve(freq_mhz)
    if e is None:
        return "UNKNOWN"
    cset = [m for m in e["candidates"] if m in mods]
    if not cset:
        return e["primary"]
    pm = {m: probs[mods.index(m)] for m in cset}
    return max(pm, key=pm.get)
