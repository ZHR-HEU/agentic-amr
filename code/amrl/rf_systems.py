"""Real-world band -> system -> modulation associations (open-world RF triage).

These are REAL facts an LLM has from pretraining but a model trained only on the
(band-held-out) training data cannot know. Used to test whether an LLM can
disambiguate low-SNR / open-world AMR via world knowledge where signal-only and
from-data models structurally cannot. Modulation names match RML2018.01A classes.

Each entry: freq_MHz -> (system, modulation, region/notes).
"""
# TRAIN bands (their band->mod mapping IS available to from-data baselines)
TRAIN_BANDS = {
    100.0: ("FM broadcast", "FM", "88-108 MHz commercial FM radio"),
    940.0: ("GSM downlink", "GMSK", "GSM 900 cellular downlink"),
    1575.42: ("GPS L1 C/A", "BPSK", "GNSS navigation"),
    5800.0: ("802.11ac WiFi", "64QAM", "5 GHz WLAN high-MCS"),
    12000.0: ("DVB-S satellite", "QPSK", "Ku-band satellite TV downlink"),
}
# TEST bands (HELD OUT from baseline training; only world knowledge maps them).
# Each test band maps UNAMBIGUOUSLY to one modulation (2.4 GHz ISM dropped: shared by
# BT/Zigbee/WiFi -> not a unique band->modulation mapping, per frontier-model self-check).
TEST_BANDS = {
    162.0: ("AIS maritime", "GMSK", "marine VHF AIS 161.975/162.025 MHz"),
    124.0: ("aviation AM voice", "AM-DSB-WC", "118-137 MHz airband AM"),
    433.92: ("ISM remote/RFID", "OOK", "433 MHz ISM on-off keying"),
    1090.0: ("ADS-B Mode-S", "OOK", "aviation transponder PPM @1090 MHz (OOK-family)"),
}

ALL_BANDS = {**TRAIN_BANDS, **TEST_BANDS}


def band_text(freq_mhz):
    """Minimal neutral context given to the agent: ONLY the frequency (so the LLM must
    recall what operates there). The system/modulation are NOT revealed."""
    return f"carrier frequency ~ {freq_mhz:g} MHz"
