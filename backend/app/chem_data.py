"""
Reagent reference data for the Condition Processor agent.

Ported and generalized from the reference `parse_cond_v2.py` script: that script had a
hard-coded regex tuned to one notebook's exact phrasing. Here we keep the same chemistry
(molecular weights, which reagents contribute Li/Na/K, nitrate, chloride, hydroxide and
sulfate per mole) but as open, extensible lookup tables that the LLM-based extraction step
can be checked against, instead of a frozen regex.

If the LLM extracts a reagent name that is not in MOLAR_MASS, the Condition Processor marks
that sample's row as masked (missing) rather than guessing a molecular weight.
"""

# g/mol. Keys are canonical reagent names. Aliases map common alternate spellings/scripts to
# the canonical name so the LLM extraction (which may echo back the notebook's own spelling)
# still resolves.
MOLAR_MASS = {
    "NMC111": 96.46,
    "Ni(NO3)2.6H2O": 290.79,
    "Ni(SO4)2.6H2O": 262.85,
    "LiCl": 42.39,
    "NaOH": 40.00,
    "KOH": 56.11,
    "LiOH_anhydrous": 23.95,   # 无水LiOH
    "LiOH.H2O": 41.96,
    "LiOH": 23.95,
    "LiNO3": 68.95,
    "NaNO3": 84.99,
    "NaCl": 58.44,
    "KCl": 74.55,
    "KBr": 119.00,
    "Na2SO4": 142.04,
    "MnCl2.4H2O": 197.91,
    "Mn(NO3)2.4H2O": 251.01,
    "La(NO3)3.6H2O": 433.01,
    "Al(NO3)3.9H2O": 375.13,
}

ALIASES = {
    "无水LiOH": "LiOH_anhydrous",
    "LiOH·H2O": "LiOH.H2O",
    "Ni(NO3)2·6H2O": "Ni(NO3)2.6H2O",
    "Ni(SO4)2·6H2O": "Ni(SO4)2.6H2O",
    "NiSO4·6H2O": "Ni(SO4)2.6H2O",
    "MnCl2·4H2O": "MnCl2.4H2O",
    "Mn(NO3)2·4H2O": "Mn(NO3)2.4H2O",
    "La(NO3)3·6H2O": "La(NO3)3.6H2O",
    "Al(NO3)3·9H2O": "Al(NO3)3.9H2O",
}

# moles of Li / Na / K contributed per mole of reagent (1 unless the formula carries more
# than one of that ion, e.g. Na2SO4 carries 2 Na).
LI_PER_MOL = {"LiCl": 1, "LiOH_anhydrous": 1, "LiOH.H2O": 1, "LiOH": 1, "LiNO3": 1}
NA_PER_MOL = {"NaOH": 1, "NaNO3": 1, "NaCl": 1, "Na2SO4": 2}
K_PER_MOL = {"KOH": 1, "KCl": 1, "KBr": 1}

# moles of NO3- / Cl- / OH- / SO4(2-) contributed per mole of reagent.
# NOTE: Al(NO3)3.9H2O is intentionally absent here -- the reference pipeline's nitrate tally
# (parse_cond_v2.py) never included the Al dopant's nitrate, even though Al_pct_TM tracks the
# same reagent's Al contribution separately. Kept bug-compatible so reprocessing old notebook
# entries reproduces the exact nitrate_per_TM values already baked into existing training data;
# Al's own contribution is fully captured via AL_SOURCE_REAGENTS/Al_pct_TM instead.
NITRATE_PER_MOL = {"Ni(NO3)2.6H2O": 2, "LiNO3": 1, "NaNO3": 1, "Mn(NO3)2.4H2O": 2, "La(NO3)3.6H2O": 3}
CHLORIDE_PER_MOL = {"LiCl": 1, "NaCl": 1, "KCl": 1, "MnCl2.4H2O": 2}
HYDROXIDE_PER_MOL = {"NaOH": 1, "KOH": 1, "LiOH_anhydrous": 1, "LiOH.H2O": 1, "LiOH": 1}
SULFATE_PER_MOL = {"Na2SO4": 1, "Ni(SO4)2.6H2O": 1}

# Reagents that supply Ni / Mn to the transition-metal pool in addition to the NMC111 feed.
NI_SOURCE_REAGENTS = {"Ni(NO3)2.6H2O", "Ni(SO4)2.6H2O"}
MN_SOURCE_REAGENTS = {"MnCl2.4H2O", "Mn(NO3)2.4H2O"}
LA_SOURCE_REAGENTS = {"La(NO3)3.6H2O"}
AL_SOURCE_REAGENTS = {"Al(NO3)3.9H2O"}

HIGH_TEMP_THRESHOLD_C = 600.0  # steps at/above this count toward T_peak/T_final/t_total_hi;
                                # steps below it are "pre_hold" (matches parse_cond_v2.py)


def canonical_name(name: str) -> str:
    name = name.strip()
    return ALIASES.get(name, name)


def molar_mass(name: str):
    return MOLAR_MASS.get(canonical_name(name))


def known_reagent(name: str) -> bool:
    return canonical_name(name) in MOLAR_MASS
