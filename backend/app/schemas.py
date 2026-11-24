"""
Pydantic schemas for the two places an LLM answer must be machine-checked: free-text
synthesis-condition extraction (Condition Processor) and an encoding proposal for an
ambiguous numeric-looking-but-categorical parameter (Encoding Determination). Plain API
request/response bodies elsewhere in the backend are left as dicts for flexibility; these two
need a strict schema because they drive numeric computation downstream.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ReagentAmount(BaseModel):
    name: str = Field(description="Reagent name exactly as it would appear in a chemistry notebook, "
                                   "e.g. 'Ni(NO3)2.6H2O', 'LiOH.H2O', 'NMC111'.")
    amount: float
    unit: Literal["mg", "g"]


class ThermalStep(BaseModel):
    temp_C: float = Field(description="Hold temperature in Celsius.")
    hold_h: float = Field(description="Hold duration in hours at this temperature.")


class AnnealSpec(BaseModel):
    temp_C: float
    atmosphere: Literal["air", "O2"]
    hold_h: float
    liOH_added: bool = Field(description="Whether extra LiOH was added before/during the post-wash anneal.")
    liOH_wt_pct: Optional[float] = Field(
        default=None, description="The added LiOH amount in wt%, ONLY if a number is actually stated in the "
                                   "text. Leave null if liOH_added is true but no percentage is given.")


class ConditionExtraction(BaseModel):
    """What the Condition Processor asks the LLM to pull out of one free-text notebook entry.
    Numeric derived quantities (molar ratios, thermal-profile summary stats) are computed
    deterministically from this structured extraction afterwards -- the LLM is never asked to
    do the arithmetic itself."""
    sample_id: str = Field(description="The sample identifier exactly as written, e.g. 'NMC811-23'.")
    target_material: str = Field(description="The target product family, e.g. 'NMC811' or 'NMC955'.")
    reagents: list[ReagentAmount] = Field(description="Every feed reagent and its amount, in the units given.")
    ball_milled: Optional[bool] = Field(description="Whether the NMC111 precursor was ball-milled before calcination.")
    dried_powder: Optional[bool] = Field(description="Whether the mixture was vacuum-dried and ground to a powder "
                                                        "before calcination, as opposed to calcined wet/as a slurry.")
    calcination_atmosphere: Optional[Literal["air", "O2", "NR"]] = Field(
        description="Furnace atmosphere during calcination; 'NR' if not stated.")
    thermal_steps: list[ThermalStep] = Field(
        description="Every temperature hold during calcination, IN THE ORDER THEY HAPPEN, including any "
                    "low-temperature pre-hold(s) below 600 C as well as the main high-temperature hold(s).")
    ramp_rates_C_per_min: list[float] = Field(
        description="Every heating/cooling ramp rate mentioned, in the order mentioned (deg C per minute).")
    wash_procedure: Optional[str] = Field(description="The wash/centrifuge procedure, in a short standardized phrase.")
    anneal: Optional[AnnealSpec] = Field(description="Post-wash anneal details, or null if the sample was not annealed.")
    notes: Optional[str] = Field(description="Anything ambiguous, unusual, or that needed a judgment call to parse "
                                              "(e.g. a likely typo, an unusual reagent, duplicate amounts listed twice).")
    missing_fields: list[str] = Field(default_factory=list,
                                       description="Names of fields above that could NOT be determined from the "
                                                   "text (leave the field itself null/empty and list its name here).")


class EncodingProposal(BaseModel):
    """What the Encoding Determination agent asks the LLM to propose for one parameter whose
    values look numeric but aren't cleanly a single float (e.g. pre_hold: 'none', '350C 5h',
    '180C 5h; 350C 5h'). The user reviews/edits this before it's applied."""
    recommended_method: Literal["label_encode_by_frequency", "one_hot", "decompose_numeric"] = Field(
        description="label_encode_by_frequency: one integer code per distinct value, most frequent = lowest code "
                    "(matches the template's 'Category codes' convention). one_hot: a 0/1 column per distinct "
                    "value. decompose_numeric: break each value into numeric sub-features (e.g. number of holds, "
                    "total hold time, peak hold temperature) when the strings actually encode structured "
                    "numeric information -- usually the best choice for hold/anneal-like fields.")
    rationale: str = Field(description="One or two sentences on why this method fits this parameter's values.")
    label_codes: Optional[dict[str, int]] = Field(
        default=None, description="Only for label_encode_by_frequency: value -> integer code.")
    decomposed_features: Optional[list[dict]] = Field(
        default=None, description="Only for decompose_numeric: a list of {name, description, values: "
                                   "{original_value: numeric_value}} -- one entry per new numeric feature.")
