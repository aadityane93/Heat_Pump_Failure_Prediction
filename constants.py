from __future__ import annotations

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

# Columns in protocols.csv where True means "there is a fault".
DIRECT_FAULT_COLUMNS = [
    "HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit",
    "HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit",
    "HeatPump_NightSetbackSetting_Activated_BeforeVisit",
    "DHW_Storage_LastDescaling_TooLongAgo",
    "HeatPump_AirSource_AirDuctsCleaningRequired",
]

# Columns where False means "there is a fault". The built flag gets `_inv`.
INVERTED_FAULT_COLUMNS = [
    "HeatPump_TechnicallyOkay",
    "HeatPump_BasicFunctionsOkay",
    "HeatPump_Installation_CorrectlyPlanned",
    "HeatPump_AirSource_AirDuctsDistanceOkay",
    "HeatPump_AirSource_AirDuctsFree",
    "HeatPump_AirSource_AirDuctsDrainOkay",
    "HeatPump_AirSource_EvaporatorClean",
]

CATEGORICAL_FAULT_COLUMN = "HeatDistribution_ExpansionTank_Pressure_Categorization"
CATEGORICAL_FAULT_FLAG = "ExpansionTank_fault"

FAULT_FLAG_COLUMNS = (
    DIRECT_FAULT_COLUMNS
    + [c + "_inv" for c in INVERTED_FAULT_COLUMNS]
    + [CATEGORICAL_FAULT_FLAG]
)

# Never allowed into the feature matrix.
LABEL_COLUMNS = ["has_fault", "n_faults"] + FAULT_FLAG_COLUMNS
# Identifiers and study bookkeeping: descriptive, but not something a model
# should learn from.
META_COLUMNS = ["Household_ID", "Group", "Visit_Date", "Visit_Year"]

FAULT_DISPLAY_NAMES = {
    "has_fault": "Any fault found at the visit",
    "HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit": "Heating curve set too high",
    "HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit": "Heating limit set too high",
    "HeatPump_NightSetbackSetting_Activated_BeforeVisit": "Night setback active",
    "DHW_Storage_LastDescaling_TooLongAgo": "Hot water tank overdue for descaling",
    "HeatPump_AirSource_AirDuctsCleaningRequired": "Air ducts need cleaning",
    "HeatPump_TechnicallyOkay_inv": "Not technically okay",
    "HeatPump_BasicFunctionsOkay_inv": "Basic functions not okay",
    "HeatPump_Installation_CorrectlyPlanned_inv": "Installation not correctly planned",
    "HeatPump_AirSource_AirDuctsDistanceOkay_inv": "Air duct clearance not okay",
    "HeatPump_AirSource_AirDuctsFree_inv": "Air ducts obstructed",
    "HeatPump_AirSource_AirDuctsDrainOkay_inv": "Air duct drain not okay",
    "HeatPump_AirSource_EvaporatorClean_inv": "Evaporator not clean",
    CATEGORICAL_FAULT_FLAG: "Expansion tank pressure not okay",
}


def pretty_target(name: str) -> str:
    return FAULT_DISPLAY_NAMES.get(name, name.replace("_", " "))


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

STATIC_COLUMNS = [
    "Building_Type",
    "Building_ConstructionYear",
    "Building_FloorAreaHeated_Total",
    "Building_Residents",
    "Building_PVSystem_Available",
    "Building_ElectricVehicle_Available",
    "HeatPump_Installation_Type",
    "HeatPump_Installation_Year",
    "HeatPump_Installation_HeatingCapacity",
    "HeatPump_Installation_Normpoint_COP",
    "HeatDistribution_System_Radiators",
    "HeatDistribution_System_FloorHeating",
    "HeatDistribution_System_BufferTankAvailable",
    "DHW_Production_ByHeatPump",
    "DHW_Production_Residents",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "Consumption level": [
        "e_mean",
        "e_std",
        "e_max",
        "e_cv",
        "e_p90",
        "e_zero_frac",
        "hp_share",
        "used_hp_submeter",
    ],
    "Weather normalisation": [
        "hdd_slope",
        "hdd_intercept",
        "hdd_r2",
        "e_per_hdd",
    ],
    "Seasonal behaviour": [
        "summer_baseload",
        "winter_mean",
        "winter_summer_ratio",
        "mild_weather_use",
        "mild_vs_summer",
    ],
    "Day-to-day pattern": [
        "e_diff_std",
        "weekend_ratio",
    ],
    "Measurement context": [
        "n_days",
        "mean_temp",
        "mean_hdd",
        "sunshine_mean",
    ],
    "Building and installation": STATIC_COLUMNS,
}

FEATURE_DESCRIPTIONS = {
    "e_mean": "Mean daily energy use",
    "e_std": "Standard deviation of daily energy use",
    "e_max": "Highest single day",
    "e_cv": "Coefficient of variation (std / mean)",
    "e_p90": "90th percentile day",
    "e_zero_frac": "Share of days below 0.5 kWh",
    "hp_share": "Heat pump share of household consumption",
    "used_hp_submeter": "1 if the heat pump has its own meter channel",
    "hdd_slope": "kWh gained per heating degree day",
    "hdd_intercept": "Consumption extrapolated to zero heating demand",
    "hdd_r2": "How well heating degree days explain consumption",
    "e_per_hdd": "Total energy divided by total heating degree days",
    "summer_baseload": "Mean use on days with no heating demand",
    "winter_mean": "Mean use below 5 °C",
    "winter_summer_ratio": "Winter mean over summer baseload",
    "mild_weather_use": "Mean use between 15 and 20 °C",
    "mild_vs_summer": "Mild-weather use over summer baseload",
    "e_diff_std": "Mean absolute day-to-day change",
    "weekend_ratio": "Weekend use over weekday use",
    "n_days": "Days of pre-visit history",
    "mean_temp": "Mean outdoor temperature over the period",
    "mean_hdd": "Mean heating degree days",
    "sunshine_mean": "Mean daily sunshine duration",
}

# --------------------------------------------------------------------------
# Look
# --------------------------------------------------------------------------

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
POSITIVE_COLOR = "#e34948"  # fault
NEGATIVE_COLOR = "#2a78d6"  # no fault

ARTIFACT_STEMS = ["dataset", "households", "protocols", "smd_overview", "daily"]
