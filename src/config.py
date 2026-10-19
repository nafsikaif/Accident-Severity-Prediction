"""Global configuration for Accident Severity Prediction pipeline.

Defines directory paths, column whitelists, strict anti-leakage exclusion lists,
target category mappings, model hyperparameters, and UI theme constants.
"""

from pathlib import Path

# =====================================================================
# DIRECTORY & FILE PATHS
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure essential output directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE_PATH = DATA_DIR / "accidents.csv"
MODEL_FILE_PATH = MODELS_DIR / "best_model.joblib"
METRICS_FILE_PATH = REPORTS_DIR / "model_metrics.csv"
CONFUSION_MATRICES_PATH = REPORTS_DIR / "confusion_matrices.joblib"

# =====================================================================
# REPRODUCIBILITY & SPLIT CONFIGURATION
# =====================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20

# =====================================================================
# TARGET DEFINITIONS & MAPPINGS
# =====================================================================
# Potential target column names present in Kaggle / raw variants
TARGET_CANDIDATES = [
    "Accident_severity",
    "accident_severity",
    "Severity",
    "Accident_Severity",
]

# Standardized 3-tier target labels
TARGET_CLASSES = ["Slight", "Serious", "Fatal"]

# Clean mapping from Kaggle raw strings to standardized 3-tier classes
TARGET_MAPPING = {
    "Slight Injury": "Slight",
    "slight injury": "Slight",
    "Slight": "Slight",
    "slight": "Slight",
    "1": "Slight",
    "Serious Injury": "Serious",
    "serious injury": "Serious",
    "Serious": "Serious",
    "serious": "Serious",
    "2": "Serious",
    "Fatal injury": "Fatal",
    "fatal injury": "Fatal",
    "Fatal": "Fatal",
    "fatal": "Fatal",
    "3": "Fatal",
}

# Baseline class distribution for realistic synthetic fallback
# Reflective of empirical road crash statistics (~80% Slight, ~17% Serious, ~3% Fatal)
SYNTHETIC_CLASS_DISTRIBUTION = {
    "Slight": 0.80,
    "Serious": 0.17,
    "Fatal": 0.03,
}

# =====================================================================
# FEATURE WHITELISTING (PHYSICAL PRE-CRASH ONLY)
# =====================================================================
# Categorical features that physically exist and are known BEFORE the collision
CATEGORICAL_FEATURES = [
    "Age_band_of_driver",
    "Sex_of_driver",
    "Driving_experience",
    "Type_of_vehicle",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
    "Type_of_junction",
    "Road_surface_type",
    "Day_of_week",
]

# Continuous or count numerical features known prior to collision
NUMERICAL_FEATURES = [
    "Number_of_vehicles_involved",
]

ALL_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Alternate column naming aliases found across Kaggle dataset distributions
COLUMN_ALIASES = {
    "Types_of_Junction": "Type_of_junction",
    "Type_of_Junction": "Type_of_junction",
    "types_of_junction": "Type_of_junction",
    "type_of_junction": "Type_of_junction",
    "Road_surface_type": "Road_surface_type",
    "road_surface_type": "Road_surface_type",
    "Weather_conditions": "Weather_conditions",
    "weather_conditions": "Weather_conditions",
    "Light_conditions": "Light_conditions",
    "light_conditions": "Light_conditions",
    "Road_surface_conditions": "Road_surface_conditions",
    "road_surface_conditions": "Road_surface_conditions",
    "Type_of_vehicle": "Type_of_vehicle",
    "type_of_vehicle": "Type_of_vehicle",
    "Age_band_of_driver": "Age_band_of_driver",
    "age_band_of_driver": "Age_band_of_driver",
    "Sex_of_driver": "Sex_of_driver",
    "sex_of_driver": "Sex_of_driver",
    "Driving_experience": "Driving_experience",
    "driving_experience": "Driving_experience",
    "Number_of_vehicles_involved": "Number_of_vehicles_involved",
    "number_of_vehicles_involved": "Number_of_vehicles_involved",
    "Day_of_week": "Day_of_week",
    "day_of_week": "Day_of_week",
}

# =====================================================================
# STRICT ANTI-LEAKAGE EXCLUSION LIST
# =====================================================================
# Explicitly drop casualty counts, injury types, post-crash ambulance/police investigations
LEAKAGE_COLUMNS = [
    "Number_of_casualties",
    "Casualty_class",
    "Casualty_severity",
    "Sex_of_casualty",
    "Age_band_of_casualty",
    "Work_of_casuality",
    "Fitness_of_casuality",
    "Pedestrian_movement",
    "Cause_of_accident",
    "Accident_injury_type",
    "Vehicle_movement",
    "Time",  # Unparsed freeform or post-reported string
    "Educational_level",  # Frequently missing or post-incident interview
    "Vehicle_driver_relation",
    "Owner_of_vehicle",
    "Service_year_of_vehicle",
    "Defect_of_vehicle",
    "Area_accident_occured",
    "Lanes_or_Medians",
    "Road_allignment",
    "Type_of_collision",
]

# =====================================================================
# CATEGORY DOMAINS (FOR SYNTHETIC GENERATION & UI INPUTS)
# =====================================================================
FEATURE_CATEGORIES = {
    "Age_band_of_driver": [
        "18-30",
        "31-50",
        "Over 51",
        "Under 18",
        "Unknown",
    ],
    "Sex_of_driver": [
        "Male",
        "Female",
        "Unknown",
    ],
    "Driving_experience": [
        "1-2yr",
        "2-5yr",
        "5-10yr",
        "Above 10yr",
        "Below 1yr",
        "No Licence",
        "Unknown",
    ],
    "Type_of_vehicle": [
        "Automobile",
        "Public (> 30 seats)",
        "Lorry (41?100Q)",
        "Lorry (11?40Q)",
        "Public (12?45 seats)",
        "Pick up upto 10Q",
        "Other",
        "Motorcycle",
        "Taxi",
        "Special vehicle",
    ],
    "Road_surface_conditions": [
        "Dry",
        "Wet or damp",
        "Snow",
        "Flood over 3cm deep",
    ],
    "Light_conditions": [
        "Daylight",
        "Darkness - lights lit",
        "Darkness - no lighting",
        "Darkness - lights unlit",
    ],
    "Weather_conditions": [
        "Normal",
        "Raining",
        "Raining and Windy",
        "Cloudy",
        "Fog or mist",
        "Windy",
        "Snow",
        "Other",
    ],
    "Type_of_junction": [
        "No junction",
        "Y Shape",
        "Crossing",
        "T Shape",
        "O Shape",
        "Other",
        "Unknown",
    ],
    "Road_surface_type": [
        "Asphalt roads",
        "Earth roads",
        "Gravel roads",
        "Asphalt roads with some defect",
        "Other",
    ],
    "Day_of_week": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
}

# =====================================================================
# UI THEME PALETTE CONSTANTS
# =====================================================================
THEME = {
    "bg_main": "#FFFFFF",
    "bg_container": "#F9F9FB",
    "primary_accent": "#B9A7E8",
    "secondary_card": "#EDE7F6",
    "text_dark": "#292536",
    "border_subtle": "#D1C4E9",
    "badge_slight_bg": "#C8E6C9",
    "badge_slight_text": "#1B5E20",
    "badge_serious_bg": "#FFE082",
    "badge_serious_text": "#E65100",
    "badge_fatal_bg": "#FFCDD2",
    "badge_fatal_text": "#B71C1C",
}
