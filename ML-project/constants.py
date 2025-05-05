import os
import platform
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).absolute().parent

# Set up the working directory
try:
    PROJECT_WORKING_DIRECTORY = Path(os.environ['ASNWD'])
except Exception as e:
    print(f"Environment variable ASNWD not set: {e}.\nUsing project root directory")
    PROJECT_WORKING_DIRECTORY = PROJECT_ROOT

# Define data directories
DATA_DIR = PROJECT_ROOT / "Data"
PREPROCESS_DIR = PROJECT_ROOT / "Preprocess"
TRAINING_DIR = PROJECT_ROOT / "Training"
EVALUATION_DIR = PROJECT_ROOT / "Evaluation"

# Create directories if they don't exist
for directory in [DATA_DIR, PREPROCESS_DIR, TRAINING_DIR, EVALUATION_DIR]:
    directory.mkdir(exist_ok=True)

SYSTEM = platform.system()
# Linux: Linux
# Mac: Darwin
# Windows: Windows
LOCAL_DEBUG = os.environ.get("LOCAL_DEBUG")
# ================================================== PLASTICC ==================================== #
PLASTICC_CLASS_MAPPING = {
    90: "SNIa",
    67: "SNIa-91bg",
    52: "SNIax",
    42: "SNII",
    62: "SNIbc",
    95: "SLSN-I",
    15: "TDE",
    64: "KN",
    88: "AGN",
    92: "RRL",
    65: "M-dwarf",
    16: "EB",
    53: "Mira",
    6: "$\mu$-Lens-Single",
}

PLASTICC_WEIGHTS_DICT = {
    6: 1 / 18,
    15: 1 / 9,
    16: 1 / 18,
    42: 1 / 18,
    52: 1 / 18,
    53: 1 / 18,
    62: 1 / 18,
    64: 1 / 9,
    65: 1 / 18,
    67: 1 / 18,
    88: 1 / 18,
    90: 1 / 18,
    92: 1 / 18,
    95: 1 / 18,
    99: 1 / 19,
    1: 1 / 18,
    2: 1 / 18,
    3: 1 / 18,
}

# ===================================================== LSST ==================================== #
LSST_FILTER_MAP = {
    0: "lsstu",
    1: "lsstg",
    2: "lsstr",
    3: "lssti",
    4: "lsstz",
    5: "lssty",
}

# Central passbands wavelengths
LSST_PB_WAVELENGTHS = {
    "lsstu": 3685.0,
    "lsstg": 4802.0,
    "lsstr": 6231.0,
    "lssti": 7542.0,
    "lsstz": 8690.0,
    "lssty": 9736.0,
}

LSST_PB_COLORS = {
    "lsstu": "#984ea3",  # Purple: https://www.color-hex.com/color/984ea3
    "lsstg": "#4daf4a",  # Green: https://www.color-hex.com/color/4daf4a
    "lsstr": "#e41a1c",  # Red: https://www.color-hex.com/color/e41a1c
    "lssti": "#377eb8",  # Blue: https://www.color-hex.com/color/377eb8
    "lsstz": "#ff7f00",  # Orange: https://www.color-hex.com/color/ff7f00
    "lssty": "#e3c530",  # Yellow: https://www.color-hex.com/color/e3c530
}


# ===================================================== ZTF ==================================== #
ZTF_FILTER_MAP = {1: "ztfg", 2: "ztfr", 3: "ztfi"}

ZTF_FILTER_MAP_COLORS = {
    1: "#4daf4a",  # Green: https://www.color-hex.com/color/4daf4a
    2: "#e41a1c",  # Red: https://www.color-hex.com/color/e41a1c
    3: "#377eb8",  # Blue: https://www.color-hex.com/color/377eb8
}

# Central passbands wavelengths --> λmean
# http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=Palomar&gname2=ZTF&asttype=
ZTF_PB_WAVELENGTHS = {
    "ztfg": 4804.79,
    "ztfr": 6436.92,
    "ztfi": 7968.22,
}

ZTF_PB_COLORS = {
    "ztfg": "#4daf4a",  # Green: https://www.color-hex.com/color/4daf4a
    "ztfr": "#e41a1c",  # Red: https://www.color-hex.com/color/e41a1c
    "ztfi": "#377eb8",  # Blue: https://www.color-hex.com/color/377eb8
}
