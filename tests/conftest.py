from pathlib import Path
from typing import Dict

import pytest


@pytest.fixture(scope="session")
def texas_paths() -> Dict[str, Path]:
    return {
        "data_file": Path("tests/data/Texas/input_texas.csv"),
        "grouping_file": Path("tests/data/Texas/grouping_texas.csv"),
        "groups_file": Path("tests/data/Texas/groups_texas.txt"),
        "output_path": Path("tests/temp/Texas"),
        "threshold": 60,
    }
