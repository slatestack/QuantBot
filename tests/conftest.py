import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantbot.config import load_config  # noqa: E402
from quantbot.core.instruments import InstrumentRegistry  # noqa: E402
from quantbot.data_engineering.connectors.synthetic import generate_universe  # noqa: E402


@pytest.fixture(scope="session")
def universe():
    return generate_universe("2016-01-01", "2018-12-31", n_equities=4, commodities=("CL", "GC", "NG"), seed=11)


@pytest.fixture(scope="session")
def registry():
    return InstrumentRegistry.from_yaml(ROOT / "config/instruments/equities.yaml",
                                        ROOT / "config/instruments/commodities_futures.yaml")


@pytest.fixture
def small_cfg():
    return load_config(ROOT / "config/strategy_synthetic.yaml", {
        "universe": {"equities": ["SYN_EQ1", "SYN_EQ2", "SYN_EQ3", "SYN_EQ4", "SYN_PAIR_X", "SYN_PAIR_Y"],
                     "futures": ["CL", "GC", "NG"]},
        "data": {"start": "2016-01-01", "end": "2018-12-31"},
        "instrument_files": [str(ROOT / "config/instruments/equities.yaml"),
                             str(ROOT / "config/instruments/commodities_futures.yaml")],
    })
