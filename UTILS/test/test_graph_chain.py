import pytest
from omegaconf import OmegaConf
from UTILS.graph_chain import GraphEquation
import os 
print(os.getcwd())

@pytest.fixture
def sample_cfg():
    # Create a minimal DictConfig for testing
    cfg = OmegaConf.load("config.yaml")
    return cfg


@pytest.mark.parametrize("equation,expected", [
    ("a = b + c", ["a", "b", "c"]),
    ("a = 5 + b", ["a", "b"]),
    ("x ^ 2 = y * z", ["x", "y", "z"]),
    ("( a + b ) = c", ["a", "b", "c"]),
    ("a = exp( b )", ["a", "b"]),
    ("+ * =", []),
    ("a   =   b   +   c", ["a", "b", "c"]),
    ("a = -5 + b", ["a", "b"]),
    ("a = -b + c", ["a", "-b", "c"]),
    ("a = sin( x ) + cos( y )", ["a", "x", "y"]),
])
def test_parse_equations(sample_cfg, equation, expected):
    # Test various equations
    equations = [equation]
    ge = GraphEquation(equations, sample_cfg, choice=0)
    result = ge._parse(equation)
    assert result == expected