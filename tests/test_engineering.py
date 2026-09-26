"""Tests for engineering calculation.

The premise is that the model shouldn't do this arithmetic, so what's checked is
that the tools compute correct answers and, more importantly, that wrong
equations fail loudly instead of returning a plausible number.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pint")
pytest.importorskip("sympy")

from tools.engineering import (  # noqa: E402
    CheckDimensionsTool, ConvertUnitsTool, EngineeringCalculateTool, SolveSymbolicTool,
)


@pytest.mark.asyncio
async def test_calculation_carries_units_through():
    result = await EngineeringCalculateTool().run(
        expression="0.5*rho*v**2*C_d*A",
        variables={"rho": "1.225 kg/m**3", "v": "30 m/s", "C_d": "0.32", "A": "2.2 m**2"},
        convert_to="N",
    )
    assert result.success
    assert result.output["result"]["value"] == pytest.approx(388.08, abs=0.01)
    assert result.output["result"]["units"] == "newton"


@pytest.mark.asyncio
async def test_impossible_conversion_is_an_error_not_a_number():
    """A conversion that isn't dimensionally possible usually means the formula
    is wrong - which is worth catching rather than quietly skipping."""
    result = await EngineeringCalculateTool().run(
        expression="m*a", variables={"m": "10 kg", "a": "9.81 m/s**2"}, convert_to="joule",
    )
    assert result.success is False
    assert "can't" in result.error


@pytest.mark.asyncio
async def test_constants_are_available_without_being_defined():
    result = await EngineeringCalculateTool().run(
        expression="m*g*h", variables={"m": "1200 kg", "h": "15 m"}, convert_to="kJ",
    )
    assert result.output["result"]["value"] == pytest.approx(176.52, abs=0.01)


@pytest.mark.asyncio
async def test_a_users_symbol_is_not_reported_as_a_constant():
    """h is Planck's constant, but here it's the caller's height."""
    result = await EngineeringCalculateTool().run(
        expression="m*g*h", variables={"m": "1 kg", "h": "1 m"},
    )
    assert result.output["constants_used"] == ["g"]


@pytest.mark.asyncio
@pytest.mark.parametrize("expression", [
    "__import__('os').system('id')",
    # Names nothing forbidden and reaches the interpreter anyway, by walking the
    # object graph. This is what an empty __builtins__ does not stop, and the
    # reason the formula is parsed rather than evaluated.
    "[c for c in ().__class__.__base__.__subclasses__()"
    " if c.__name__ == 'BuiltinImporter'][0].load_module('os').getcwd()",
    "().__class__.__base__.__subclasses__()",
    "abs.__globals__",
    "(lambda: 1)()",
    "open('/etc/passwd').read()",
    # Arithmetic, but a hang rather than an answer.
    "2**10**9",
])
async def test_expressions_cannot_reach_the_interpreter(expression):
    result = await EngineeringCalculateTool().run(expression=expression)
    assert result.success is False, expression


@pytest.mark.asyncio
@pytest.mark.parametrize("expression,variables", [
    ("sqrt(a**2 + b**2)", {"a": "3 m", "b": "4 m"}),
    ("atan2(1, 1)", None),
    ("-x + 2*(x/2)", {"x": "5 m"}),
    ("v**2/(2*a)", {"v": "10 m/s", "a": "2 m/s**2"}),
    ("F/A", {"F": "100 N", "A": "0.01 m**2"}),
])
async def test_the_arithmetic_a_formula_is_made_of_still_works(expression, variables):
    """The whitelist has to be wide enough to be a calculator."""
    result = await EngineeringCalculateTool().run(expression=expression, variables=variables)
    assert result.success is True, result.error


@pytest.mark.asyncio
async def test_a_missing_variable_is_reported():
    result = await EngineeringCalculateTool().run(expression="m*a", variables={"m": "1 kg"})
    assert result.success is False


# --- Dimensional checking ---------------------------------------------------------

@pytest.mark.asyncio
async def test_a_correct_equation_checks_out():
    result = await CheckDimensionsTool().run(
        left="F", right="m*a",
        variables={"F": "98.1 N", "m": "10 kg", "a": "9.81 m/s**2"},
    )
    assert result.output["dimensionally_consistent"] is True
    assert result.output["values_match"] is True


@pytest.mark.asyncio
async def test_a_dimensionally_wrong_equation_is_caught():
    result = await CheckDimensionsTool().run(
        left="F", right="m*v",
        variables={"F": "98.1 N", "m": "10 kg", "v": "9.81 m/s"},
    )
    assert result.output["dimensionally_consistent"] is False
    assert "wrong" in result.output["verdict"]


@pytest.mark.asyncio
async def test_right_dimensions_wrong_number_is_distinguished():
    """Consistent units with different values is a different mistake from
    inconsistent units, and worth naming separately."""
    result = await CheckDimensionsTool().run(
        left="F", right="m*a",
        variables={"F": "500 N", "m": "10 kg", "a": "9.81 m/s**2"},
    )
    assert result.output["dimensionally_consistent"] is True
    assert result.output["values_match"] is False


@pytest.mark.asyncio
async def test_units_compare_in_base_units():
    """kN and N are the same quantity; the check must not report otherwise."""
    result = await CheckDimensionsTool().run(
        left="F", right="m*a",
        variables={"F": "0.0981 kN", "m": "10 kg", "a": "9.81 m/s**2"},
    )
    assert result.output["values_match"] is True


# --- Conversion -------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("quantity,units,expected", [
    ("2500 psi", "MPa", 17.2369),
    ("100 km/h", "m/s", 27.7778),
    ("450 degF", "K", 505.3722),
])
async def test_conversions(quantity, units, expected):
    result = await ConvertUnitsTool().run(quantity=quantity, to_units=units)
    assert result.output["to"]["value"] == pytest.approx(expected, rel=1e-4)


@pytest.mark.asyncio
async def test_incompatible_conversion_is_refused():
    result = await ConvertUnitsTool().run(quantity="10 kg", to_units="metre")
    assert result.success is False


# --- Symbolic ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rearranging_a_formula():
    result = await SolveSymbolicTool().run(
        operation="solve", expression="v**2 = u**2 + 2*a*s", symbol="s",
    )
    assert "v**2" in result.output["result"]


@pytest.mark.asyncio
async def test_derivative_and_integral():
    derivative = await SolveSymbolicTool().run(operation="diff", expression="x**3", symbol="x")
    assert derivative.output["result"] == "3*x**2"
    integral = await SolveSymbolicTool().run(operation="integrate", expression="3*x**2", symbol="x")
    assert integral.output["result"] == "x**3"


@pytest.mark.asyncio
async def test_substitution_turns_a_derived_formula_into_a_number():
    result = await SolveSymbolicTool().run(
        operation="solve", expression="v**2 = u**2 + 2*a*s", symbol="s",
        substitutions={"v": "30", "u": "0", "a": "9.81"},
    )
    assert float(result.output["substituted"][0]) == pytest.approx(45.87, abs=0.01)


@pytest.mark.asyncio
async def test_ambiguous_solve_asks_which_symbol():
    result = await SolveSymbolicTool().run(operation="solve", expression="a*x + b")
    assert result.success is False
    assert "which symbol" in result.error


@pytest.mark.asyncio
async def test_unparseable_expression_is_reported():
    result = await SolveSymbolicTool().run(operation="simplify", expression="((((")
    assert result.success is False


@pytest.mark.asyncio
@pytest.mark.parametrize("expression", [
    # sympy's parse_expr and sympify are eval() underneath. Before these were
    # sanitised, the first two of these returned the working directory and the
    # contents of a file respectively.
    "__import__('os').getcwd()",
    "open('/etc/hostname').read()",
    "().__class__.__base__.__subclasses__()",
    "exec('x=1')",
    "x.__class__.__mro__",
])
async def test_symbolic_expressions_cannot_reach_the_interpreter(expression):
    result = await SolveSymbolicTool().run(operation="simplify", expression=expression)
    assert result.success is False, expression


@pytest.mark.asyncio
async def test_a_substitution_cannot_reach_the_interpreter_either():
    result = await SolveSymbolicTool().run(
        operation="solve", expression="a*x + b", symbol="x",
        substitutions={"a": "__import__('os').getcwd()", "b": "1"})
    # The solve itself is fine; it is the substitution that is refused, and it is
    # reported rather than dropped.
    assert result.success is True
    assert "substituted" not in result.output
    assert result.output["substitution_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,expression,symbol", [
    ("solve", "v**2 = u**2 + 2*a*s", "v"),
    ("diff", "sqrt(x) + exp(x) + log(x)", "x"),
    ("integrate", "sin(x)", "x"),
    ("simplify", "(x**2 - 1)/(x - 1)", ""),
    ("factor", "x**2 + 2*x + 1", ""),
])
async def test_real_symbolic_work_still_parses(operation, expression, symbol):
    """The namespace has to be wide enough to do algebra with."""
    result = await SolveSymbolicTool().run(operation=operation, expression=expression,
                                          symbol=symbol)
    assert result.success is True, result.error
