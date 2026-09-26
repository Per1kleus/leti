"""Engineering calculation: units, symbolic algebra, and numerical work.

The premise is that a language model should not be doing engineering arithmetic
in its head. Every number here comes from pint (units and dimensions) or sympy
(algebra, calculus, equation solving), and anything beyond their reach runs as
real code through run_code - which is why this module has no interpreter of its
own and no second way to execute Python or MATLAB.

Units are the point rather than a convenience. A calculation carrying its units
is a calculation that can be checked: pint refuses to add a length to a time,
so dimensional errors surface as errors instead of as plausible wrong numbers.
check_dimensions exists so an equation can be checked before it's trusted, which
is the mechanical half of "find my mistake".

The INPUTS -> ASSUMPTIONS -> EQUATIONS -> UNITS -> CALCULATIONS -> RESULTS ->
CHECKS presentation lives in the system prompt, not here: it's how results are
written up, and the tools supply the parts it's written from. What the tools
guarantee is that the numbers in it were computed rather than recalled.
"""
from __future__ import annotations

import ast
import asyncio
import math
import operator
import re
from typing import Any, Dict, Optional

from tools.base import BaseTool, ToolParameter, ToolResult

# Constants an engineering calculation reaches for often enough that looking
# them up should not be a separate step - and where a misremembered value is a
# silently wrong answer. Values are CODATA/ISO standard.
CONSTANTS: Dict[str, Dict[str, str]] = {
    "g": {"value": "9.80665 m/s**2", "name": "Standard gravity"},
    "R": {"value": "8.314462618 J/(mol*K)", "name": "Universal gas constant"},
    "sigma": {"value": "5.670374419e-8 W/(m**2*K**4)", "name": "Stefan-Boltzmann constant"},
    "k_B": {"value": "1.380649e-23 J/K", "name": "Boltzmann constant"},
    "N_A": {"value": "6.02214076e23 1/mol", "name": "Avogadro constant"},
    "c": {"value": "299792458 m/s", "name": "Speed of light in vacuum"},
    "h": {"value": "6.62607015e-34 J*s", "name": "Planck constant"},
    "atm": {"value": "101325 Pa", "name": "Standard atmosphere"},
    "R_air": {"value": "287.052874 J/(kg*K)", "name": "Specific gas constant, dry air"},
    "gamma_air": {"value": "1.4 dimensionless", "name": "Ratio of specific heats, air"},
}

_UNIT_REGISTRY = None


def unit_registry():
    """One pint registry for the process - pint quantities from different
    registries can't be combined, which shows up as confusing type errors."""
    global _UNIT_REGISTRY
    if _UNIT_REGISTRY is None:
        import pint

        _UNIT_REGISTRY = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
    return _UNIT_REGISTRY


def parse_quantity(text: str):
    """'9.81 m/s^2' or '300 K' into a pint quantity carrying its units."""
    registry = unit_registry()
    cleaned = str(text).strip().replace("^", "**")
    if cleaned in CONSTANTS:
        cleaned = CONSTANTS[cleaned]["value"]
    return registry.Quantity(cleaned)


# --- Evaluating a formula ------------------------------------------------------------
#
# This used to be eval() with {"__builtins__": {}}, on the belief that an empty
# builtins dict is a sandbox. It is not: an expression that never names anything
# forbidden can still walk to the interpreter through the object graph, e.g.
#
#     [c for c in ().__class__.__base__.__subclasses__()
#      if c.__name__ == "BuiltinImporter"][0].load_module("os")
#
# and this tool's action class is `execute`, so it does not stop to ask. The
# expression comes from the model, and the model reads web pages, so "the model
# would not write that" is not a control.
#
# So the formula is parsed and walked instead of evaluated. Only the node types a
# formula is made of are allowed - numbers, names, arithmetic, and calls to the
# functions the namespace actually provides. Attribute access, subscripting,
# comprehensions, lambdas and everything else is refused by default, which means a
# new Python syntax cannot quietly become a new way through.

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# An expression is arithmetic, so a very large exponent is a hang rather than an
# answer: 2**10**9 is not a calculation anyone asked for.
MAX_EXPONENT = 1_000_000


class FormulaError(Exception):
    """The expression is not arithmetic. The message says which part of it."""


def evaluate_formula(expression: str, namespace: Dict[str, Any]) -> Any:
    """Work out `expression` using only `namespace` and arithmetic.

    Raises FormulaError for anything that is not a formula, and whatever the
    arithmetic itself raises (pint's dimensionality errors, ZeroDivisionError)
    for one that is.
    """
    text = str(expression).replace("^", "**")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"that is not a complete expression ({e.msg})") from e
    return _eval_node(tree.body, namespace)


def _eval_node(node: ast.AST, namespace: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
            return node.value
        raise FormulaError(f"{node.value!r} is not a number")

    if isinstance(node, ast.Name):
        if node.id in namespace:
            return namespace[node.id]
        raise FormulaError(f"{node.id!r} has no value - pass it in variables")

    if isinstance(node, ast.BinOp):
        handler = _BINARY.get(type(node.op))
        if handler is None:
            raise FormulaError(f"{type(node.op).__name__} is not an arithmetic operator")
        left = _eval_node(node.left, namespace)
        right = _eval_node(node.right, namespace)
        if handler is operator.pow and isinstance(right, (int, float)) and abs(right) > MAX_EXPONENT:
            raise FormulaError(f"an exponent of {right} is too large to work out")
        return handler(left, right)

    if isinstance(node, ast.UnaryOp):
        handler = _UNARY.get(type(node.op))
        if handler is None:
            raise FormulaError(f"{type(node.op).__name__} is not an arithmetic operator")
        return handler(_eval_node(node.operand, namespace))

    if isinstance(node, ast.Call):
        # Only a bare name, and only one the namespace supplies: no attribute
        # lookups, no calling the result of another call.
        if not isinstance(node.func, ast.Name):
            raise FormulaError("only the named functions can be called")
        function = namespace.get(node.func.id)
        if function is None or not callable(function):
            raise FormulaError(f"there is no function called {ast.unparse(node.func)!r}")
        if node.keywords:
            raise FormulaError(f"{node.func.id}() takes its arguments in order, not by name")
        return function(*[_eval_node(a, namespace) for a in node.args])

    raise FormulaError(f"{_describe(node)} is not part of a formula")


# The refusal goes back to the model, which is the thing that wrote the
# expression, so it names the construct rather than the AST class.
_CONSTRUCTS = {
    ast.Attribute: "reading an attribute with .", ast.Subscript: "indexing with []",
    ast.ListComp: "a list comprehension", ast.GeneratorExp: "a generator expression",
    ast.DictComp: "a dict comprehension", ast.SetComp: "a set comprehension",
    ast.Lambda: "a lambda", ast.IfExp: "an if/else expression",
    ast.List: "a list", ast.Dict: "a dict", ast.Set: "a set", ast.Tuple: "a tuple",
    ast.Compare: "a comparison", ast.BoolOp: "and/or", ast.Starred: "unpacking with *",
    ast.NamedExpr: "an assignment with :=", ast.Await: "await", ast.JoinedStr: "an f-string",
}


def _describe(node: ast.AST) -> str:
    return _CONSTRUCTS.get(type(node), type(node).__name__)


def _format_quantity(quantity) -> Dict[str, Any]:
    magnitude = quantity.magnitude
    return {
        "value": float(magnitude) if isinstance(magnitude, (int, float)) else str(magnitude),
        "units": str(quantity.units),
        "dimensionality": str(quantity.dimensionality) or "dimensionless",
        "formatted": f"{quantity:~P}",
    }


# --- Reading a symbolic expression ---------------------------------------------------
#
# sympy's parse_expr and sympify are eval() underneath, and sympy's own
# documentation says so: "sympify uses eval, and should not be used on
# unsanitised input". Both were being handed the model's string directly, and
# solve_symbolic is `execute`, so it does not stop to ask. Measured before the
# fix: parse_expr("__import__('os').getcwd()") returned the working directory and
# parse_expr("open('/etc/hostname').read()") returned the file.
#
# Two things close it, and both are needed. The expression is walked first, so
# there is no attribute access and no subscripting to traverse the object graph
# with. Then it is parsed against a namespace that holds the sympy names a
# symbolic expression legitimately uses and an empty __builtins__, so `open` and
# `__import__` are not resolvable - without that, eval() supplies real builtins
# whatever the namespace says. An unknown name becomes a Symbol, as it always
# did, which is what lets "m*a" mean what it looks like.

# Nodes a symbolic expression is made of. Attribute, Subscript, Lambda, the
# comprehensions and everything else not named here is refused.
_SYMBOLIC_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Call, ast.keyword,
    ast.BinOp, ast.UnaryOp, ast.Tuple, ast.List, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)

# Built once and shared: importing sympy is slow and this does not change.
_SYMBOLIC_NAMESPACE: Optional[Dict[str, Any]] = None


def _symbolic_namespace() -> Dict[str, Any]:
    global _SYMBOLIC_NAMESPACE
    if _SYMBOLIC_NAMESPACE is None:
        import sympy

        allowed = (
            "Symbol symbols Eq Ne Lt Le Gt Ge Rational Integer Float Matrix "
            "sqrt cbrt exp log ln sin cos tan asin acos atan atan2 sinh cosh tanh "
            "asinh acosh atanh Abs sign floor ceiling factorial binomial gamma "
            "pi E I oo nan Sum Product Integral Derivative Limit Function "
            "simplify expand factor together apart cancel diff integrate limit "
            "solve nsolve re im conjugate arg Min Max root Pow Mul Add"
        ).split()
        namespace = {name: getattr(sympy, name) for name in allowed if hasattr(sympy, name)}
        # eval() fills __builtins__ in itself when the globals it is given have no
        # entry for it, so this is the line that keeps open() and __import__ out.
        namespace["__builtins__"] = {}
        _SYMBOLIC_NAMESPACE = namespace
    return _SYMBOLIC_NAMESPACE


def parse_symbolic(text: str):
    """Read `text` as a symbolic expression. Raises FormulaError if it is not one."""
    from sympy.parsing.sympy_parser import parse_expr

    source = str(text).replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"that is not a complete expression ({e.msg})") from e
    for node in ast.walk(tree):
        if not isinstance(node, _SYMBOLIC_NODES):
            raise FormulaError(f"{_describe(node)} is not part of an expression")
    try:
        return parse_expr(source, global_dict=_symbolic_namespace())
    except FormulaError:
        raise
    except Exception as e:
        raise FormulaError(str(e)) from e


class EngineeringCalculateTool(BaseTool):
    name = "engineering_calculate"
    description = (
        "Evaluate an engineering expression with units, so the arithmetic and the unit "
        "algebra are both done properly rather than in your head. Give the expression and "
        "the values of its symbols with units: expression '0.5*rho*v**2*C_d*A', variables "
        "{'rho': '1.225 kg/m**3', 'v': '30 m/s', 'C_d': '0.32', 'A': '2.2 m**2'}.\n"
        "Named constants g, R, sigma, k_B, N_A, c, h, atm, R_air and gamma_air are available "
        "without defining them. Set `convert_to` for the units you want the answer in - a "
        "conversion that isn't dimensionally possible is reported as an error, which is how "
        "a wrong formula gets caught.\n"
        "Use this for every number you report. Do not compute engineering results yourself."
    )
    parameters = [
        ToolParameter(name="expression", type="string",
                      description="Expression to evaluate, e.g. '0.5*rho*v**2*C_d*A'."),
        ToolParameter(name="variables", type="object", required=False,
                      description="Symbol -> value with units, e.g. {'v': '30 m/s', 'A': '2.2 m**2'}."),
        ToolParameter(name="convert_to", type="string", required=False,
                      description="Units for the result, e.g. 'kN' or 'kW'."),
        ToolParameter(name="description", type="string", required=False,
                      description="What is being calculated, for the record."),
    ]

    async def run(self, expression: str, variables: Optional[Dict[str, Any]] = None,
                  convert_to: str = "", description: str = "", **kwargs) -> ToolResult:
        registry = unit_registry()
        resolved: Dict[str, Any] = {}
        inputs: Dict[str, Any] = {}

        for name, raw in (variables or {}).items():
            try:
                quantity = parse_quantity(raw)
            except Exception as e:
                return ToolResult(success=False, error=f"Couldn't read {name}={raw!r}: {e}")
            resolved[name] = quantity
            inputs[name] = _format_quantity(quantity)

        # Constants are available unless the caller defined that name themselves.
        for name, spec in CONSTANTS.items():
            if name not in resolved:
                resolved[name] = parse_quantity(spec["value"])

        namespace = {
            **resolved,
            "pi": math.pi, "e": math.e,
            "sqrt": lambda q: q ** 0.5,
            "sin": lambda q: math.sin(q.to("rad").magnitude if hasattr(q, "to") else q),
            "cos": lambda q: math.cos(q.to("rad").magnitude if hasattr(q, "to") else q),
            "tan": lambda q: math.tan(q.to("rad").magnitude if hasattr(q, "to") else q),
            "log": math.log, "log10": math.log10, "exp": math.exp, "abs": abs,
            "atan2": math.atan2, "asin": math.asin, "acos": math.acos, "atan": math.atan,
        }

        try:
            result = evaluate_formula(expression, namespace)
        except Exception as e:
            return ToolResult(success=False, error=(
                f"Couldn't evaluate {expression!r}: {e}. Check that every symbol has a value "
                f"and that the units are consistent."
            ))

        if not hasattr(result, "units"):
            result = registry.Quantity(result, "dimensionless")

        converted = None
        if convert_to:
            try:
                converted = result.to(convert_to.replace("^", "**"))
            except Exception as e:
                # A conversion that isn't dimensionally possible usually means the
                # formula is wrong, so it's reported rather than silently skipped.
                return ToolResult(success=False, error=(
                    f"The result is {result.units} ({result.dimensionality}), which can't "
                    f"convert to {convert_to}: {e}. Either the target units or the formula "
                    f"is wrong."
                ), output={"result": _format_quantity(result), "inputs": inputs})

        final = converted if converted is not None else result
        return ToolResult(success=True, output={
            "description": description,
            "expression": expression,
            "inputs": inputs,
            # Word boundaries, and only names the caller didn't define: "m*g*h"
            # uses standard gravity, but h there is the caller's height, not
            # Planck's constant, and a substring match claims otherwise.
            "constants_used": sorted(
                name for name in CONSTANTS
                if name not in (variables or {})
                and re.search(rf"\b{re.escape(name)}\b", str(expression))
            ),
            "result": _format_quantity(final),
            "result_in_si": _format_quantity(result.to_base_units()),
        })


class ConvertUnitsTool(BaseTool):
    name = "convert_units"
    description = (
        "Convert a quantity between units, including compound and imperial ones: "
        "'2500 psi' to 'MPa', '1.2 lbf*ft' to 'N*m', '450 degF' to 'K'. A conversion "
        "between incompatible dimensions is an error, not a number."
    )
    parameters = [
        ToolParameter(name="quantity", type="string", description="Value with units, e.g. '2500 psi'."),
        ToolParameter(name="to_units", type="string", description="Target units, e.g. 'MPa'."),
    ]

    async def run(self, quantity: str, to_units: str, **kwargs) -> ToolResult:
        try:
            source = parse_quantity(quantity)
        except Exception as e:
            return ToolResult(success=False, error=f"Couldn't read {quantity!r}: {e}")
        try:
            converted = source.to(to_units.replace("^", "**"))
        except Exception as e:
            return ToolResult(success=False, error=(
                f"{source.units} ({source.dimensionality}) can't convert to {to_units}: {e}"
            ))
        return ToolResult(success=True, output={
            "from": _format_quantity(source),
            "to": _format_quantity(converted),
            "si": _format_quantity(source.to_base_units()),
        })


class CheckDimensionsTool(BaseTool):
    name = "check_dimensions"
    description = (
        "Check that an equation is dimensionally consistent - that both sides reduce to the "
        "same dimensions. This is the mechanical half of 'find my mistake': an equation that "
        "fails here is wrong regardless of how the numbers look. Give both sides with units "
        "for each symbol, e.g. left 'F', right 'm*a', variables {'F':'100 N','m':'10 kg','a':'9.81 m/s**2'}."
    )
    parameters = [
        ToolParameter(name="left", type="string", description="Left-hand side expression."),
        ToolParameter(name="right", type="string", description="Right-hand side expression."),
        ToolParameter(name="variables", type="object",
                      description="Symbol -> value with units for every symbol used."),
    ]

    async def run(self, left: str, right: str, variables: Optional[Dict[str, Any]] = None,
                  **kwargs) -> ToolResult:
        calculator = EngineeringCalculateTool()
        sides = {}
        for label, expression in (("left", left), ("right", right)):
            result = await calculator.run(expression=expression, variables=variables)
            if not result.success:
                return ToolResult(success=False, error=f"{label} side: {result.error}")
            sides[label] = result.output["result"]

        left_dim, right_dim = sides["left"]["dimensionality"], sides["right"]["dimensionality"]
        consistent = left_dim == right_dim

        # Both sides in base units, so 'kN' and 'N' compare as equal magnitudes.
        values_match = None
        if consistent:
            left_si = parse_quantity(f"{sides['left']['value']} {sides['left']['units']}").to_base_units()
            right_si = parse_quantity(f"{sides['right']['value']} {sides['right']['units']}").to_base_units()
            values_match = math.isclose(
                float(left_si.magnitude), float(right_si.magnitude), rel_tol=1e-6, abs_tol=1e-12
            )

        return ToolResult(success=True, output={
            "equation": f"{left} = {right}",
            "left": sides["left"],
            "right": sides["right"],
            "dimensionally_consistent": consistent,
            "values_match": values_match,
            "verdict": (
                "Dimensions disagree - the equation is wrong." if not consistent
                else "Dimensions agree and both sides evaluate to the same value." if values_match
                else "Dimensions agree but the two sides evaluate differently - check the numbers."
            ),
        })


class SolveSymbolicTool(BaseTool):
    name = "solve_symbolic"
    description = (
        "Symbolic mathematics with sympy: solve equations, rearrange a formula for a "
        "different variable, differentiate, integrate, simplify, or expand. Use this to "
        "derive a relationship before putting numbers in, and to rearrange rather than "
        "doing algebra by hand.\n"
        "Operations: solve (for a symbol), diff, integrate, simplify, expand, factor, limit. "
        "Write the expression in Python/sympy syntax: x**2, sqrt(x), sin(x), exp(x)."
    )
    parameters = [
        ToolParameter(name="operation", type="string",
                      enum=["solve", "diff", "integrate", "simplify", "expand", "factor", "limit"],
                      description="What to do."),
        ToolParameter(name="expression", type="string",
                      description="The expression, or the equation as 'left = right' for solve."),
        ToolParameter(name="symbol", type="string", required=False,
                      description="Symbol to solve for, differentiate by, or integrate over."),
        ToolParameter(name="at", type="string", required=False,
                      description="For limit: the value the symbol approaches, e.g. '0' or 'oo'."),
        ToolParameter(name="substitutions", type="object", required=False,
                      description="Values to substitute into the result, e.g. {'x': '2'}."),
    ]

    async def run(self, operation: str, expression: str, symbol: str = "", at: str = "",
                  substitutions: Optional[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._solve, operation, expression, symbol, at, substitutions or {}
        )

    def _solve(self, operation, expression, symbol, at, substitutions) -> ToolResult:
        import sympy

        try:
            if "=" in expression and operation == "solve":
                left, _, right = expression.partition("=")
                parsed = sympy.Eq(parse_symbolic(left.strip()), parse_symbolic(right.strip()))
            else:
                parsed = parse_symbolic(expression)
        except Exception as e:
            return ToolResult(success=False, error=f"Couldn't parse {expression!r}: {e}")

        try:
            target = sympy.Symbol(symbol) if symbol else None
            if operation == "solve":
                if target is None:
                    free = sorted(parsed.free_symbols, key=str)
                    if len(free) != 1:
                        return ToolResult(success=False, error=(
                            f"Say which symbol to solve for - the expression has {[str(s) for s in free]}."
                        ))
                    target = free[0]
                result = sympy.solve(parsed, target)
            elif operation == "diff":
                result = sympy.diff(parsed, target) if target else sympy.diff(parsed)
            elif operation == "integrate":
                result = sympy.integrate(parsed, target) if target else sympy.integrate(parsed)
            elif operation == "simplify":
                result = sympy.simplify(parsed)
            elif operation == "expand":
                result = sympy.expand(parsed)
            elif operation == "factor":
                result = sympy.factor(parsed)
            elif operation == "limit":
                if target is None:
                    return ToolResult(success=False, error="limit needs a symbol.")
                result = sympy.limit(parsed, target, parse_symbolic(at or 0))
            else:
                return ToolResult(success=False, error=f"Unknown operation '{operation}'.")
        except Exception as e:
            return ToolResult(success=False, error=f"{operation} failed: {e}")

        output: Dict[str, Any] = {
            "operation": operation,
            "input": str(parsed),
            "result": str(result),
            "latex": sympy.latex(result),
        }

        if substitutions:
            try:
                mapping = {sympy.Symbol(k): parse_symbolic(str(v))
                           for k, v in substitutions.items()}
                items = result if isinstance(result, (list, tuple)) else [result]
                output["substituted"] = [
                    str(sympy.N(item.subs(mapping))) if hasattr(item, "subs") else str(item)
                    for item in items
                ]
            except Exception as e:
                output["substitution_error"] = str(e)

        return ToolResult(success=True, output=output)
