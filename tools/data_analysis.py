"""Loading, inspecting, cleaning, analysing and plotting datasets.

Every tool here takes a `path` and loads it the same way, through load_dataset,
so adding a format is one entry in LOADERS rather than a new tool: CSV, TSV,
Excel, JSON, JSON Lines, Parquet, MATLAB .mat and delimited text all arrive as
one pandas DataFrame and everything downstream is format-blind.

What the tools deliberately don't do is decide for the user. inspect_dataset
reports what's there - types, missing values, duplicates, outlier counts - and
names the problems; clean_dataset only performs the operations it's asked for
and reports exactly what each one changed. Dropping rows or filling values
changes what the numbers mean, so that stays an explicit instruction rather
than something that happens quietly on load.

Results land in the active project's folder when there is one (see
tools/projects.py), so a cleaned file or a chart belongs to the work it came
from instead of somewhere the user has to be told about.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config_loader import resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult
from tools.projects import project_dir, resolve_project

MAX_PREVIEW_ROWS = 10
# Datasets are read into memory; past this it's worth telling the user rather
# than quietly taking the machine's RAM with it.
LARGE_FILE_BYTES = 500 * 1024 * 1024


def _read_matlab(path: Path, **kwargs):
    """MATLAB .mat files hold named arrays, not a table.

    The largest 2-D numeric array is the one that behaves like a dataset; the
    rest are usually parameters and metadata. Which one was picked is reported,
    because guessing wrong silently would be worse than guessing wrong loudly.
    """
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat

    raw = loadmat(str(path), squeeze_me=True)
    candidates = {
        key: np.atleast_2d(value)
        for key, value in raw.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.size
        and np.issubdtype(value.dtype, np.number)
    }
    if not candidates:
        raise ValueError(
            f"No numeric arrays in {path.name}. Variables present: "
            f"{[k for k in raw if not k.startswith('__')]}"
        )
    name, array = max(candidates.items(), key=lambda kv: kv[1].size)
    if array.shape[0] == 1 and array.shape[1] > 1:
        array = array.T
    frame = pd.DataFrame(array, columns=[f"{name}_{i + 1}" for i in range(array.shape[1])])
    frame.attrs["matlab_variable"] = name
    frame.attrs["matlab_other_variables"] = sorted(k for k in candidates if k != name)
    return frame


LOADERS = {
    ".csv": lambda p, **kw: __import__("pandas").read_csv(p, **kw),
    ".tsv": lambda p, **kw: __import__("pandas").read_csv(p, sep="\t", **kw),
    ".txt": lambda p, **kw: __import__("pandas").read_csv(p, sep=None, engine="python", **kw),
    ".json": lambda p, **kw: __import__("pandas").read_json(p, **kw),
    ".jsonl": lambda p, **kw: __import__("pandas").read_json(p, lines=True, **kw),
    ".ndjson": lambda p, **kw: __import__("pandas").read_json(p, lines=True, **kw),
    ".parquet": lambda p, **kw: __import__("pandas").read_parquet(p, **kw),
    ".xlsx": lambda p, **kw: __import__("pandas").read_excel(p, **kw),
    ".xlsm": lambda p, **kw: __import__("pandas").read_excel(p, **kw),
    ".xls": lambda p, **kw: __import__("pandas").read_excel(p, **kw),
    ".mat": _read_matlab,
}


def dataset_path(path: str, project: Optional[str] = None) -> Path:
    """Resolve a dataset path, looking inside the active project first.

    "analyse turbine.csv" should find the file in the project the user is working
    in without them spelling out the folder.
    """
    target = resolve_project(project)
    if target and not Path(path).is_absolute():
        try:
            candidate = project_dir(target) / path
            if candidate.is_file():
                return candidate
        except (ValueError, KeyError):
            pass
    return resolve_path(path)


def output_dir(project: Optional[str] = None) -> Path:
    """Where generated files go: the active project, else ./data/analysis."""
    target = resolve_project(project)
    if target:
        try:
            directory = project_dir(target)
            if directory.is_dir():
                return directory
        except (ValueError, KeyError):
            pass
    directory = resolve_path("./data/analysis")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_dataset(path: str, project: Optional[str] = None, sheet: Optional[str] = None, **kwargs):
    """One loader for every format, so every tool below is format-blind."""
    resolved = dataset_path(path, project)
    if not resolved.is_file():
        raise FileNotFoundError(f"No such dataset: {resolved}")

    loader = LOADERS.get(resolved.suffix.lower())
    if loader is None:
        raise ValueError(
            f"Don't know how to read '{resolved.suffix}'. Supported: "
            f"{', '.join(sorted(LOADERS))}."
        )
    if sheet is not None and resolved.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        kwargs["sheet_name"] = sheet

    size = resolved.stat().st_size
    frame = loader(resolved, **kwargs)
    frame.attrs["source_path"] = str(resolved)
    frame.attrs["source_bytes"] = size
    frame.attrs["large_file"] = size > LARGE_FILE_BYTES
    return frame


def _json_safe(value: Any) -> Any:
    """Tool results are JSON-serialised. NaN, numpy scalars and timestamps aren't."""
    import numpy as np
    import pandas as pd

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        # json.dumps emits bare NaN/Infinity, which isn't valid JSON and which
        # some parsers reject outright.
        return None if (math.isnan(number) or math.isinf(number)) else round(number, 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [_json_safe(v) for v in list(value)]
    if pd.isna(value) if not hasattr(value, "__len__") else False:
        return None
    return str(value)


# Characters that make a numeric column read as text: thousands separators,
# currency symbols, stray percent signs. Shared by inspect (to notice) and clean
# (to fix), so the two can't disagree about what counts as convertible.
NUMERIC_NOISE = r"[,\s$€£%]"


def coerce_numeric(series):
    """A text series as numbers, ignoring formatting noise."""
    import pandas as pd

    return pd.to_numeric(
        series.astype(str).str.replace(NUMERIC_NOISE, "", regex=True).str.strip(),
        errors="coerce",
    )


def _outlier_summary(series) -> Optional[Dict[str, Any]]:
    """Outliers by the IQR rule - the conventional definition, stated so the
    reader knows which one produced these counts."""
    import pandas as pd

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 4:
        return None
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return {"count": 0, "rule": "IQR", "lower_bound": _json_safe(q1), "upper_bound": _json_safe(q3)}
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (numeric < low) | (numeric > high)
    return {
        "count": int(mask.sum()),
        "rule": "IQR (outside Q1-1.5*IQR .. Q3+1.5*IQR)",
        "lower_bound": _json_safe(low),
        "upper_bound": _json_safe(high),
        "examples": _json_safe(numeric[mask].head(5).tolist()),
    }


class InspectDatasetTool(BaseTool):
    name = "inspect_dataset"
    description = (
        "Load a dataset and describe it: shape, column types, missing values, duplicate "
        "rows, outliers, and per-column statistics. Supports CSV, TSV, TXT, Excel, JSON, "
        "JSON Lines, Parquet and MATLAB .mat.\n"
        "Do this first, before analysing or cleaning anything - the problems it names "
        "(what's missing, what's duplicated, which columns aren't the type they look like) "
        "are what decides which analysis is appropriate."
    )
    parameters = [
        ToolParameter(name="path", type="string", description="Path to the dataset."),
        ToolParameter(name="project", type="string", required=False, description="Project to look in. Defaults to the active one."),
        ToolParameter(name="sheet", type="string", required=False, description="Worksheet name, for Excel files."),
    ]

    async def run(self, path: str, project: str = "", sheet: str = "", **kwargs) -> ToolResult:
        try:
            frame = await asyncio.get_running_loop().run_in_executor(
                None, lambda: load_dataset(path, project, sheet or None)
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        import pandas as pd

        columns = []
        for name in frame.columns:
            series = frame[name]
            info: Dict[str, Any] = {
                "name": str(name),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean()) * 100, 2),
                "unique": int(series.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(series):
                described = series.describe()
                info["stats"] = {k: _json_safe(v) for k, v in described.items() if k != "count"}
                outliers = _outlier_summary(series)
                if outliers:
                    info["outliers"] = outliers
            else:
                info["most_common"] = _json_safe(series.value_counts().head(5).to_dict())
                # A text column whose values all parse as numbers is usually a
                # numeric column that a stray value or a thousands separator
                # turned into text - worth naming, since it silently breaks stats.
                coerced = coerce_numeric(series)
                if series.notna().any() and coerced.notna().mean() > 0.9:
                    info["looks_numeric_but_is_text"] = True
                    info["convert_with"] = f"clean_dataset operation 'convert_numeric:{name}'"
            columns.append(info)

        duplicates = int(frame.duplicated().sum())
        issues = []
        if duplicates:
            issues.append(f"{duplicates} duplicate row(s).")
        for column in columns:
            if column["missing_percent"] > 0:
                issues.append(f"'{column['name']}' is {column['missing_percent']}% missing.")
            if column.get("looks_numeric_but_is_text"):
                issues.append(f"'{column['name']}' is stored as text but reads as numeric.")
            if column.get("outliers", {}).get("count"):
                issues.append(f"'{column['name']}' has {column['outliers']['count']} outlier(s) by the IQR rule.")

        return ToolResult(success=True, output={
            "path": frame.attrs.get("source_path"),
            "rows": int(len(frame)),
            "columns": len(frame.columns),
            "duplicate_rows": duplicates,
            "memory_mb": round(frame.memory_usage(deep=True).sum() / 1e6, 2),
            "column_details": columns,
            "preview": _json_safe(frame.head(MAX_PREVIEW_ROWS).to_dict(orient="records")),
            "issues": issues or ["No missing values, duplicates or outliers detected."],
            "matlab_variable": frame.attrs.get("matlab_variable"),
            "note": (
                "Cleaning changes what the numbers mean, so nothing here has been "
                "changed - call clean_dataset with the specific operations you want."
            ),
        })


class CleanDatasetTool(BaseTool):
    name = "clean_dataset"
    description = (
        "Clean a dataset with the operations you name, and write the result to a new file - "
        "the original is never modified. Operations: drop_duplicates, drop_missing_rows, "
        "fill_missing (mean/median/mode/zero/a literal value), convert_numeric (text columns "
        "that hold numbers), strip_whitespace, drop_columns, rename_columns, remove_outliers.\n"
        "Every operation reports what it actually changed. Say what you did and what it cost - "
        "'filled 12 missing values with the median' is a different dataset from the one the "
        "user gave you, and conclusions drawn afterwards depend on it."
    )
    parameters = [
        ToolParameter(name="path", type="string", description="Dataset to clean."),
        ToolParameter(
            name="operations", type="array", items_type="string",
            description=(
                "Operations in order. Each is 'name' or 'name:argument', e.g. "
                "'drop_duplicates', 'fill_missing:median', 'drop_columns:notes,temp', "
                "'remove_outliers:power', 'convert_numeric:price'."
            ),
        ),
        ToolParameter(name="output_path", type="string", required=False,
                      description="Where to write the cleaned file. Defaults to <name>_cleaned.<ext>."),
        ToolParameter(name="project", type="string", required=False, description="Project to work in."),
        ToolParameter(name="sheet", type="string", required=False, description="Worksheet name, for Excel."),
    ]

    async def run(self, path: str, operations: List[str], output_path: str = "",
                  project: str = "", sheet: str = "", **kwargs) -> ToolResult:
        try:
            frame = await asyncio.get_running_loop().run_in_executor(
                None, lambda: load_dataset(path, project, sheet or None)
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        import pandas as pd

        before_rows, before_columns = len(frame), list(frame.columns)
        applied: List[Dict[str, Any]] = []

        for raw in operations or []:
            name, _, argument = str(raw).partition(":")
            name, argument = name.strip().lower(), argument.strip()
            rows_before, missing_before = len(frame), int(frame.isna().sum().sum())

            try:
                if name == "drop_duplicates":
                    frame = frame.drop_duplicates()
                    applied.append({"operation": name, "rows_removed": rows_before - len(frame)})

                elif name == "drop_missing_rows":
                    subset = [c.strip() for c in argument.split(",") if c.strip()] or None
                    frame = frame.dropna(subset=subset)
                    applied.append({"operation": name, "columns": subset or "any",
                                    "rows_removed": rows_before - len(frame)})

                elif name == "fill_missing":
                    strategy = argument or "median"
                    filled = 0
                    for column in frame.columns:
                        gaps = int(frame[column].isna().sum())
                        if not gaps:
                            continue
                        numeric = pd.api.types.is_numeric_dtype(frame[column])
                        if strategy == "mean" and numeric:
                            value = frame[column].mean()
                        elif strategy == "median" and numeric:
                            value = frame[column].median()
                        elif strategy == "mode":
                            modes = frame[column].mode()
                            value = modes.iloc[0] if len(modes) else None
                        elif strategy == "zero":
                            value = 0 if numeric else ""
                        else:
                            value = argument
                        if value is None:
                            continue
                        frame[column] = frame[column].fillna(value)
                        filled += gaps
                    applied.append({"operation": name, "strategy": strategy, "values_filled": filled})

                elif name == "convert_numeric":
                    targets = ([c.strip() for c in argument.split(",") if c.strip()]
                               or [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])])
                    converted = {}
                    for column in targets:
                        if column not in frame.columns:
                            continue
                        # Strip the separators that make a numeric column read as
                        # text before giving up on it.
                        coerced = coerce_numeric(frame[column])
                        if coerced.notna().sum() > 0:
                            lost = int(coerced.isna().sum() - frame[column].isna().sum())
                            frame[column] = coerced
                            converted[column] = {"became_missing": max(0, lost)}
                    applied.append({"operation": name, "columns_converted": converted})

                elif name == "strip_whitespace":
                    text_columns = [c for c in frame.columns if frame[c].dtype == object]
                    for column in text_columns:
                        frame[column] = frame[column].astype(str).str.strip()
                    applied.append({"operation": name, "columns": text_columns})

                elif name == "drop_columns":
                    targets = [c.strip() for c in argument.split(",") if c.strip()]
                    present = [c for c in targets if c in frame.columns]
                    frame = frame.drop(columns=present)
                    applied.append({"operation": name, "dropped": present,
                                    "not_found": [c for c in targets if c not in present]})

                elif name == "rename_columns":
                    mapping = {}
                    for pair in argument.split(","):
                        old, _, new = pair.partition("=")
                        if old.strip() and new.strip():
                            mapping[old.strip()] = new.strip()
                    frame = frame.rename(columns=mapping)
                    applied.append({"operation": name, "renamed": mapping})

                elif name == "remove_outliers":
                    targets = ([c.strip() for c in argument.split(",") if c.strip()]
                               or [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])])
                    removed = {}
                    for column in targets:
                        if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
                            continue
                        series = frame[column]
                        q1, q3 = series.quantile(0.25), series.quantile(0.75)
                        iqr = q3 - q1
                        if iqr == 0:
                            continue
                        keep = series.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr) | series.isna()
                        removed[column] = int((~keep).sum())
                        frame = frame[keep]
                    applied.append({"operation": name, "rule": "IQR", "rows_removed_per_column": removed})

                else:
                    applied.append({"operation": name, "error": "unknown operation - skipped"})
                    continue

            except Exception as e:
                applied.append({"operation": name, "error": str(e)})

            missing_after = int(frame.isna().sum().sum())
            applied[-1]["missing_values_delta"] = missing_after - missing_before

        source = Path(frame.attrs.get("source_path", path))
        if output_path:
            destination = Path(output_path) if Path(output_path).is_absolute() else output_dir(project) / output_path
        else:
            destination = output_dir(project) / f"{source.stem}_cleaned{source.suffix or '.csv'}"
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            await asyncio.get_running_loop().run_in_executor(None, _write_frame, frame, destination)
        except Exception as e:
            return ToolResult(success=False, error=f"Cleaned the data but couldn't write it: {e}")

        return ToolResult(success=True, output={
            "source": str(source),
            "output_path": str(destination),
            "rows_before": before_rows,
            "rows_after": int(len(frame)),
            "rows_removed": before_rows - int(len(frame)),
            "columns_before": len(before_columns),
            "columns_after": len(frame.columns),
            "operations": applied,
            "note": "The original file is unchanged. Report what cleaning removed or filled.",
        })


def _write_frame(frame, destination: Path) -> None:
    """Write in whatever format the destination's extension asks for."""
    suffix = destination.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        frame.to_excel(destination, index=False)
    elif suffix == ".json":
        frame.to_json(destination, orient="records", indent=2)
    elif suffix in (".jsonl", ".ndjson"):
        frame.to_json(destination, orient="records", lines=True)
    elif suffix == ".parquet":
        frame.to_parquet(destination, index=False)
    elif suffix == ".tsv":
        frame.to_csv(destination, sep="\t", index=False)
    else:
        frame.to_csv(destination, index=False)


class ExportDatasetTool(BaseTool):
    name = "export_dataset"
    description = (
        "Convert a dataset to another format, or write a filtered subset of it. The output "
        "format follows the extension you give: .csv, .tsv, .xlsx, .json, .jsonl, .parquet."
    )
    parameters = [
        ToolParameter(name="path", type="string", description="Dataset to export."),
        ToolParameter(name="output_path", type="string", description="Destination, extension decides the format."),
        ToolParameter(name="columns", type="array", items_type="string", required=False,
                      description="Only these columns."),
        ToolParameter(name="query", type="string", required=False,
                      description="Row filter in pandas query syntax, e.g. 'power > 100 and month == 3'."),
        ToolParameter(name="project", type="string", required=False, description="Project to work in."),
    ]

    async def run(self, path: str, output_path: str, columns: Optional[List[str]] = None,
                  query: str = "", project: str = "", **kwargs) -> ToolResult:
        try:
            frame = await asyncio.get_running_loop().run_in_executor(
                None, lambda: load_dataset(path, project)
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        rows_before = len(frame)
        if query:
            try:
                frame = frame.query(query)
            except Exception as e:
                return ToolResult(success=False, error=f"Couldn't apply the filter {query!r}: {e}")
        if columns:
            missing = [c for c in columns if c not in frame.columns]
            if missing:
                return ToolResult(success=False, error=(
                    f"No such column(s): {missing}. Available: {list(frame.columns)}"
                ))
            frame = frame[columns]

        destination = (Path(output_path) if Path(output_path).is_absolute()
                       else output_dir(project) / output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.get_running_loop().run_in_executor(None, _write_frame, frame, destination)
        except Exception as e:
            return ToolResult(success=False, error=f"Couldn't write {destination}: {e}")

        return ToolResult(success=True, output={
            "output_path": str(destination),
            "rows_written": int(len(frame)),
            "rows_filtered_out": rows_before - int(len(frame)),
            "columns": [str(c) for c in frame.columns],
        })


class AnalyzeDatasetTool(BaseTool):
    name = "analyze_dataset"
    description = (
        "Statistical analysis of a dataset: summary statistics, correlations, group "
        "comparisons, trends over an ordered column, and linear regression with an "
        "R-squared and p-value.\n"
        "Choose the analysis that fits the question and the data - correlations on numeric "
        "columns, group_by to compare segments, trend for something ordered by time. "
        "Correlation is not causation and a p-value is not proof; report the numbers with "
        "what they do and don't establish."
    )
    parameters = [
        ToolParameter(name="path", type="string", description="Dataset to analyse."),
        ToolParameter(
            name="analyses", type="array", items_type="string",
            description=(
                "Which analyses to run: 'summary', 'correlations', 'group_by:<column>', "
                "'trend:<value_column>:<order_column>', 'regression:<y>:<x1,x2,...>', "
                "'distribution:<column>'."
            ),
        ),
        ToolParameter(name="project", type="string", required=False, description="Project to work in."),
        ToolParameter(name="sheet", type="string", required=False, description="Worksheet name, for Excel."),
    ]

    async def run(self, path: str, analyses: List[str], project: str = "",
                  sheet: str = "", **kwargs) -> ToolResult:
        try:
            frame = await asyncio.get_running_loop().run_in_executor(
                None, lambda: load_dataset(path, project, sheet or None)
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        import numpy as np
        import pandas as pd

        numeric_columns = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        results: Dict[str, Any] = {}

        for raw in analyses or ["summary"]:
            parts = str(raw).split(":")
            kind = parts[0].strip().lower()
            try:
                if kind == "summary":
                    results["summary"] = {
                        "rows": int(len(frame)),
                        "numeric_columns": numeric_columns,
                        "statistics": _json_safe(frame[numeric_columns].describe().to_dict())
                        if numeric_columns else {},
                    }

                elif kind == "correlations":
                    if len(numeric_columns) < 2:
                        results["correlations"] = {"error": "Needs at least two numeric columns."}
                        continue
                    matrix = frame[numeric_columns].corr(numeric_only=True)
                    pairs = []
                    for i, a in enumerate(numeric_columns):
                        for b in numeric_columns[i + 1:]:
                            value = matrix.loc[a, b]
                            if pd.notna(value):
                                pairs.append({"columns": [a, b], "correlation": _json_safe(value)})
                    pairs.sort(key=lambda p: abs(p["correlation"] or 0), reverse=True)
                    results["correlations"] = {
                        "strongest_pairs": pairs[:15],
                        "matrix": _json_safe(matrix.to_dict()),
                        "note": "Pearson correlation. Strong correlation is not causation.",
                    }

                elif kind == "group_by":
                    column = parts[1] if len(parts) > 1 else ""
                    if column not in frame.columns:
                        results[raw] = {"error": f"No column '{column}'. Have: {list(frame.columns)}"}
                        continue
                    grouped = frame.groupby(column)[numeric_columns].agg(["count", "mean", "sum"])
                    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
                    results[f"group_by:{column}"] = {
                        "groups": int(frame[column].nunique()),
                        "table": _json_safe(grouped.head(50).to_dict(orient="index")),
                    }

                elif kind == "trend":
                    value_column = parts[1] if len(parts) > 1 else ""
                    order_column = parts[2] if len(parts) > 2 else ""
                    if value_column not in frame.columns or order_column not in frame.columns:
                        results[raw] = {"error": f"Needs 'trend:<value>:<order>' with real columns."}
                        continue
                    ordered = frame[[order_column, value_column]].dropna().sort_values(order_column)
                    values = pd.to_numeric(ordered[value_column], errors="coerce").dropna()
                    if len(values) < 3:
                        results[raw] = {"error": "Not enough numeric points for a trend."}
                        continue
                    from scipy import stats

                    x = np.arange(len(values))
                    fit = stats.linregress(x, values.to_numpy())
                    first, last = values.iloc[0], values.iloc[-1]
                    results[f"trend:{value_column}"] = {
                        "ordered_by": order_column,
                        "points": int(len(values)),
                        "first": _json_safe(first),
                        "last": _json_safe(last),
                        "change": _json_safe(last - first),
                        "percent_change": _json_safe((last - first) / first * 100) if first else None,
                        "slope_per_step": _json_safe(fit.slope),
                        "r_squared": _json_safe(fit.rvalue ** 2),
                        "p_value": _json_safe(fit.pvalue),
                        "direction": "rising" if fit.slope > 0 else "falling" if fit.slope < 0 else "flat",
                    }

                elif kind == "regression":
                    y_column = parts[1] if len(parts) > 1 else ""
                    x_columns = [c.strip() for c in (parts[2] if len(parts) > 2 else "").split(",") if c.strip()]
                    missing = [c for c in [y_column] + x_columns if c not in frame.columns]
                    if missing or not x_columns:
                        results[raw] = {"error": f"Needs 'regression:<y>:<x1,x2>'; missing {missing}"}
                        continue
                    subset = frame[[y_column] + x_columns].apply(pd.to_numeric, errors="coerce").dropna()
                    if len(subset) <= len(x_columns) + 1:
                        results[raw] = {"error": "Not enough complete rows to fit."}
                        continue
                    y = subset[y_column].to_numpy()
                    design = np.column_stack([np.ones(len(subset))] + [subset[c].to_numpy() for c in x_columns])
                    coefficients, residuals, rank, _ = np.linalg.lstsq(design, y, rcond=None)
                    predicted = design @ coefficients
                    ss_res = float(((y - predicted) ** 2).sum())
                    ss_tot = float(((y - y.mean()) ** 2).sum())
                    results[f"regression:{y_column}"] = {
                        "predictors": x_columns,
                        "observations": int(len(subset)),
                        "intercept": _json_safe(coefficients[0]),
                        "coefficients": {c: _json_safe(v) for c, v in zip(x_columns, coefficients[1:])},
                        "r_squared": _json_safe(1 - ss_res / ss_tot) if ss_tot else None,
                        "note": ("Ordinary least squares. Coefficients describe association in "
                                 "this sample, not causation, and assume a linear relationship."),
                    }

                elif kind == "distribution":
                    column = parts[1] if len(parts) > 1 else ""
                    if column not in frame.columns:
                        results[raw] = {"error": f"No column '{column}'."}
                        continue
                    series = pd.to_numeric(frame[column], errors="coerce").dropna()
                    if series.empty:
                        results[raw] = {"counts": _json_safe(frame[column].value_counts().head(20).to_dict())}
                        continue
                    from scipy import stats

                    results[f"distribution:{column}"] = {
                        "mean": _json_safe(series.mean()),
                        "median": _json_safe(series.median()),
                        "std": _json_safe(series.std()),
                        "skew": _json_safe(series.skew()),
                        "kurtosis": _json_safe(series.kurtosis()),
                        "percentiles": {str(p): _json_safe(series.quantile(p / 100))
                                        for p in (1, 5, 25, 50, 75, 95, 99)},
                        "normality_p_value": _json_safe(stats.normaltest(series).pvalue)
                        if len(series) >= 20 else None,
                    }

                else:
                    results[raw] = {"error": f"Unknown analysis '{kind}'."}

            except Exception as e:
                results[raw] = {"error": str(e)}

        return ToolResult(success=True, output={
            "path": frame.attrs.get("source_path"),
            "rows": int(len(frame)),
            "results": results,
        })


class VisualizeDatasetTool(BaseTool):
    name = "visualize_dataset"
    description = (
        "Plot a dataset and save the chart as a PNG in the project folder. Kinds: line, bar, "
        "scatter, histogram, box, correlation_heatmap. The chart is shown to the user "
        "automatically once it's made."
    )
    parameters = [
        ToolParameter(name="path", type="string", description="Dataset to plot."),
        ToolParameter(name="kind", type="string",
                      enum=["line", "bar", "scatter", "histogram", "box", "correlation_heatmap"],
                      description="Chart type."),
        ToolParameter(name="x", type="string", required=False, description="Column for the x axis."),
        ToolParameter(name="y", type="array", items_type="string", required=False,
                      description="Column(s) for the y axis."),
        ToolParameter(name="title", type="string", required=False, description="Chart title."),
        ToolParameter(name="output_path", type="string", required=False, description="Where to save the PNG."),
        ToolParameter(name="project", type="string", required=False, description="Project to work in."),
    ]

    async def run(self, path: str, kind: str, x: str = "", y: Optional[List[str]] = None,
                  title: str = "", output_path: str = "", project: str = "", **kwargs) -> ToolResult:
        try:
            frame = await asyncio.get_running_loop().run_in_executor(
                None, lambda: load_dataset(path, project)
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        destination = (Path(output_path) if output_path and Path(output_path).is_absolute()
                       else output_dir(project) / (output_path or f"{Path(path).stem}_{kind}.png"))
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, _plot, frame, kind, x, list(y or []), title, destination
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Couldn't draw the chart: {e}")

        # Shown in the GUI through the orchestrator's visual_callback, using the
        # "images" payload the HUD already renders rather than a new visual type.
        # The PNG is inlined as a data URI because the browser can't load a local
        # file path, and inlining beats opening a file route onto the disk.
        chart_label = title or f"{kind} chart"
        visual = {"type": "images", "query": chart_label, "items": [{
            "title": chart_label,
            "thumbnail": _data_uri(destination),
            "url": _data_uri(destination),
        }]}
        return ToolResult(success=True, output={
            "chart_path": str(destination),
            "kind": kind,
            "visual": visual,
        })


def _data_uri(path: Path) -> str:
    """A PNG as a data: URI, so the HUD can show it without a route to the disk."""
    import base64

    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _plot(frame, kind: str, x: str, y: List[str], title: str, destination: Path) -> None:
    """Draw and save. Runs in a thread - matplotlib is synchronous."""
    import matplotlib
    # Agg: there's no display to draw into, and importing a GUI backend off the
    # main thread is what makes matplotlib crash rather than plot.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    y_columns = [c for c in y if c in frame.columns] or [c for c in numeric if c != x][:4]

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(10, 6))
    try:
        if kind == "correlation_heatmap":
            matrix = frame[numeric].corr(numeric_only=True)
            image = axes.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
            axes.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
            axes.set_yticks(range(len(matrix.columns)), matrix.columns)
            figure.colorbar(image, ax=axes)
        elif kind == "histogram":
            for column in y_columns:
                axes.hist(pd.to_numeric(frame[column], errors="coerce").dropna(),
                          bins=30, alpha=0.65, label=str(column))
            axes.legend()
        elif kind == "box":
            data = [pd.to_numeric(frame[c], errors="coerce").dropna() for c in y_columns]
            axes.boxplot(data, tick_labels=[str(c) for c in y_columns])
        elif kind == "scatter":
            if not x:
                raise ValueError("scatter needs an x column")
            for column in y_columns:
                axes.scatter(frame[x], frame[column], alpha=0.6, label=str(column))
            axes.set_xlabel(x)
            axes.legend()
        elif kind == "bar":
            plot_frame = frame.set_index(x)[y_columns] if x else frame[y_columns]
            plot_frame.head(40).plot(kind="bar", ax=axes)
        else:  # line
            plot_frame = frame.set_index(x)[y_columns] if x else frame[y_columns]
            plot_frame.plot(ax=axes)

        axes.set_title(title or destination.stem.replace("_", " "))
        axes.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(destination, dpi=120)
    finally:
        plt.close(figure)
