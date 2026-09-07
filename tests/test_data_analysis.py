"""Tests for dataset loading, inspection, cleaning, analysis and export."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("pandas")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools.data_analysis import (  # noqa: E402
    AnalyzeDatasetTool, CleanDatasetTool, ExportDatasetTool, InspectDatasetTool,
    VisualizeDatasetTool, coerce_numeric, load_dataset,
)


@pytest.fixture
def messy_csv(tmp_path):
    """A dataset with every problem inspect is supposed to name."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "month": np.repeat(np.arange(1, 11), 10),
        "wind_speed": rng.normal(9, 2, 100).round(2),
        "site": rng.choice(["north", "south"], 100),
        "price": [f"{v:,.2f} €" for v in rng.normal(120, 10, 100)],
    })
    frame["power_kw"] = (frame.wind_speed ** 3 * 1.2).round(1)
    frame.loc[5:9, "power_kw"] = np.nan
    frame.loc[0, "power_kw"] = 99999
    frame = pd.concat([frame, frame.iloc[[1, 2]]])
    path = tmp_path / "turbine.csv"
    frame.to_csv(path, index=False)
    return path


# --- Loading -------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", [".csv", ".xlsx", ".json", ".jsonl", ".tsv"])
async def test_every_format_loads_into_the_same_shape(tmp_path, suffix):
    """Adding a format is one loader entry; the tools stay format-blind."""
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / f"data{suffix}"
    {".csv": lambda: frame.to_csv(path, index=False),
     ".tsv": lambda: frame.to_csv(path, sep="\t", index=False),
     ".xlsx": lambda: frame.to_excel(path, index=False),
     ".json": lambda: frame.to_json(path, orient="records"),
     ".jsonl": lambda: frame.to_json(path, orient="records", lines=True)}[suffix]()

    result = await InspectDatasetTool().run(path=str(path))
    assert result.success, result.error
    assert result.output["rows"] == 3
    assert result.output["columns"] == 2


def test_matlab_file_picks_the_largest_numeric_array(tmp_path):
    """A .mat holds named arrays, not a table; which one was used is reported."""
    from scipy.io import savemat

    savemat(tmp_path / "d.mat", {"readings": np.arange(40).reshape(20, 2).astype(float),
                                 "config": np.array([1.0])})
    frame = load_dataset(str(tmp_path / "d.mat"))
    assert frame.attrs["matlab_variable"] == "readings"
    assert frame.shape == (20, 2)


@pytest.mark.asyncio
async def test_unknown_format_says_what_is_supported(tmp_path):
    (tmp_path / "x.docx").write_bytes(b"nope")
    result = await InspectDatasetTool().run(path=str(tmp_path / "x.docx"))
    assert result.success is False
    assert ".csv" in result.error


@pytest.mark.asyncio
async def test_missing_file_is_reported(tmp_path):
    result = await InspectDatasetTool().run(path=str(tmp_path / "nope.csv"))
    assert result.success is False


# --- Inspection ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inspection_names_every_problem(messy_csv):
    output = (await InspectDatasetTool().run(path=str(messy_csv))).output
    issues = " ".join(output["issues"])
    assert output["duplicate_rows"] == 2
    assert "duplicate" in issues
    assert "missing" in issues
    assert "outlier" in issues
    assert "text but reads as numeric" in issues


@pytest.mark.asyncio
async def test_inspection_changes_nothing(messy_csv):
    before = messy_csv.read_bytes()
    await InspectDatasetTool().run(path=str(messy_csv))
    assert messy_csv.read_bytes() == before


@pytest.mark.asyncio
async def test_results_are_json_serialisable(tmp_path):
    """NaN is not valid JSON, and tool results are serialised to reach the model."""
    (tmp_path / "gaps.csv").write_text("a,b\n1,\n,2\n")
    result = await InspectDatasetTool().run(path=str(tmp_path / "gaps.csv"))
    json.dumps(result.output)


def test_numeric_detection_and_conversion_agree():
    """inspect flags what clean can fix - the same stripping on both sides."""
    series = pd.Series(["1,200.50 €", "990.00 €", "1,500.75 €"])
    converted = coerce_numeric(series)
    assert converted.notna().all()
    assert converted.iloc[0] == 1200.50


# --- Cleaning ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cleaning_never_touches_the_original(messy_csv, tmp_path):
    before = messy_csv.read_bytes()
    result = await CleanDatasetTool().run(
        path=str(messy_csv), output_path=str(tmp_path / "clean.csv"),
        operations=["drop_duplicates", "fill_missing:median"],
    )
    assert result.success
    assert messy_csv.read_bytes() == before
    assert (tmp_path / "clean.csv").is_file()


@pytest.mark.asyncio
async def test_each_operation_reports_what_it_changed(messy_csv, tmp_path):
    """Cleaning changes what the numbers mean, so it has to be accountable."""
    result = await CleanDatasetTool().run(
        path=str(messy_csv), output_path=str(tmp_path / "c.csv"),
        operations=["drop_duplicates", "fill_missing:median", "remove_outliers:power_kw"],
    )
    operations = {op["operation"]: op for op in result.output["operations"]}
    assert operations["drop_duplicates"]["rows_removed"] == 2
    assert operations["fill_missing"]["values_filled"] == 5
    assert operations["remove_outliers"]["rows_removed_per_column"]["power_kw"] >= 1


@pytest.mark.asyncio
async def test_an_unknown_operation_is_reported_not_silently_skipped(messy_csv, tmp_path):
    result = await CleanDatasetTool().run(
        path=str(messy_csv), output_path=str(tmp_path / "c.csv"), operations=["descale_flux"],
    )
    assert "error" in result.output["operations"][0]


# --- Analysis ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_correlation_finds_a_real_relationship(tmp_path):
    x = np.linspace(0, 10, 100)
    pd.DataFrame({"x": x, "y": 3 * x + 1, "noise": np.random.default_rng(1).normal(0, 1, 100)}
                 ).to_csv(tmp_path / "d.csv", index=False)

    output = (await AnalyzeDatasetTool().run(path=str(tmp_path / "d.csv"),
                                             analyses=["correlations"])).output
    top = output["results"]["correlations"]["strongest_pairs"][0]
    assert set(top["columns"]) == {"x", "y"}
    assert top["correlation"] > 0.99


@pytest.mark.asyncio
async def test_regression_recovers_a_known_slope(tmp_path):
    x = np.linspace(0, 10, 100)
    pd.DataFrame({"x": x, "y": 2.5 * x + 7}).to_csv(tmp_path / "d.csv", index=False)

    output = (await AnalyzeDatasetTool().run(path=str(tmp_path / "d.csv"),
                                             analyses=["regression:y:x"])).output
    fit = output["results"]["regression:y"]
    assert fit["coefficients"]["x"] == pytest.approx(2.5, abs=1e-6)
    assert fit["intercept"] == pytest.approx(7.0, abs=1e-6)


@pytest.mark.asyncio
async def test_trend_direction_matches_the_data(tmp_path):
    pd.DataFrame({"month": range(1, 13), "revenue": range(100, 220, 10)}
                 ).to_csv(tmp_path / "d.csv", index=False)
    output = (await AnalyzeDatasetTool().run(
        path=str(tmp_path / "d.csv"), analyses=["trend:revenue:month"])).output
    assert output["results"]["trend:revenue"]["direction"] == "rising"


@pytest.mark.asyncio
async def test_a_bad_analysis_spec_reports_rather_than_crashes(tmp_path):
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(tmp_path / "d.csv", index=False)
    output = (await AnalyzeDatasetTool().run(
        path=str(tmp_path / "d.csv"), analyses=["group_by:nonexistent"])).output
    assert "error" in output["results"]["group_by:nonexistent"]


# --- Output -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chart_uses_a_visual_type_the_gui_renders(tmp_path):
    """showVisual only handles 'images' and 'diagram'; anything else is dropped."""
    pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}).to_csv(tmp_path / "d.csv", index=False)
    result = await VisualizeDatasetTool().run(
        path=str(tmp_path / "d.csv"), kind="line", y=["a", "b"],
        output_path=str(tmp_path / "chart.png"),
    )
    assert (tmp_path / "chart.png").is_file()
    assert result.output["visual"]["type"] == "images"
    assert result.output["visual"]["items"][0]["thumbnail"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_export_filters_and_converts_format(tmp_path):
    pd.DataFrame({"site": ["n", "s", "n"], "v": [1, 2, 3]}).to_csv(tmp_path / "d.csv", index=False)
    result = await ExportDatasetTool().run(
        path=str(tmp_path / "d.csv"), output_path=str(tmp_path / "out.xlsx"),
        query="site == 'n'", columns=["v"],
    )
    assert result.output["rows_written"] == 2
    assert list(pd.read_excel(tmp_path / "out.xlsx").columns) == ["v"]


@pytest.mark.asyncio
async def test_export_names_columns_that_do_not_exist(tmp_path):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "d.csv", index=False)
    result = await ExportDatasetTool().run(
        path=str(tmp_path / "d.csv"), output_path=str(tmp_path / "o.csv"), columns=["nope"],
    )
    assert result.success is False
    assert "nope" in result.error
