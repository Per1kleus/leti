"""Tests for project workspaces."""
from __future__ import annotations

import shutil

import pytest

from tools.projects import (
    CreateProjectTool, DeleteProjectTool, ListProjectsTool, OpenProjectTool,
    UpdateProjectTool, get_active_project, project_dir, projects_root,
    resolve_project, safe_project_path, set_active_project,
)


@pytest.fixture(autouse=True)
def clean_projects(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.projects_root", lambda: tmp_path / "projects")
    monkeypatch.setattr("tools.projects._state_path", lambda: tmp_path / "active.json")
    (tmp_path / "projects").mkdir()
    yield
    shutil.rmtree(tmp_path / "projects", ignore_errors=True)


@pytest.mark.parametrize("name", ["../../etc", "a/../../b", "", "   ", "a/b/c/d/e"])
def test_project_names_cannot_escape_the_projects_root(name):
    """The name is joined onto the root to build a real directory that create
    makes and delete rmtree's, so a '..' segment would act outside it."""
    with pytest.raises(ValueError):
        safe_project_path(name)


@pytest.mark.asyncio
async def test_projects_nest_as_a_real_directory_hierarchy():
    await CreateProjectTool().run(name="University", make_active=False)
    result = await CreateProjectTool().run(name="University/MATLAB", make_active=False)
    assert result.success
    assert project_dir("University/MATLAB").is_dir()
    assert project_dir("University/MATLAB").parent == project_dir("University")


@pytest.mark.asyncio
async def test_a_parent_does_not_claim_its_childs_files():
    await CreateProjectTool().run(name="University", make_active=False)
    await CreateProjectTool().run(name="University/MATLAB", make_active=False)
    (project_dir("University") / "syllabus.pdf").write_text("x")
    (project_dir("University/MATLAB") / "sim.m").write_text("x")

    counts = {p["name"]: p["file_count"] for p in (await ListProjectsTool().run()).output["projects"]}
    assert counts["University"] == 1
    assert counts["University/MATLAB"] == 1


@pytest.mark.asyncio
async def test_a_shortened_name_resolves_to_the_one_project_it_matches():
    """"Continue the MATLAB project" shouldn't fail because the project's full
    path is University/MATLAB."""
    await CreateProjectTool().run(name="University/MATLAB", make_active=False)
    result = await OpenProjectTool().run(name="MATLAB")
    assert result.success
    assert result.output["project"] == "University/MATLAB"


@pytest.mark.asyncio
async def test_an_ambiguous_short_name_asks_rather_than_guessing():
    await CreateProjectTool().run(name="University/Reports", make_active=False)
    await CreateProjectTool().run(name="Business/Reports", make_active=False)
    result = await OpenProjectTool().run(name="Reports")
    assert result.success is False
    assert "several projects" in result.error


@pytest.mark.asyncio
async def test_instructions_and_files_reach_the_model():
    await CreateProjectTool().run(name="University/MATLAB", description="Course sims")
    await UpdateProjectTool().run(name="University/MATLAB", instructions="Always use SI units.")
    (project_dir("University/MATLAB") / "turbine.m").write_text("% sim")

    context = (await OpenProjectTool().run(name="University/MATLAB")).output["context"]
    assert "Always use SI units." in context
    assert "turbine.m" in context
    assert "Course sims" in context


@pytest.mark.asyncio
async def test_the_active_project_carries_into_later_calls():
    """What makes a project a workspace rather than a folder: tools that take an
    optional project argument fall back to the active one."""
    await CreateProjectTool().run(name="Business/Clients")
    assert resolve_project(None) == "Business/Clients"
    assert resolve_project("University") == "University"


@pytest.mark.asyncio
async def test_deleting_requires_repeating_the_name():
    """The model picks this argument and the delete is unrecoverable."""
    await CreateProjectTool().run(name="Personal/Finance", make_active=False)
    refused = await DeleteProjectTool().run(name="Personal/Finance", confirm_name="Finance")
    assert refused.success is False
    assert project_dir("Personal/Finance").is_dir()

    deleted = await DeleteProjectTool().run(name="Personal/Finance", confirm_name="Personal/Finance")
    assert deleted.success
    assert not project_dir("Personal/Finance").exists()


@pytest.mark.asyncio
async def test_deleting_reports_the_nested_projects_that_go_with_it():
    await CreateProjectTool().run(name="University", make_active=False)
    await CreateProjectTool().run(name="University/MATLAB", make_active=False)
    result = await DeleteProjectTool().run(name="University", confirm_name="University")
    assert "University/MATLAB" in result.output


@pytest.mark.asyncio
async def test_active_pointer_clears_when_its_project_is_deleted():
    await CreateProjectTool().run(name="Temp")
    assert get_active_project() == "Temp"
    await DeleteProjectTool().run(name="Temp", confirm_name="Temp")
    assert get_active_project() is None


def test_active_pointer_survives_a_stale_name():
    """The pointer file outlives the project folder; a stale name must read as
    'no project' rather than pointing at something that isn't there."""
    set_active_project("Does/NotExist")
    assert get_active_project() is None
