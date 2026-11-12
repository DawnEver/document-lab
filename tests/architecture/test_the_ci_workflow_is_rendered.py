"""The GitHub Actions workflow is RENDERED from TOML (`dev_lab.dev.ciworkflows`): no drift, no hand-written file."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dev_lab.dev.ciworkflows import drift, workflows

ROOT = Path(__file__).resolve().parents[2]


def test_the_repo_runs_the_family_pr_lifecycle() -> None:
    # dev-lab #92: a GitHub repo renders its forge's whole base set.
    # review-prepare.yml gained 2026-11-10: a judge repo renders the bundle prepare its judge reads;
    # review-again.yml left 2026-11-11 with dev-lab #443 (the judge is advisory; Actions alone gate).
    assert set(workflows(ROOT)) == {
        'ci.yml',
        'continue.yml',
        'merge-tick.yml',
        'review-prepare.yml',
    }


def test_every_workflow_matches_its_toml() -> None:
    assert drift(ROOT) == []


def _syncs() -> list[str]:
    return [
        step['run']
        for workflow in workflows(ROOT).values()
        for job in workflow['jobs'].values()
        for step in job.get('steps', [])
        if str(step.get('run', '')).startswith('uv sync')
    ]


def test_hosted_jobs_install_no_unproduced_resource_extra() -> None:
    # A hosted runner never produces the `unproduced` parts ([tool.dev_lab.parts]), so it never installs their
    # extras: `--all-extras` pulled `gpu` (marker-pdf -> pillow 10.4.0, no wheel for the runner's Python) and failed
    # the `parts` job on a source build (e0b6e02). Planted control: `--all-extras` or `--extra gpu` reads red here.

    unproduced = tomllib.loads((ROOT / 'pyproject.toml').read_text())['tool']['dev_lab']['parts']['unproduced']
    syncs = _syncs()
    assert syncs
    for run in syncs:
        assert '--all-extras' not in run
        assert not any(f'--extra {name}' in run for name in unproduced)
