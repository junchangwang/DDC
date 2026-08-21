#!/usr/bin/env python3
"""Render all selected-delivery figures from the fresh compact CSVs."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_PLOT = HERE.parent / "cr_native_no_btv_20260820" / "plot_results.py"


def load_module():
    spec = importlib.util.spec_from_file_location("base_plot", BASE_PLOT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_PLOT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CR_NATIVE_SOURCE = {
        "OR": "OR_op_conv",
        "AND": "AND_op_conv",
        "NOT": "NOT_op_conv",
        "COMP": "COMP_op",
    }
    return module


def rename_outputs(paths: list[Path], old: str, new: str) -> list[Path]:
    result = []
    for path in paths:
        renamed = path.with_name(path.name.replace(old, new))
        path.replace(renamed)
        result.append(renamed)
    return result


def main() -> None:
    module = load_module()
    results = HERE / "results"
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    earth = module.read_logical(
        results / "earth_selected_delivery.csv", "earth", module.EARTH_CASES,
        ("OR", "COMP"),
    )
    job = module.read_logical(
        results / "job_selected_delivery.csv", "job", module.JOB_CASES,
        ("COMP",),
    )
    cluster = module.read_logical(
        results / "cluster_selected_delivery.csv", "cluster",
        module.CLUSTER_CASES, module.OPERATIONS, cluster=True,
    )
    density = module.read_density(results / "density_selected_delivery.csv")
    outputs: list[Path] = []
    outputs.extend(module.plot_relative_bars(
        earth, module.EARTH_CASES, ("OR", "COMP"), module.EARTH_LABELS,
        figures / "earth_or_comp_runopt_btv_literal_relative", 240,
    ))
    outputs.extend(module.plot_relative_bars(
        job, module.JOB_CASES, ("COMP",), module.JOB_LABELS,
        figures / "job_comp_runopt_btv_literal_relative", 240,
    ))
    cluster_outputs = module.plot_cluster(cluster, figures, 240)
    outputs.extend(rename_outputs(cluster_outputs, "_native", "_runopt_btv_literal"))
    density_outputs = module.plot_density(density, figures, 240)
    outputs.extend(rename_outputs(density_outputs, "_native", "_runopt_btv_literal"))
    for path in outputs:
        print(f"wrote {path}")
    print(f"rendered {len(outputs) // 2} figures")


if __name__ == "__main__":
    main()
