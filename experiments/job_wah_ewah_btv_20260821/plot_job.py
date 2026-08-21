#!/usr/bin/env python3
"""Render the isolated JOB WAH/EWAH BTV-delivery sensitivity."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_PLOT = HERE.parent / "cr_native_no_btv_20260820" / "plot_results.py"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("base_plot", BASE_PLOT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_PLOT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CR_NATIVE_SOURCE["COMP"] = "COMP_op"
    return module


def main() -> None:
    module = load_plot_module()
    rows = module.read_logical(
        HERE / "results" / "job_wah_ewah_btv.csv",
        "job", module.JOB_CASES, ("COMP",),
    )
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    outputs = module.plot_relative_bars(
        rows, module.JOB_CASES, ("COMP",), module.JOB_LABELS,
        figures / "job_comp_wah_ewah_btv_literal_relative", 240,
    )
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
