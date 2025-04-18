"""
Optuna study analysis and visualization script (refactored with jitter support).

- pathlib for filesystem operations
- Centralized logging
- Generic plotting helper to reduce code duplication
- Optional jitter on parallel‑coordinate plots to reduce over‑plotting
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np
import optuna
from optuna.study import Study
from optuna.trial import FrozenTrial

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------


@dataclass
class StudyConfig:
    """Runtime configuration for analysis."""

    study_name: str
    db_path: Path
    output_dir: Path
    top_percent: float = 100.0

    # jitter settings (only applied to parallel‑coordinate plots)
    jitter_frac: float = 0.0  # e.g. 0.02 gives 2 % of axis range
    jitter_seed: int = 0


# ----------------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------------


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """Return module‑level logger with consistent formatting."""

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return logger


logger = setup_logger()

# ----------------------------------------------------------------------------
# filesystem helpers
# ----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# study I/O helpers
# ----------------------------------------------------------------------------


def load_study(name: str, storage: str) -> Study:
    """Load Optuna Study from given storage."""

    study = optuna.load_study(study_name=name, storage=storage)
    logger.info("Loaded study '%s' with %d trials", name, len(study.trials))
    return study


def create_temp_study(trials: Sequence[FrozenTrial], template: Study) -> Study:
    """Create in‑memory study that contains only the supplied trials."""

    tmp = optuna.create_study(
        study_name=f"tmp_{template.study_name}",
        direction=template.direction,
        storage=None,
    )
    tmp.add_trials(list(trials))
    return tmp


# ----------------------------------------------------------------------------
# jitter utility for parallel‑coordinate plots
# ----------------------------------------------------------------------------


def add_jitter_to_parallel(fig, frac: float = 0.01, seed: int = 0) -> None:  # noqa: ANN001
    """Add Gaussian jitter to each numerical axis of a plotly parallel‑coordinate figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure returned by ``optuna.visualization.plot_parallel_coordinate``.
    frac : float, default 0.01
        Standard deviation of noise, expressed as a fraction of axis range.
    seed : int, default 0
        RNG seed for reproducibility.
    """

    rng = np.random.default_rng(seed)

    for trace in fig.data:
        # parallel‑coordinate in plotly stores dimensions under trace.dimensions
        # Each dimension has a .values list (numbers or strings)
        for dim in trace.dimensions:
            if not hasattr(dim, "values"):
                continue
            try:
                vals = np.asarray(dim.values, dtype=float)
            except (TypeError, ValueError):
                # non‑numeric axis → skip
                continue

            span = vals.max() - vals.min()
            if span == 0:
                continue  # constant axis → skip

            jitter = rng.normal(0.0, frac * span, size=vals.shape)
            dim.values = list(vals + jitter)


# ----------------------------------------------------------------------------
# generic plot helper
# ----------------------------------------------------------------------------


PlotFunc = Callable[..., "go.Figure"]  # type: ignore[name-defined]


def _save_plot(
    target: Union[Study, Sequence[FrozenTrial]],
    base: Study,
    out_dir: Path,
    filename: str,
    plot_fn: PlotFunc,
    jitter_frac: float = 0.0,
    jitter_seed: int = 0,
) -> None:
    """Generate plot with *plot_fn* and save as HTML.

    If *jitter_frac* > 0 and the plot function corresponds to a parallel‑coordinate
    plot, Gaussian jitter is applied for readability.
    """

    # normalise target to Study
    study: Study
    if isinstance(target, Study):
        study = target
    else:
        study = create_temp_study(target, base)

    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        logger.warning("No completed trials for %s ‑ skipping", filename)
        return

    params = next((list(t.params.keys()) for t in study.trials if t.params), None)
    needs_params = "params" in inspect.signature(plot_fn).parameters

    if needs_params and params is None:
        logger.warning("Could not determine parameters for %s ‑ skipping", filename)
        return

    try:
        fig = (
            plot_fn(study, params=params)  # type: ignore[arg-type]
            if needs_params
            else plot_fn(study)  # type: ignore[arg-type]
        )

        if jitter_frac > 0.0 and "parallel_coordinate" in filename:
            add_jitter_to_parallel(fig, frac=jitter_frac, seed=jitter_seed)

        out_path = out_dir / f"{filename}.html"
        fig.write_html(out_path)
        logger.info("Saved %s", out_path)
    except Exception as exc:
        logger.error("Failed to create %s: %s", filename, exc)


# ----------------------------------------------------------------------------
# main analysis entrypoint
# ----------------------------------------------------------------------------


def analyze(cfg: StudyConfig) -> None:
    """Run analysis with plots; optional jitter for parallel‑coordinate plots."""

    ensure_dir(cfg.output_dir)
    study = load_study(cfg.study_name, f"sqlite:///{cfg.db_path}")

    # full study plots (no jitter needed except parallel‑coordinate)
    _save_plot(
        study,
        study,
        cfg.output_dir,
        "optimization_history",
        optuna.visualization.plot_optimization_history,
    )
    _save_plot(
        study,
        study,
        cfg.output_dir,
        "param_importances",
        optuna.visualization.plot_param_importances,
    )
    _save_plot(
        study,
        study,
        cfg.output_dir,
        "slice_plot_all",
        optuna.visualization.plot_slice,
    )
    _save_plot(
        study,
        study,
        cfg.output_dir,
        "parallel_coordinate_all",
        optuna.visualization.plot_parallel_coordinate,
        jitter_frac=cfg.jitter_frac,
        jitter_seed=cfg.jitter_seed,
    )

    # top‑N percent plots
    if cfg.top_percent < 100.0:
        completed = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        completed.sort(
            key=lambda t: t.value,
            reverse=study.direction == optuna.study.StudyDirection.MAXIMIZE,
        )
        k = max(1, int(len(completed) * (cfg.top_percent / 100)))
        top_trials = completed[:k]
        suffix = f"_top_{int(cfg.top_percent)}"

        _save_plot(
            top_trials,
            study,
            cfg.output_dir,
            f"slice_plot{suffix}",
            optuna.visualization.plot_slice,
        )
        _save_plot(
            top_trials,
            study,
            cfg.output_dir,
            f"parallel_coordinate{suffix}",
            optuna.visualization.plot_parallel_coordinate,
            jitter_frac=cfg.jitter_frac,
            jitter_seed=cfg.jitter_seed,
        )


# ----------------------------------------------------------------------------
# cli usage
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # user editable section
    STUDY_NAME = "05-24-50__IAT_201702_optimize_building"
    DB_PATH = Path("./trials") / STUDY_NAME / "study.db"
    OUTPUT_DIR = Path("./optuna_plots_refactored")
    TOP_PERCENT = 20.0

    analyze(
        StudyConfig(
            study_name=STUDY_NAME,
            db_path=DB_PATH,
            output_dir=OUTPUT_DIR,
            top_percent=TOP_PERCENT,
            jitter_frac=0.01,  # 2 % jitter on parallel‑coordinate plots
            jitter_seed=42,
        )
    )
