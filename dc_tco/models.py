"""AI model size trend analysis and projection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config, ProjectionMode


# ---------------------------------------------------------------------------
# Notable AI models dataset
# ---------------------------------------------------------------------------

def _bundled_csv_path() -> Path:
    """Path to the bundled notable_ai_models.csv."""
    return Path(__file__).resolve().parent.parent / "data" / "notable_ai_models.csv"


def parse_params(param) -> Optional[float]:
    if isinstance(param, (int, float, np.integer, np.floating)) and not isinstance(param, bool):
        return None if np.isnan(param) else float(param)
    if not isinstance(param, str):
        return None
    return parse_params_str(param)


def parse_params_str(
    param: str
) -> Optional[float]:
    """Parse a parameter-count string like '70B', '1.5M', '500K' to a float."""
    assert isinstance(param, str)
    p = param.strip().replace(",", "")
    match = re.match(r"^([\d.]+)\s*([BKMGT]?)$", p, re.IGNORECASE)
    if match:
        try:
            MULTIPLIERS = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9, "T": 1e12}
            base = float(match.group(1))
            multiplier = MULTIPLIERS.get(match.group(2).upper(), 1)
            return base * multiplier
        except (ValueError, KeyError):
            return None
    try:
        return float(p)
    except (ValueError, TypeError):
        return None


def load_notable_models(
    csv_path: Optional[str] = None,
    col_params: Optional[str] = None,
    col_date: Optional[str] = None,
    col_name: Optional[str] = None,
    min_year: int = 2015,
) -> pd.DataFrame:
    """Load an AI-models dataset from CSV.

    Works with the Epoch AI notable-models CSV out of the box, but also
    accepts any CSV that has columns for *parameter count*, *date/year*,
    and (optionally) *model name*.  Pass explicit column names via
    ``col_params`` / ``col_date`` / ``col_name`` when auto-detection
    doesn't match your file's headers.

    Returns a DataFrame with columns: year, model_name, parameters (float).
    """
    if csv_path is None or csv_path == "":
        csv_path = str(_bundled_csv_path())

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Notable models CSV not found: {path}")

    df = pd.read_csv(path)

    # --- Resolve parameter column ---
    param_col = col_params
    if not param_col:
        for col in ["Parameters", "parameters", "params", "Params",
                    "Training compute (FLOP)", "parameter_count"]:
            if col in df.columns:
                param_col = col
                break
    if param_col is None or param_col not in df.columns:
        raise ValueError(
            f"Cannot identify parameter column in CSV. "
            f"Set model_trends.csv_col_params in your config. "
            f"Available columns: {list(df.columns)}"
        )

    # --- Resolve date column ---
    date_col = col_date
    if not date_col:
        for col in ["Publication date", "publication_date", "year", "Year",
                    "Date", "date", "release_date"]:
            if col in df.columns:
                date_col = col
                break
    if date_col is None or date_col not in df.columns:
        raise ValueError(
            f"Cannot identify date column in CSV. "
            f"Set model_trends.csv_col_date in your config. "
            f"Available columns: {list(df.columns)}"
        )

    # --- Resolve name column ---
    name_col = col_name
    if not name_col:
        for col in ["System", "model_name", "Model", "Name", "name"]:
            if col in df.columns:
                name_col = col
                break

    # Parse parameters
    df["parameters"] = df[param_col].apply(parse_params)
    df = df.dropna(subset=["parameters"])
    df = df[df["parameters"] > 0]

    # Parse year
    if df[date_col].dtype == object or pd.api.types.is_string_dtype(df[date_col]):
        df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    else:
        df["year"] = df[date_col].astype(int)

    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[df["year"] >= min_year]

    if name_col:
        df["model_name"] = df[name_col]
    else:
        df["model_name"] = "unknown"

    return df[["year", "model_name", "parameters"]].reset_index(drop=True)


def compute_yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute P50, P99, and mean parameter count per year.

    Returns a DataFrame indexed by year with columns: p50, p99, mean.
    """
    grouped = df.groupby("year")["parameters"]
    stats = pd.DataFrame({
        "p50": grouped.quantile(0.5),
        "p99": grouped.quantile(0.99),
        "mean": grouped.mean(),
    })
    return stats.sort_index()


# ---------------------------------------------------------------------------
# Model size projection
# ---------------------------------------------------------------------------

def project(values, known_years, future_years, mode) -> np.ndarray:
    projected = np.zeros(len(known_years) + len(future_years))
    n_known = len(values)
    projected[:n_known] = values
    if mode == ProjectionMode.FLAT:
        projected[n_known:] = values[-1]
    elif mode == ProjectionMode.LINEAR:
        coeffs = np.polyfit(known_years, values, 1)
        projected[n_known:] = np.polyval(coeffs, future_years)
    elif mode == ProjectionMode.EXPONENTIAL:
        log_vals = np.log(values)
        slope, intercept = np.polyfit(known_years, log_vals, 1)
        projected[n_known:] = np.exp(intercept + slope * np.array(future_years))
    else:
        raise ValueError(f"Unknown projection mode: {mode}")
    return projected


def project_model_sizes(cfg: Config) -> List[float]:
    """Project model sizes for each quarter of the simulation.

    Returns a list of model sizes (in raw parameter count) for each quarter,
    where each year shares the same model size (repeated per quarter).
    """
    csv_path = cfg.model_trends.notable_models_csv
    try:
        df = load_notable_models(
            csv_path,
            col_params=cfg.model_trends.csv_col_params,
            col_date=cfg.model_trends.csv_col_date,
            col_name=cfg.model_trends.csv_col_name,
            min_year=cfg.model_trends.csv_min_year,
        )
        stats = compute_yearly_stats(df)
    except (FileNotFoundError, ValueError):
        # Fallback: use a default trajectory based on known models
        stats = _default_model_stats()

    # Use P50 for projection (conservative estimate)
    known_years = np.array(stats.index, dtype=float)
    known_sizes = np.array(stats["p50"], dtype=float)

    # Project forward
    sim_start_year = cfg.simulation.base_year
    sim_end_year = sim_start_year + cfg.simulation.total_years
    all_years = np.arange(sim_start_year, sim_end_year + 1)

    # Split into known and future
    future_mask = all_years > known_years[-1]
    future_years = all_years[future_mask]

    if len(future_years) > 0:
        projected = project(known_sizes, known_years, future_years, cfg.model_trends.trend)
    else:
        projected = known_sizes.copy()

    # Build a lookup: year -> model_size
    year_to_size: dict[int, float] = {}
    n_known = len(known_sizes)
    future_idx = 0
    for y in all_years:
        if y <= known_years[-1] and y in stats.index:
            year_to_size[int(y)] = float(stats.loc[int(y), "p50"])  # type: ignore[arg-type,index]
        elif y > known_years[-1] and future_idx < len(future_years):
            year_to_size[int(y)] = float(projected[n_known + future_idx])
            future_idx += 1

    # Expand to per-quarter (repeat each year's model 4 times)
    qpy = cfg.simulation.quarters_per_year
    model_sizes = []
    for yr in range(cfg.simulation.total_years):
        actual_year = sim_start_year + yr
        size = year_to_size.get(int(actual_year), known_sizes[-1])
        assert size
        for _ in range(qpy):
            model_sizes.append(max(size, 1e6))  # Floor at 1M params

    # --- Apply model size contraction scenario (Scenario 2) ---
    # At contraction_year, per-task compute intensity drops to beta * C_trend(t_r)
    # and stays flat for all t >= t_r.
    contraction = cfg.scenarios.model_size_contraction
    if contraction.enabled:
        contraction_q = contraction.contraction_year * qpy
        if 0 <= contraction_q < len(model_sizes):
            cap = contraction.beta * model_sizes[contraction_q]
            for q in range(contraction_q, len(model_sizes)):
                model_sizes[q] = cap

    return model_sizes


def _default_model_stats() -> pd.DataFrame:
    """Fallback model size stats when CSV is unavailable."""
    data = {
        2015: 1e8,    # ~100M params
        2016: 2e8,
        2017: 3e8,
        2018: 1e9,    # BERT-large era
        2019: 1.5e9,
        2020: 1.75e10,  # GPT-3
        2021: 2e10,
        2022: 5e10,
        2023: 7e10,    # Llama-2 70B
        2024: 4e11,    # DeepSeek V3 671B
        2025: 1e12,    # Llama-4 Behemoth ~2T
    }
    df = pd.DataFrame({"p50": data, "p99": data, "mean": data})
    df.index.name = "year"
    return df


# ---------------------------------------------------------------------------
# Model schedule (distillation + migration)
# ---------------------------------------------------------------------------

@dataclass
class ModelAtQuarter:
    """Model assignment for a given quarter."""
    primary_model_size: float     # Size of the currently deployed model (parameters)
    distilled_model_size: float   # Size of the distilled version
    migration_fraction: float     # Fraction of demand on full-size (new) model


def get_model_at_quarter(
    q: int,
    model_sizes: List[float],
    cfg: Config,
) -> ModelAtQuarter:
    """Determine which model and what fraction to serve at quarter *q*.

    In the first 2 quarters after a new model release (each year boundary),
    INITIAL_MIGRATION_FRACTION of demand runs on the new full model, rest
    on the previous distilled model.  After that, 100% switches to the
    new distilled model.
    """
    qpy = cfg.simulation.quarters_per_year
    distill_frac = cfg.model_trends.distilled_model_size_fraction
    migration_frac = cfg.workload.initial_migration_fraction

    # Determine current year and quarter within year
    year_idx = q // qpy
    quarter_in_year = q % qpy

    current_size = model_sizes[min(q, len(model_sizes) - 1)]

    # Previous year's model size (for distillation)
    if year_idx > 0:
        prev_q = (year_idx - 1) * qpy
        prev_size = model_sizes[min(prev_q, len(model_sizes) - 1)]
    else:
        prev_size = current_size

    if quarter_in_year < 2:
        # First 2 quarters: migrating — split between new full + old distilled
        return ModelAtQuarter(
            primary_model_size=current_size,
            distilled_model_size=prev_size * distill_frac,
            migration_fraction=migration_frac,
        )
    else:
        # Quarters 3-4: fully on distilled version of current model
        return ModelAtQuarter(
            primary_model_size=current_size * distill_frac,
            distilled_model_size=current_size * distill_frac,
            migration_fraction=1.0,
        )
