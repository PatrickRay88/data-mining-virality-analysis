"""Generate Stage A/B diagnostics artifacts.

Run with:
    python -m src.cli.run_model_diagnostics --data data/features.parquet
"""

from pathlib import Path

import click

from src.models.model_diagnostics import generate_diagnostics_report


def _parse_quantiles(value: str) -> tuple[float, ...]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(float(item) for item in parts)


def _parse_fractions(value: str) -> tuple[float, ...]:
    return _parse_quantiles(value)


@click.command()
@click.option(
    "--data",
    default="data/features.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Parquet file with engineered features",
)
@click.option(
    "--output",
    default="docs/stage_model_diagnostics.json",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write diagnostics JSON",
)
@click.option(
    "--temporal-quantiles",
    default="0.6,0.65,0.7,0.75,0.8",
    help="Comma-separated quantiles for temporal splits",
)
@click.option(
    "--bootstrap-iterations",
    default=50,
    show_default=True,
    help="Number of bootstrap resamples",
)
@click.option(
    "--bootstrap-trim-fraction",
    default=0.1,
    show_default=True,
    help="Trim fraction applied when summarizing bootstrap metrics",
)
@click.option(
    "--bootstrap-max-abs-skew",
    default=10.0,
    show_default=True,
    help="Skip bootstrap samples whose residual skew exceeds this absolute value",
)
@click.option(
    "--learning-quantile",
    default=0.7,
    show_default=True,
    help="Base quantile for learning curve split",
)
@click.option(
    "--learning-fractions",
    default="0.3,0.5,0.7,1.0",
    help="Comma-separated fractions for learning curve",
)
@click.option(
    "--blocked-min-train",
    default=400,
    show_default=True,
    help="Minimum training examples required for a blocked CV fold",
)
@click.option(
    "--blocked-min-test",
    default=200,
    show_default=True,
    help="Minimum test examples required for a blocked CV fold",
)
@click.option(
    "--blocked-min-target-std",
    default=0.01,
    show_default=True,
    help="Minimum standard deviation of stage targets required for a blocked CV fold",
)
@click.option(
    "--blocked-min-stage-b-rmse",
    default=0.05,
    show_default=True,
    help="Minimum Stage B RMSE required for a blocked CV fold (filters degenerate residual splits)",
)
@click.option(
    "--tables-dir",
    default="docs/diagnostics",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write tabular diagnostics artifacts",
)
@click.option(
    "--figures-dir",
    default="docs/figures/diagnostics",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write diagnostics plots",
)
def main(
    data: Path,
    output: Path,
    temporal_quantiles: str,
    bootstrap_iterations: int,
    bootstrap_trim_fraction: float,
    bootstrap_max_abs_skew: float,
    learning_quantile: float,
    learning_fractions: str,
    blocked_min_train: int,
    blocked_min_test: int,
    blocked_min_target_std: float,
    blocked_min_stage_b_rmse: float,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    quantiles = _parse_quantiles(temporal_quantiles)
    fractions = _parse_fractions(learning_fractions)
    report = generate_diagnostics_report(
        data_path=data,
        output_path=output,
        temporal_quantiles=quantiles,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_trim_fraction=bootstrap_trim_fraction,
        bootstrap_max_abs_skew=bootstrap_max_abs_skew,
        learning_curve_quantile=learning_quantile,
        learning_curve_fractions=fractions,
        blocked_min_train=blocked_min_train,
        blocked_min_test=blocked_min_test,
        blocked_min_target_std=blocked_min_target_std,
        blocked_min_stage_b_rmse=blocked_min_stage_b_rmse,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )
    click.echo(f"Diagnostics written to {output}")
    click.echo(f"Temporal splits evaluated: {len(report['temporal_splits'])}")
    click.echo(f"Blocked folds evaluated: {len(report['blocked_cross_validation'])}")
    click.echo(f"Bootstrap iterations captured: {len(report['bootstrap']['records'])}")


if __name__ == "__main__":
    main()
