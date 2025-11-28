#!/usr/bin/env python3
"""CLI for Stage B embedding experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.stage_b_embeddings import (  # pylint: disable=wrong-import-position
    StageBEmbeddingConfig,
    run_stage_b_embedding_experiment,
    run_stage_b_embedding_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage B residual modeling with sentence-transformer embeddings.",
    )
    parser.add_argument(
        "--stage-outputs",
        default="outputs/title_lift/stage_model_outputs.parquet",
        type=Path,
        help="Path to Stage A/B combined outputs parquet (must contain residual R).",
    )
    parser.add_argument(
        "--features",
        default="data/features.parquet",
        type=Path,
        help="Feature table with created timestamp used to build temporal split.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model to load for title embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Batch size for embedding inference.",
    )
    parser.add_argument(
        "--split-quantile",
        default=0.7,
        type=float,
        help="Temporal quantile used for train/test split (match Stage A baseline).",
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="Apply PCA to embeddings before regression.",
    )
    parser.add_argument(
        "--pca-components",
        default=128,
        type=int,
        help="Number of PCA components when --use-pca is specified.",
    )
    parser.add_argument(
        "--regressor",
        default="ridge",
        choices=["ridge", "elasticnet", "mlp"],
        help="Regressor to fit on embeddings.",
    )
    parser.add_argument(
        "--ridge-alphas",
        default="0.1,1.0,10.0",
        help="Comma-separated Ridge alpha values (if regressor=ridge).",
    )
    parser.add_argument(
        "--elasticnet-l1-ratios",
        default="0.1,0.5,0.9",
        help="Comma-separated l1_ratio values (if regressor=elasticnet).",
    )
    parser.add_argument(
        "--elasticnet-alphas",
        default="0.01,0.1,1.0",
        help="Comma-separated alpha grid (if regressor=elasticnet).",
    )
    parser.add_argument(
        "--mlp-hidden",
        default="256,128",
        help="Comma-separated hidden layer sizes (if regressor=mlp).",
    )
    parser.add_argument(
        "--mlp-max-iter",
        default=400,
        type=int,
        help="Maximum training iterations for MLP regressor.",
    )
    parser.add_argument(
        "--random-state",
        default=42,
        type=int,
        help="Random seed for PCA/MLP/ElasticNet.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/title_lift/stage_b_embeddings.json",
        type=Path,
        help="Where to write experiment metrics as JSON.",
    )
    parser.add_argument(
        "--predictions-output",
        default="outputs/title_lift/stage_b_embeddings_predictions.parquet",
        type=Path,
        help="Optional parquet with test predictions (post_id, actual, pred).",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Run a hyperparameter grid search instead of a single configuration.",
    )
    parser.add_argument(
        "--grid-model-names",
        help="Comma-separated list of sentence-transformer models when using --search.",
    )
    parser.add_argument(
        "--grid-regressors",
        help="Comma-separated list of regressors when using --search.",
    )
    parser.add_argument(
        "--grid-use-pca",
        help="Comma-separated booleans (true,false) controlling PCA usage during --search.",
    )
    parser.add_argument(
        "--grid-pca-components",
        help="Comma-separated PCA component counts evaluated during --search.",
    )
    parser.add_argument(
        "--grid-batch-sizes",
        help="Comma-separated embedding batch sizes evaluated during --search.",
    )
    parser.add_argument(
        "--grid-split-quantiles",
        help="Comma-separated temporal split quantiles evaluated during --search.",
    )
    parser.add_argument(
        "--grid-mlp-hidden",
        help=(
            "Semicolon-separated hidden layer definitions for MLP (each definition is comma-separated). "
            "Used when --search and regressor=mlp appears in the grid."
        ),
    )
    parser.add_argument(
        "--grid-mlp-max-iters",
        help="Comma-separated MLP max_iter values when using --search.",
    )
    parser.add_argument(
        "--grid-ridge-alpha-sets",
        help="Semicolon-separated Ridge alpha grids (comma-separated floats) for --search.",
    )
    parser.add_argument(
        "--grid-elasticnet-alpha-sets",
        help="Semicolon-separated ElasticNet alpha grids (comma-separated floats) for --search.",
    )
    parser.add_argument(
        "--grid-elasticnet-l1-sets",
        help="Semicolon-separated ElasticNet l1_ratio grids (comma-separated floats) for --search.",
    )
    parser.add_argument(
        "--grid-random-states",
        help="Comma-separated random seeds when using --search.",
    )
    parser.add_argument(
        "--scoring",
        default="pairwise_accuracy",
        choices=[
            "pairwise_accuracy",
            "test_rmse",
            "train_rmse",
            "test_r2",
            "train_r2",
            "test_mae",
            "train_mae",
        ],
        help="Primary metric to maximise when --search is enabled.",
    )
    parser.add_argument(
        "--search-output-dir",
        default=Path("outputs/title_lift/stage_b_embedding_search"),
        type=Path,
        help="Directory to write per-configuration metrics during --search.",
    )
    parser.add_argument(
        "--search-predictions-dir",
        type=Path,
        help="Optional directory to store per-configuration predictions during --search.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the tqdm progress bar during --search runs.",
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        help="Optional log file to append per-configuration progress during --search.",
    )
    return parser.parse_args()


def _parse_floats(payload: str) -> tuple[float, ...]:
    items = [item.strip() for item in payload.split(",") if item.strip()]
    return tuple(float(item) for item in items)


def _parse_ints(payload: str) -> tuple[int, ...]:
    items = [item.strip() for item in payload.split(",") if item.strip()]
    return tuple(int(item) for item in items)


def _parse_strings(payload: str) -> tuple[str, ...]:
    items = [item.strip() for item in payload.split(",") if item.strip()]
    return tuple(items)


def _parse_bools(payload: str) -> tuple[bool, ...]:
    truth_map = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    items = [item.strip().lower() for item in payload.split(",") if item.strip()]
    parsed: list[bool] = []
    for item in items:
        if item not in truth_map:
            raise ValueError(f"Cannot parse boolean value from '{item}'")
        parsed.append(truth_map[item])
    return tuple(parsed)


def _parse_float_tuples(payload: str) -> tuple[tuple[float, ...], ...]:
    groups = [chunk.strip() for chunk in payload.split(";") if chunk.strip()]
    parsed: list[tuple[float, ...]] = []
    for group in groups:
        values = _parse_floats(group)
        if values:
            parsed.append(values)
    return tuple(parsed)


def _parse_int_tuples(payload: str) -> tuple[tuple[int, ...], ...]:
    groups = [chunk.strip() for chunk in payload.split(";") if chunk.strip()]
    parsed: list[tuple[int, ...]] = []
    for group in groups:
        values = _parse_ints(group)
        if values:
            parsed.append(values)
    return tuple(parsed)


def _expand_search_configs(
    base_config: StageBEmbeddingConfig,
    args: argparse.Namespace,
) -> tuple[list[StageBEmbeddingConfig], dict[str, object]]:
    model_grid = (
        _parse_strings(args.grid_model_names)
        if args.grid_model_names
        else (base_config.model_name,)
    )
    regressor_grid = (
        _parse_strings(args.grid_regressors)
        if args.grid_regressors
        else (base_config.regressor,)
    )
    use_pca_grid = (
        _parse_bools(args.grid_use_pca)
        if args.grid_use_pca
        else (base_config.use_pca,)
    )
    pca_components_grid = (
        _parse_ints(args.grid_pca_components)
        if args.grid_pca_components
        else (base_config.pca_components,)
    )
    batch_size_grid = (
        _parse_ints(args.grid_batch_sizes)
        if args.grid_batch_sizes
        else (base_config.batch_size,)
    )
    split_quantile_grid = (
        _parse_floats(args.grid_split_quantiles)
        if args.grid_split_quantiles
        else (base_config.split_quantile,)
    )
    random_state_grid = (
        _parse_ints(args.grid_random_states)
        if args.grid_random_states
        else (base_config.random_state,)
    )
    ridge_alpha_sets = (
        _parse_float_tuples(args.grid_ridge_alpha_sets)
        if args.grid_ridge_alpha_sets
        else (tuple(base_config.ridge_alphas),)
    )
    elasticnet_alpha_sets = (
        _parse_float_tuples(args.grid_elasticnet_alpha_sets)
        if args.grid_elasticnet_alpha_sets
        else (tuple(base_config.elasticnet_alphas),)
    )
    elasticnet_l1_sets = (
        _parse_float_tuples(args.grid_elasticnet_l1_sets)
        if args.grid_elasticnet_l1_sets
        else (tuple(base_config.elasticnet_l1_ratios),)
    )
    mlp_hidden_grid = (
        _parse_int_tuples(args.grid_mlp_hidden)
        if args.grid_mlp_hidden
        else (tuple(base_config.mlp_hidden_layers),)
    )
    mlp_max_iter_grid = (
        _parse_ints(args.grid_mlp_max_iters)
        if args.grid_mlp_max_iters
        else (base_config.mlp_max_iter,)
    )

    configs: list[StageBEmbeddingConfig] = []
    for model_name in model_grid:
        for use_pca in use_pca_grid:
            if use_pca:
                pca_candidates = pca_components_grid
            else:
                pca_candidates = (base_config.pca_components,)
            for pca_components in pca_candidates:
                for batch_size in batch_size_grid:
                    for split_quantile in split_quantile_grid:
                        for random_state in random_state_grid:
                            for regressor in regressor_grid:
                                reg_lower = regressor.lower()
                                if reg_lower == "ridge":
                                    for ridge_alphas in ridge_alpha_sets:
                                        configs.append(
                                            StageBEmbeddingConfig(
                                                split_quantile=split_quantile,
                                                model_name=model_name,
                                                batch_size=batch_size,
                                                use_pca=use_pca,
                                                pca_components=pca_components,
                                                regressor="ridge",
                                                ridge_alphas=tuple(ridge_alphas),
                                                elasticnet_l1_ratios=tuple(
                                                    base_config.elasticnet_l1_ratios
                                                ),
                                                elasticnet_alphas=tuple(
                                                    base_config.elasticnet_alphas
                                                ),
                                                mlp_hidden_layers=tuple(
                                                    base_config.mlp_hidden_layers
                                                ),
                                                mlp_max_iter=base_config.mlp_max_iter,
                                                random_state=random_state,
                                            )
                                        )
                                elif reg_lower == "elasticnet":
                                    for en_alphas in elasticnet_alpha_sets:
                                        for en_l1 in elasticnet_l1_sets:
                                            configs.append(
                                                StageBEmbeddingConfig(
                                                    split_quantile=split_quantile,
                                                    model_name=model_name,
                                                    batch_size=batch_size,
                                                    use_pca=use_pca,
                                                    pca_components=pca_components,
                                                    regressor="elasticnet",
                                                    ridge_alphas=tuple(
                                                        base_config.ridge_alphas
                                                    ),
                                                    elasticnet_l1_ratios=tuple(en_l1),
                                                    elasticnet_alphas=tuple(en_alphas),
                                                    mlp_hidden_layers=tuple(
                                                        base_config.mlp_hidden_layers
                                                    ),
                                                    mlp_max_iter=base_config.mlp_max_iter,
                                                    random_state=random_state,
                                                )
                                            )
                                elif reg_lower == "mlp":
                                    for hidden_layers in mlp_hidden_grid:
                                        for mlp_max_iter in mlp_max_iter_grid:
                                            configs.append(
                                                StageBEmbeddingConfig(
                                                    split_quantile=split_quantile,
                                                    model_name=model_name,
                                                    batch_size=batch_size,
                                                    use_pca=use_pca,
                                                    pca_components=pca_components,
                                                    regressor="mlp",
                                                    ridge_alphas=tuple(
                                                        base_config.ridge_alphas
                                                    ),
                                                    elasticnet_l1_ratios=tuple(
                                                        base_config.elasticnet_l1_ratios
                                                    ),
                                                    elasticnet_alphas=tuple(
                                                        base_config.elasticnet_alphas
                                                    ),
                                                    mlp_hidden_layers=tuple(hidden_layers),
                                                    mlp_max_iter=mlp_max_iter,
                                                    random_state=random_state,
                                                )
                                            )
                                else:  # pragma: no cover - validated by argparse choices
                                    raise ValueError(
                                        f"Unsupported regressor in grid: {regressor}"
                                    )

    if not configs:
        raise ValueError("No configurations generated for embedding grid search")

    grid_spec = {
        "model_names": list(model_grid),
        "regressors": list(regressor_grid),
        "use_pca": list(use_pca_grid),
        "pca_components": list(pca_components_grid),
        "batch_sizes": list(batch_size_grid),
        "split_quantiles": list(split_quantile_grid),
        "random_states": list(random_state_grid),
        "ridge_alpha_sets": [list(seq) for seq in ridge_alpha_sets],
        "elasticnet_alpha_sets": [list(seq) for seq in elasticnet_alpha_sets],
        "elasticnet_l1_sets": [list(seq) for seq in elasticnet_l1_sets],
        "mlp_hidden_layers": [list(seq) for seq in mlp_hidden_grid],
        "mlp_max_iters": list(mlp_max_iter_grid),
    }

    return configs, grid_spec


def main() -> None:
    args = parse_args()

    base_config = StageBEmbeddingConfig(
        split_quantile=args.split_quantile,
        model_name=args.model_name,
        batch_size=args.batch_size,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        regressor=args.regressor,
        ridge_alphas=_parse_floats(args.ridge_alphas) or (1.0,),
        elasticnet_l1_ratios=_parse_floats(args.elasticnet_l1_ratios) or (0.5,),
        elasticnet_alphas=_parse_floats(args.elasticnet_alphas) or (0.1,),
        mlp_hidden_layers=_parse_ints(args.mlp_hidden) or (256, 128),
        mlp_max_iter=args.mlp_max_iter,
        random_state=args.random_state,
    )

    if args.search:
        configs, grid_spec = _expand_search_configs(base_config, args)
        predictions_dir = (
            args.search_predictions_dir
            if args.search_predictions_dir is not None
            else args.search_output_dir / "predictions"
        )
        progress = not args.no_progress_bar
        log_path = args.progress_log
        if log_path is None:
            log_path = args.search_output_dir / "grid_run.log"
        search_result = run_stage_b_embedding_grid(
            stage_outputs_path=args.stage_outputs,
            features_path=args.features,
            configs=configs,
            output_dir=args.search_output_dir,
            predictions_dir=predictions_dir,
            scoring=args.scoring,
            grid_spec=grid_spec,
            progress=progress,
            log_path=log_path,
        )

        print("Stage B embedding grid search complete.")
        print(f"  Scoring metric: {search_result.scoring}")
        print(f"  Best score:     {search_result.best_score:.3f}")
        best_cfg = search_result.best_result.config
        print(f"  Best model:     {best_cfg.get('model_name')}")
        print(f"  Best regressor: {best_cfg.get('regressor')}")
        print(
            f"  Pairwise accuracy: {search_result.best_result.pairwise_accuracy:.3f}"
            if search_result.best_result.pairwise_accuracy is not None
            else "  Pairwise accuracy: n/a"
        )
        print(f"  Test RMSE:      {search_result.best_result.test_rmse:.3f}")
        print(f"  Test R^2:       {search_result.best_result.test_r2:.3f}")
        print(f"  Train RMSE:     {search_result.best_result.train_rmse:.3f}")
        print(f"  Train R^2:      {search_result.best_result.train_r2:.3f}")
        print(f"  Configurations evaluated: {len(search_result.ranked_results)}")
        if log_path is not None:
            print(f"  Progress log:   {log_path}")
    else:
        result = run_stage_b_embedding_experiment(
            stage_outputs_path=args.stage_outputs,
            features_path=args.features,
            output_path=args.output_json,
            predictions_path=args.predictions_output,
            config=base_config,
        )

        print("Stage B embedding metrics:")
        print(f"  Train RMSE: {result.train_rmse:.3f}")
        print(f"  Test RMSE:  {result.test_rmse:.3f}")
        print(f"  Train R^2:  {result.train_r2:.3f}")
        print(f"  Test R^2:   {result.test_r2:.3f}")
        print(
            f"  Pairwise accuracy: {result.pairwise_accuracy:.3f}"
            if result.pairwise_accuracy is not None
            else "  Pairwise accuracy: n/a"
        )
        print(
            f"  Pairwise comparisons: {result.pairwise_pairs:,}"
            if result.pairwise_pairs is not None
            else "  Pairwise comparisons: n/a"
        )
        print(f"  Train rows: {result.n_train}")
        print(f"  Test rows:  {result.n_test}")
        print(f"  Split time: {result.split_time}")


if __name__ == "__main__":
    main()
