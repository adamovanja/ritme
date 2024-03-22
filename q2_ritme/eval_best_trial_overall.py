import argparse
import os

from q2_ritme.evaluate_all_experiments import (
    best_trial_name,
    compare_trials,
    get_all_exp_analyses,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-run evaluation over all experiments."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="experiments/models",
        help="Path where the models are stored.",
    )
    parser.add_argument(
        "--overall_comparison_output",
        type=str,
        default=None,
        help="Output path for the overall comparison. If not provided, it defaults to "
        "a 'compare_all' directory inside the base path.",
    )
    parser.add_argument(
        "--ls_model_types",
        type=str,
        nargs="+",
        default=["nn_reg", "nn_class", "xgb", "linreg", "rf"],
        help="List of model types to evaluate. Separate each model type with a space.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Use the provided arguments
    model_path = args.model_path
    overall_comparison_output = args.overall_comparison_output or os.path.join(
        model_path, "compare_all"
    )
    ls_model_types = args.ls_model_types

    # Ensure the overall comparison output directory exists
    os.makedirs(overall_comparison_output, exist_ok=True)

    # Find best trial over all experiments for each model type
    best_trials_overall = {}
    for model in ls_model_types:
        # read all ExperimentAnalysis objects from this directory
        experiment_dir = f"{model_path}/*/{model}"
        analyses_ls = get_all_exp_analyses(experiment_dir)

        # identify best trial from all analyses of this model type
        best_trials_overall[model] = best_trial_name(
            analyses_ls, "rmse_val", mode="min"
        )

    compare_trials(best_trials_overall, model_path, overall_comparison_output)


if __name__ == "__main__":
    main()
