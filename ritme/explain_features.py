import os
import pickle
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import typer
import xgboost as xgb
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline

from ritme._decorators import helper_function, main_function
from ritme.evaluate_models import TunedModel, load_best_model
from ritme.model_space.static_trainables import NeuralNet

plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("seaborn-v0_8-pastel")


@helper_function
def _get_predict_fn(tmodel: TunedModel):
    """Return a callable f(X_numpy) -> 1-D predictions for the underlying model."""
    model = tmodel.model
    if isinstance(model, NeuralNet):
        device = next(model.parameters()).device

        def _predict(X: np.ndarray) -> np.ndarray:
            model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float32).to(device)
                out = model(X_t)
                out = model._prepare_predictions(out)
                return out.detach().cpu().numpy().flatten()

        return _predict
    if isinstance(model, xgb.core.Booster):
        return lambda X: model.predict(xgb.DMatrix(X)).flatten()
    # sklearn-like (linreg, rf)
    return lambda X: model.predict(X).flatten()


@helper_function
def _classification_class_names(tmodel: TunedModel) -> Optional[List[str]]:
    """Return original-target-space class labels for a classification
    ``TunedModel``, or ``None`` for non-classification models."""
    model = tmodel.model
    if isinstance(model, dict):  # TRAC
        return None
    if isinstance(model, NeuralNet):
        if model.nn_type not in ("classification", "ordinal_regression"):
            return None
        classes = list(model.classes)
    elif isinstance(model, xgb.core.Booster):
        if tmodel.model_type != "xgb_class" or tmodel.label_encoder is None:
            return None
        return [str(c) for c in tmodel.label_encoder.classes_]
    else:
        if not hasattr(model, "classes_"):
            return None
        classes = list(model.classes_)
    if tmodel.label_encoder is not None:
        classes = list(
            tmodel.label_encoder.inverse_transform(np.asarray(classes).astype(int))
        )
    return [str(c) for c in classes]


@helper_function
def _build_explainer(tmodel: TunedModel, X_background: pd.DataFrame) -> shap.Explainer:
    """Build a SHAP explainer appropriate for the underlying model type."""
    model = tmodel.model
    if isinstance(model, dict):
        raise TypeError(
            "SHAP explanation is not supported for TRAC models. "
            "TRAC coefficients already provide direct feature importance."
        )
    if isinstance(model, xgb.core.Booster):
        return shap.TreeExplainer(model)
    if isinstance(model, NeuralNet):
        predict_fn = _get_predict_fn(tmodel)
        return shap.KernelExplainer(predict_fn, X_background.values)
    # sklearn tree-based or linear models
    try:
        return shap.Explainer(model, X_background)
    except Exception:
        predict_fn = _get_predict_fn(tmodel)
        return shap.KernelExplainer(predict_fn, X_background.values)


@helper_function
def _is_trac_model(model) -> bool:
    return isinstance(model, dict) and "matrix_a" in model and "model" in model


@helper_function
def _is_coef_model(model) -> bool:
    """Return True for trainables whose feature importance is fully captured
    by their fitted coefficients (TRAC dict or sklearn ``Pipeline`` whose
    final step has a ``coef_`` attribute)."""
    if _is_trac_model(model):
        return True
    if isinstance(model, Pipeline):
        final = model.steps[-1][1]
        return hasattr(final, "coef_")
    return False


@helper_function
def _extract_coefficients(tmodel: TunedModel, feature_names: List[str]) -> pd.DataFrame:
    """Return a long-form per-feature coefficient table for a
    coefficient-bearing model.

    For binary / regression linear models the returned DataFrame has columns
    ``feature``, ``coefficient``, ``abs_coefficient``. For multi-class
    classifiers a ``class`` column is added and rows are repeated per class.
    For TRAC models the labels come from the ``alpha`` DataFrame's index
    (the log-contrast names installed by ``_bundle_trac_model``, with the
    ``intercept`` row dropped) rather than from ``feature_names``.
    """
    model = tmodel.model
    if _is_trac_model(model):
        alpha = model["model"]["alpha"]
        if "intercept" in alpha.index:
            alpha = alpha.drop("intercept")
        values = alpha.values.astype(float)
        return pd.DataFrame(
            {
                "feature": alpha.index.tolist(),
                "coefficient": values,
                "abs_coefficient": np.abs(values),
            }
        )

    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
    coef = np.asarray(estimator.coef_)
    # sklearn LogisticRegression stores binary coefficients as shape (1, F).
    # Collapse the singleton class axis so binary classifiers produce the
    # same one-row-per-feature table as a regressor.
    if coef.ndim == 2 and coef.shape[0] == 1:
        coef = coef.ravel()

    if coef.ndim == 1:
        if len(coef) != len(feature_names):
            raise ValueError(
                f"Number of coefficients ({len(coef)}) does not match number "
                f"of feature names ({len(feature_names)})."
            )
        return pd.DataFrame(
            {
                "feature": list(feature_names),
                "coefficient": coef,
                "abs_coefficient": np.abs(coef),
            }
        )

    # Multi-class: one row per (class, feature).
    if coef.shape[1] != len(feature_names):
        raise ValueError(
            f"Number of coefficient columns ({coef.shape[1]}) does not match "
            f"number of feature names ({len(feature_names)})."
        )
    class_names = _classification_class_names(tmodel)
    if class_names is None or len(class_names) != coef.shape[0]:
        class_names = [f"class_{i}" for i in range(coef.shape[0])]
    records = []
    for ci, cls in enumerate(class_names):
        for fi, fname in enumerate(feature_names):
            records.append(
                {
                    "feature": fname,
                    "class": cls,
                    "coefficient": float(coef[ci, fi]),
                    "abs_coefficient": float(abs(coef[ci, fi])),
                }
            )
    return pd.DataFrame.from_records(records)


@main_function
def compute_feature_importance(
    tmodel: TunedModel,
    train_val: pd.DataFrame,
) -> pd.DataFrame:
    """Return a per-feature coefficient table for coefficient-bearing
    trainables (``linreg``, ``logreg``, ``trac``).

    Args:
        tmodel: A fitted TunedModel whose underlying model exposes
            coefficients directly (sklearn ``Pipeline`` with a final step
            that has ``coef_``, or a TRAC dict with ``"model"`` / ``"matrix_a"``).
        train_val: Training data used to recover the design-matrix column
            order (and hence the feature labels) the model was fit on.

    Returns:
        A long-form DataFrame with columns ``feature``, ``coefficient``,
        ``abs_coefficient`` (plus ``class`` for multi-class classifiers).

    Raises:
        TypeError: If ``tmodel.model`` is not a coefficient-bearing model.
    """
    if not _is_coef_model(tmodel.model):
        raise TypeError(
            "compute_feature_importance only supports coefficient-bearing "
            "models (linreg / logreg / trac). Use compute_shap_values for "
            "tree- and neural-network-based trainables."
        )
    X_train = tmodel.build_design_matrix(train_val, split="train")
    return _extract_coefficients(tmodel, X_train.columns.tolist())


@helper_function
def _draw_importance_bar(ax, sub: pd.DataFrame, max_display: int) -> None:
    """Render a horizontal bar of the ``max_display`` most-important features
    onto ``ax``, ordered with the largest ``|coefficient|`` at the top."""
    top = (
        sub.sort_values("abs_coefficient", ascending=False)
        .head(max_display)
        .sort_values("abs_coefficient", ascending=True)
    )
    ax.barh(top["feature"], top["coefficient"])
    ax.set_xlabel("coefficient")


@main_function
def plot_feature_importance_bar(
    importance: pd.DataFrame,
    max_display: int = 20,
    show: bool = True,
) -> Optional[Figure]:
    """Bar plot of feature importances ranked by ``abs_coefficient``.

    Args:
        importance: DataFrame produced by :func:`compute_feature_importance`.
        max_display: Maximum number of features to display — per class for
            multi-class importances, otherwise overall.
        show: When True, render via ``plt.show()`` and return ``None``.
            When False, return the Figure for further manipulation.

    Returns:
        The matplotlib Figure, or ``None`` when ``show=True``.
    """
    if "class" in importance.columns:
        class_names = importance["class"].unique().tolist()
        per_class_height = max(4, max_display * 0.35)
        fig, axes = plt.subplots(
            len(class_names), 1, figsize=(10, per_class_height * len(class_names))
        )
        if len(class_names) == 1:
            axes = [axes]
        for ax, name in zip(axes, class_names):
            _draw_importance_bar(
                ax, importance[importance["class"] == name], max_display
            )
            ax.set_title(f"Class: {name}")
        plt.tight_layout()
        if show:
            plt.show()
            plt.close(fig)
            return None
        return fig

    fig, ax = plt.subplots(1, 1, figsize=(10, max(4, max_display * 0.35)))
    _draw_importance_bar(ax, importance, max_display)
    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


@main_function
def compute_shap_values(
    tmodel: TunedModel,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    max_background_samples: int | None = None,
) -> shap.Explanation:
    """Compute SHAP values for the test set using a single trained model.

    Args:
        tmodel: A fitted TunedModel (must have been run on train split first).
        train_val: Training data used as SHAP background.
        test: Test data on which SHAP values are computed.
        max_background_samples: If set, subsample the background to this many
            rows (useful to speed up KernelExplainer for large datasets).
            By default the full training set is used.

    Returns:
        shap.Explanation with .values, .base_values, .data, and .feature_names.
    """
    X_train = tmodel.build_design_matrix(train_val, split="train")
    X_test = tmodel.build_design_matrix(test, split="test")

    if max_background_samples is not None and max_background_samples < X_train.shape[0]:
        X_bg = shap.sample(X_train, max_background_samples)
    else:
        X_bg = X_train

    explainer = _build_explainer(tmodel, X_bg)

    model = tmodel.model
    if isinstance(model, xgb.core.Booster):
        sv = explainer.shap_values(xgb.DMatrix(X_test))
        explanation = shap.Explanation(
            values=sv,
            base_values=np.atleast_1d(explainer.expected_value),
            data=X_test.values,
            feature_names=X_test.columns.tolist(),
        )
    elif isinstance(model, NeuralNet):
        sv = explainer.shap_values(X_test.values)
        explanation = shap.Explanation(
            values=np.array(sv),
            base_values=np.atleast_1d(explainer.expected_value),
            data=X_test.values,
            feature_names=X_test.columns.tolist(),
        )
    else:
        explanation = explainer(X_test)

    if explanation.values.ndim == 3:
        class_names = _classification_class_names(tmodel)
        if class_names is not None and len(class_names) == explanation.values.shape[2]:
            explanation.output_names = class_names

    return explanation


@main_function
def plot_shap_summary(
    shap_values: shap.Explanation,
    max_display: int = 20,
    plot_type: str = "dot",
    show: bool = True,
) -> Optional[Figure]:
    """Create a SHAP summary plot showing per-sample SHAP value distributions.

    For multi-class explanations (3-D values), one summary plot is rendered
    per class in a vertically stacked figure, using ``plot_type``.

    Args:
        shap_values: SHAP Explanation object from ``compute_shap_values``.
        max_display: Maximum number of features to display.
        plot_type: One of "dot" (beeswarm, default), "bar", "violin".
        show: When True, render via ``plt.show()`` and return ``None`` (so
            Jupyter does not auto-render the return value, causing a duplicate
            plot). When False, return the Figure for further manipulation.

    Returns:
        The matplotlib Figure, or ``None`` when ``show=True``.
    """
    values = shap_values.values
    if values.ndim == 3:
        n_classes = values.shape[2]
        class_names = (
            list(shap_values.output_names)
            if getattr(shap_values, "output_names", None) is not None
            else [f"class_{i}" for i in range(n_classes)]
        )
        per_class_height = max(6, max_display * 0.35)
        fig, axes = plt.subplots(
            n_classes, 1, figsize=(10, per_class_height * n_classes)
        )
        if n_classes == 1:
            axes = [axes]
        for i, (ax, name) in enumerate(zip(axes, class_names)):
            plt.sca(ax)
            shap.summary_plot(
                values[:, :, i],
                shap_values.data,
                feature_names=shap_values.feature_names,
                max_display=max_display,
                plot_type=plot_type,
                show=False,
            )
            ax.set_title(f"Class: {name}")
        # shap.summary_plot resizes the active figure; restore so all subplots
        # actually get their intended height.
        fig.set_size_inches(10, per_class_height * n_classes)
        plt.tight_layout()
        if show:
            plt.show()
            plt.close(fig)
            return None
        return fig

    fig, ax = plt.subplots(1, 1, figsize=(10, max(6, max_display * 0.35)))
    plt.sca(ax)
    shap.summary_plot(
        values,
        shap_values.data,
        feature_names=shap_values.feature_names,
        max_display=max_display,
        plot_type=plot_type,
        show=False,
    )
    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


@main_function
def plot_shap_bar(
    shap_values: shap.Explanation,
    max_display: int = 20,
    show: bool = True,
) -> Optional[Figure]:
    """Create a SHAP bar plot of mean absolute SHAP values.

    For multi-class explanations (3-D values), one bar plot is rendered per
    class in a vertically stacked figure.

    Args:
        shap_values: SHAP Explanation object from ``compute_shap_values``.
        max_display: Maximum number of features to display.
        show: When True, render via ``plt.show()`` and return ``None`` (so
            Jupyter does not auto-render the return value, causing a duplicate
            plot). When False, return the Figure for further manipulation.

    Returns:
        The matplotlib Figure, or ``None`` when ``show=True``.
    """
    values = shap_values.values
    if values.ndim == 3:
        n_classes = values.shape[2]
        class_names = (
            list(shap_values.output_names)
            if getattr(shap_values, "output_names", None) is not None
            else [f"class_{i}" for i in range(n_classes)]
        )
        per_class_height = max(6, max_display * 0.35)
        fig, axes = plt.subplots(
            n_classes, 1, figsize=(10, per_class_height * n_classes)
        )
        if n_classes == 1:
            axes = [axes]
        base = shap_values.base_values
        for i, (ax, name) in enumerate(zip(axes, class_names)):
            plt.sca(ax)
            base_i = (
                base[:, i]
                if (base is not None and np.asarray(base).ndim == 2)
                else base
            )
            sub = shap.Explanation(
                values=values[:, :, i],
                base_values=base_i,
                data=shap_values.data,
                feature_names=shap_values.feature_names,
            )
            shap.plots.bar(sub, max_display=max_display, show=False)
            ax.set_title(f"Class: {name}")
            ax.set_xlabel("mean(|SHAP value|)" if i == n_classes - 1 else "")
        # shap.plots.bar resizes the active figure; restore so all subplots
        # actually get their intended height.
        fig.set_size_inches(10, per_class_height * n_classes)
        plt.tight_layout()
        if show:
            plt.show()
            plt.close(fig)
            return None
        return fig

    fig, ax = plt.subplots(1, 1, figsize=(10, max(6, max_display * 0.35)))
    plt.sca(ax)
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# ----------------------------------------------------------------------------
@main_function
def cli_explain_features(
    path_to_exp: str,
    model_type: str,
    path_to_train_val: str,
    path_to_test: str,
    max_display: int = 20,
    max_background_samples: Optional[int] = None,
) -> None:
    """Compute feature importance for a single best tuned model.

    For coefficient-bearing trainables (``linreg``, ``logreg``, ``trac``) the
    coefficients themselves are emitted; SHAP is skipped entirely. For
    tree- and neural-network-based trainables SHAP is computed as before.

    Args:
        path_to_exp: Path to the experiment directory containing
            ``<model_type>_best_model.pkl``.
        model_type: Model type to explain (e.g. "linreg", "rf", "xgb").
        path_to_train_val: Path to the pickled train/validation DataFrame.
        path_to_test: Path to the pickled test DataFrame.
        max_display: Maximum number of features shown in plots.
        max_background_samples: If set, subsample the SHAP background to this
            many rows. By default the full training set is used. Ignored for
            coefficient-bearing models.

    Side Effects:
        For coefficient-bearing models, writes into path_to_exp:
            ``feature_importance_<model_type>.csv``,
            ``feature_importance_bar_<model_type>.png``.
        Otherwise writes:
            ``shap_values_<model_type>.pkl``,
            ``shap_summary_<model_type>.png``,
            ``shap_bar_<model_type>.png``.
    """
    train_val = pd.read_pickle(path_to_train_val)
    test = pd.read_pickle(path_to_test)

    tmodel = load_best_model(model_type, path_to_exp)

    if _is_coef_model(tmodel.model):
        importance = compute_feature_importance(tmodel, train_val)
        csv_path = os.path.join(path_to_exp, f"feature_importance_{model_type}.csv")
        importance.to_csv(csv_path, index=False)
        print(f"Feature importance (coefficients) saved in {csv_path}.")

        fig_bar = plot_feature_importance_bar(
            importance, max_display=max_display, show=False
        )
        assert fig_bar is not None  # show=False guarantees a Figure
        bar_path = os.path.join(path_to_exp, f"feature_importance_bar_{model_type}.png")
        fig_bar.savefig(bar_path, bbox_inches="tight")
        plt.close(fig_bar)
        print(f"Feature importance bar plot saved in {bar_path}.")
        return

    explanation = compute_shap_values(
        tmodel,
        train_val,
        test,
        max_background_samples=max_background_samples,
    )

    sv_path = os.path.join(path_to_exp, f"shap_values_{model_type}.pkl")
    with open(sv_path, "wb") as f:
        pickle.dump(explanation, f)
    print(f"SHAP values saved in {sv_path}.")

    fig_summary = plot_shap_summary(
        explanation,
        max_display=max_display,
        show=False,
    )
    assert fig_summary is not None  # show=False guarantees a Figure
    summary_path = os.path.join(path_to_exp, f"shap_summary_{model_type}.png")
    fig_summary.savefig(summary_path, bbox_inches="tight")
    plt.close(fig_summary)
    print(f"SHAP summary plot saved in {summary_path}.")

    fig_bar = plot_shap_bar(
        explanation,
        max_display=max_display,
        show=False,
    )
    assert fig_bar is not None  # show=False guarantees a Figure
    bar_path = os.path.join(path_to_exp, f"shap_bar_{model_type}.png")
    fig_bar.savefig(bar_path, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"SHAP bar plot saved in {bar_path}.")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    typer.run(cli_explain_features)
