from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from paymentguard.utils.visualization import apply_theme


def generate_shap_summary_plot(
    explainer: any,
    X_sample: np.ndarray,
    feature_names: list[str],
    output_path: str | Path = "reports/shap_summary.png",
    max_display: int = 15,
) -> None:
    import shap

    apply_theme()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    shap_values = explainer.shap_values(X_sample)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
