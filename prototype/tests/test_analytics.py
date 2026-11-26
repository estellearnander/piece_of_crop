import numpy as np
import pandas as pd

from analytics import metrics_table, read_verified_predictions, summary_table
from piece_of_crop.evaluate import insurer_cost_matrix


def test_saved_runs_require_identical_split(tmp_path):
    results = tmp_path / "results"
    dest = results / "predictions"
    dest.mkdir(parents=True)
    pd.DataFrame({"file":["a.tif", "b.tif"], "class_name":["AnnualCrop","Forest"],
                  "label":[0,1], "split":["test","test"]}).to_csv(results / "split.csv", index=False)
    base = {"file":["a.tif","b.tif"], "y_true":[0,1], "y_pred":[0,1]}
    base.update({f"p_{i}":[1.0 if i == 0 else 0.,1.0 if i == 1 else 0.] for i in range(10)})
    pd.DataFrame(base).to_csv(dest/"rf_rgb_seed0.csv", index=False)
    df_wrong = pd.DataFrame(base)
    df_wrong["file"] = ["z.tif","b.tif"]
    df_wrong.to_csv(dest/"rf_13band_seed0.csv", index=False)
    runs, warnings = read_verified_predictions(tmp_path)
    assert len(runs) == 1
    assert any("Split mismatch" in x for x in warnings)
    metrics = metrics_table(runs, insurer_cost_matrix())
    assert metrics.iloc[0].accuracy == 1
    assert metrics.iloc[0].mean_severity == 0
    assert len(summary_table(metrics)) == 1
