
from pathlib import Path
from run import plot_metric_vs_param, _pretty_metric
import results_io

## --- Added file to plot only from csv without running all simulations again --- ##
# -- Simply exchange metric_key with wanted metric and then do the following command in terminal -- #

csv_path = Path("results/summary/sweep_20260402_092351.csv")
metric_key = "powerlaw_alpha_mean"
summary = results_io.load_summary(csv_path)

fig = plot_metric_vs_param(
    summary,
    metric_key=metric_key,
    show_colorbar=True,
    title=f"{_pretty_metric(metric_key)}",
)

output_path = Path("results/separate_plots") / f"{csv_path.stem}_{metric_key}_heatmap.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=200)
