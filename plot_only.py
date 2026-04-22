
from pathlib import Path
from run import plot_metric_vs_param, _pretty_metric
import results_io

## --- Added file to plot only from csv without running all simulations again --- ##
# -- Simply exchange metric_key with wanted metric and then do the following command in terminal -- #

csv_path = Path("results/summary/sweep_20260422_162956.csv")

# --- Choose which metric to plot here ---
# Good options after your new power-law changes:
# "powerlaw_valid_mean"   -> fraction of valid fits
# "powerlaw_alpha_mean"   -> mean alpha
# "powerlaw_k_min_mean"   -> mean k_min
# "powerlaw_n_tail_mean"  -> mean n_tail
# "powerlaw_ks_mean"      -> mean KS statistic
metric_key = "powerlaw_ks_mean"

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
