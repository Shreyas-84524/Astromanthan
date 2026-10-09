"""Visualize saved final-test predictions without retraining or tuning."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
d = pd.read_csv(OUT / 'predictions.csv')
y = (d.Object_Class == 'STAR').to_numpy().astype(int)
pred = (d.predicted_class == 'STAR').to_numpy().astype(int)
correct = y == pred
rng = np.random.default_rng(42)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
for mask, label, colour, alpha, size in [
    (correct, 'Correct', '#246ea3', .13, 8),
    (~correct, 'Incorrect', '#d4512f', .85, 22),
]:
    axes[0].scatter(y[mask] + rng.uniform(-.1, .1, mask.sum()),
        pred[mask] + rng.uniform(-.1, .1, mask.sum()),
        s=size, color=colour, alpha=alpha, edgecolors='none', label=label)
    axes[1].scatter(y[mask] + rng.uniform(-.13, .13, mask.sum()),
        d.probability_STAR.to_numpy()[mask], s=size, color=colour,
        alpha=alpha, edgecolors='none', label=label)
axes[0].plot([-.2, 1.2], [-.2, 1.2], '--', color='#555555', lw=1)
axes[0].set(xticks=[0, 1], xticklabels=['GALAXY', 'STAR'],
    yticks=[0, 1], yticklabels=['GALAXY', 'STAR'],
    xlim=(-.3, 1.3), ylim=(-.3, 1.3), xlabel='Actual class',
    ylabel='Predicted class', title='Actual vs predicted labels')
axes[1].axhline(.5, ls='--', color='#555555', lw=1, label='Decision threshold')
axes[1].set(xticks=[0, 1], xticklabels=['GALAXY', 'STAR'],
    xlim=(-.3, 1.3), ylim=(-.04, 1.04), xlabel='Actual class',
    ylabel='Predicted probability of STAR', title='Actual class vs model probability')
axes[1].legend(loc='center left', fontsize=8)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle(f'Final astronomy model: {len(d):,} test objects; {(~correct).sum():,} errors', fontsize=13)
fig.text(.5, .01, 'Horizontal/label jitter separates overlapping points; it does not change labels or probabilities.',
    ha='center', fontsize=9, color='#555555')
fig.tight_layout(rect=(0, .04, 1, .95))
path = OUT / 'figures' / '07_actual_vs_predicted.png'
fig.savefig(path, dpi=180)
plt.close(fig)
print(path)
