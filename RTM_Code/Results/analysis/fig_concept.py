import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path as _P
_RES = _P(__file__).resolve().parents[1]
_ROOT = _RES.parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig = plt.figure(figsize=(12.5, 6.2))
ax = fig.add_axes([0.02, 0.50, 0.96, 0.46]); ax.set_xlim(0, 100); ax.set_ylim(-2, 30); ax.axis('off')
# column
n = 20; x0, x1, y0, h = 14, 84, 8, 10
for i in range(n):
    ax.add_patch(Rectangle((x0 + i * (x1 - x0) / n, y0), (x1 - x0) / n, h, fc='#e7ecf3' if i else '#cfe0f5', ec='#5b6b80', lw=0.8))
ax.text((x0 + x1) / 2, y0 + h + 1.4, 'One-dimensional linear column of N equal cells (kinetic primary minerals, equilibrium secondary minerals)', ha='center', fontsize=10.5)
ax.annotate('', xy=(x0 - 0.3, y0 + h / 2), xytext=(3, y0 + h / 2), arrowprops=dict(arrowstyle='-|>', lw=2.2, color='#c0392b'))
ax.text(1.5, y0 + h / 2 - 1.6, 'Inlet (cell 1): CO$_2$ added\nas mass; water slugs (WAG);\nflux = injected volume /\ncross-section + regional flux', fontsize=9.3, va='top', color='#7b1d13')
ax.annotate('', xy=(97, y0 + h / 2), xytext=(x1 + 0.3, y0 + h / 2), arrowprops=dict(arrowstyle='-|>', lw=2.2, color='#1f4e9c'))
ax.text(98.5, y0 + h / 2 - 1.6, 'Outlet: flux boundary;\nexported carbon is\naccounted in the ledger', fontsize=9.3, va='top', ha='right', color='#1f3f7a')
ax.text(x0, y0 + h + 0.6, 'x = 0', ha='center', fontsize=10); ax.text(x1, y0 + h + 0.6, 'x = L', ha='center', fontsize=10)
ax.text((x0 + x1) / 2, y0 - 5.6, 'Injection-driven closure: L = 5 m, 50 cells of 0.1 m (dispersivity 0.1 m).   Overpressure-driven closure: L = 50 m, 10 cells of 5 m (dispersivity 1 m).', ha='center', fontsize=9.8)
ax.text((x0 + x1) / 2, y0 - 8.0, 'Initial state: pre-equilibrated formation water in every cell; no CO$_2$ phase; no lateral, vertical or radial dimension.', ha='center', fontsize=9.8)
# schedules
sx = fig.add_axes([0.10, 0.13, 0.86, 0.30]); sx.set_xlim(0, 5); sx.set_ylim(0, 3.6)
for k, (lab, win) in enumerate((('Single-burst', [(0, 0.082)]), ('Continuous', [(0, 1.0)]), ('Pulsed WAG', [(0, 0.25), (0.5, 0.75)]))):
    yy = 2.6 - k * 1.0
    sx.add_patch(Rectangle((0, yy - 0.28), 5, 0.56, fc='#f2f2f2', ec='none'))
    for a, b in win:
        sx.add_patch(Rectangle((a, yy - 0.28), b - a, 0.56, fc='#e08a00', ec='k', lw=0.6))
    if lab == 'Pulsed WAG':
        for a, b in ((0.25, 0.5), (0.75, 1.0)):
            sx.add_patch(Rectangle((a, yy - 0.28), b - a, 0.56, fc='#4aa3df', ec='k', lw=0.6))
    sx.text(-0.08, yy, lab, ha='right', va='center', fontsize=11)
sx.set_yticks([]); sx.set_xlabel('Time (yr)  -  same CO$_2$ mass in every mode: 600 t (Mhow), 810 t (GRB); total simulated period 5 yr', fontsize=11)
for s in ('top', 'right', 'left'): sx.spines[s].set_visible(False)
sx.add_patch(Rectangle((3.3, 2.95), 0.25, 0.3, fc='#e08a00', ec='k', lw=0.6)); sx.text(3.6, 3.1, 'CO$_2$ injection', va='center', fontsize=10)
sx.add_patch(Rectangle((4.15, 2.95), 0.25, 0.3, fc='#4aa3df', ec='k', lw=0.6)); sx.text(4.45, 3.1, 'water slug', va='center', fontsize=10)
fig.savefig(str(_RES / 'figures' / 'concept.png'), dpi=200); print('ok')
