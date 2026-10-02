import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from pathlib import Path as _P
_RES = _P(__file__).resolve().parents[1]
_ROOT = _RES.parent
im = Image.open(str(_RES / 'figures' / 'toc_original.jpeg')).convert('RGB')
NAVY = (23/255, 14/255, 79/255); BLUE = (75/255, 120/255, 151/255); PURP = (109/255, 91/255, 131/255)
def render(w, h, draw):
    fig = plt.figure(figsize=(w/100, h/100), dpi=100); draw(fig)
    fig.canvas.draw(); arr = fig.canvas.buffer_rgba(); img = Image.frombuffer('RGBA', fig.canvas.get_width_height(), arr).convert('RGB'); plt.close(fig); return img
# ---- middle-bottom panel (x 425-968, y 395-832)
def mid(fig):
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 547); ax.set_ylim(437, 0); ax.axis('off')
    ax.add_patch(FancyBboxPatch((22, 4), 494, 64, boxstyle='round,pad=0,rounding_size=32', fc=NAVY, ec='none'))
    ax.text(269, 37, 'What the corrected model shows', color='white', ha='center', va='center', fontsize=21)
    items = [('Same CO$_2$ mass in every schedule;', 'full carbon ledger (closure < 10$^{-5}$ %)'),
             ('Overpressure-driven flow: native water', 'saturates trapping; no ranking possible'),
             ('Injection-driven flow: WAG ahead by', '~9 pp, but continuous + same water = WAG'),
             ('Gain follows water throughput, not', 'alternation; lost at regional flow ≥ 0.1 m/yr')]
    y = 110
    for a, b in items:
        ax.plot(34, y + 12, 'o', color=NAVY, ms=7)
        ax.text(50, y, a, fontsize=15.5, va='center', color='#222'); ax.text(50, y + 27, b, fontsize=15.5, va='center', color='#222')
        y += 80
right_w, right_h = 386, 724
BARS = {'Mhow Basalt (India)': ([32.0, 29.5, 38.2], BLUE), 'Grande Ronde Basalt (USA)': ([23.4, 23.9, 32.7], PURP)}
def right(fig):
    for k, (title, (vals, c)) in enumerate(BARS.items()):
        ax = fig.add_axes([0.17, 0.63 - k * 0.44, 0.78, 0.27])
        b = ax.bar(['Single-burst', 'Continuous', 'Pulsed WAG'], vals, color=c, width=0.65)
        for r, v in zip(b, vals): ax.text(r.get_x() + r.get_width()/2, v + 1.5, '%.1f' % v, ha='center', fontsize=12)
        ax.set_ylim(0, 50); ax.set_ylabel('Net efficiency (%)', fontsize=11.5); ax.tick_params(labelsize=10.5)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
    fig.text(0.5, 0.012, '5 yr, injection-driven flow, net of no-injection\ncontrol. Model-based; not field-validated;\nWallula/CarbFix report 60–95 % at 2 yr.', ha='center', fontsize=10.5, color='#333', va='bottom')
out = im.copy()
m = render(547, 437, mid); out.paste(m, (424, 395))
rt = render(right_w, right_h, right); out.paste(rt, (978, 108))
out.save(str(_RES / 'figures' / 'toc_revised.jpg'), quality=95); print(out.size)
