"""
app/services/chart_service.py
==============================
Renders matplotlib / seaborn charts server-side and returns them as
base64-encoded PNG strings that can be embedded directly in HTML:

    <img src="data:image/png;base64,{{ chart_b64 }}">

All functions accept plain Python lists/dicts so they can be called from
any route without touching SQLAlchemy directly.

Design notes
------------
* ``matplotlib.use('Agg')`` is called at import time so the module is safe
  to import in a headless Flask worker (no display required).
* Every function creates and closes its own Figure — no global state leaks.
* A shared warm colour palette mirrors the existing ScoreStat CSS variables.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import matplotlib
matplotlib.use('Agg')                       # headless, must come before pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ── Palette (matches CSS variables in custom.css) ──────────────────────────
PAL = {
    'bg':        '#fdf7ee',
    'panel':     '#f4ece0',
    'amber':     '#d9a319',
    'amber_lt':  '#f2d47a',
    'green':     '#7eaa4f',
    'green_lt':  '#c2dda0',
    'red':       '#b64032',
    'red_lt':    '#e8a89e',
    'blue':      '#4178b4',
    'blue_lt':   '#a8c4e8',
    'purple':    '#7c4dbe',
    'muted':     '#836650',
    'border':    '#dbcab3',
    'text':      '#362418',
    'text2':     '#5a3f2d',
}

# Seaborn theme applied globally once
sns.set_theme(style='whitegrid', rc={
    'axes.facecolor':    PAL['bg'],
    'figure.facecolor':  PAL['bg'],
    'grid.color':        PAL['border'],
    'grid.linewidth':    0.6,
    'axes.edgecolor':    PAL['border'],
    'axes.labelcolor':   PAL['text2'],
    'xtick.color':       PAL['muted'],
    'ytick.color':       PAL['muted'],
    'text.color':        PAL['text'],
    'font.family':       'DejaVu Sans',
})


# ── Helpers ────────────────────────────────────────────────────────────────

def _to_b64(fig: plt.Figure) -> str:
    """Serialise a matplotlib Figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def _base_fig(w: float = 8, h: float = 3.8) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PAL['bg'])
    ax.set_facecolor(PAL['panel'])
    for spine in ax.spines.values():
        spine.set_edgecolor(PAL['border'])
        spine.set_linewidth(0.8)
    return fig, ax


def _no_data_chart(message: str = 'No data yet', w: float = 7, h: float = 3) -> str:
    fig, ax = _base_fig(w, h)
    ax.text(0.5, 0.5, message, ha='center', va='center',
            fontsize=13, color=PAL['muted'],
            transform=ax.transAxes)
    ax.axis('off')
    return _to_b64(fig)


# ── Player charts ──────────────────────────────────────────────────────────

def player_runs_trend(batting_history: list[dict]) -> str:
    """
    Bar chart of runs per innings with a strike-rate line overlay.
    Bars are coloured: green ≥50, amber ≥20, red <20.
    """
    if not batting_history:
        return _no_data_chart('No batting innings recorded yet')

    labels = [f"Inn {i+1}" for i in range(len(batting_history))]
    runs   = [h['runs'] for h in batting_history]
    sr     = [h['strike_rate'] for h in batting_history]

    colours = [
        PAL['green'] if r >= 50 else PAL['amber'] if r >= 20 else PAL['red']
        for r in runs
    ]

    fig, ax1 = _base_fig(max(7, len(labels) * 0.72), 4)
    ax2 = ax1.twinx()

    x = np.arange(len(labels))
    bars = ax1.bar(x, runs, color=colours, width=0.55, zorder=3,
                   edgecolor=PAL['bg'], linewidth=0.5)

    ax2.plot(x, sr, color=PAL['amber'], linewidth=2, marker='o',
             markersize=5, markerfacecolor=PAL['amber_lt'],
             markeredgecolor=PAL['amber'], zorder=4, label='Strike Rate')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax1.set_ylabel('Runs', color=PAL['text2'], fontsize=10)
    ax2.set_ylabel('Strike Rate', color=PAL['amber'], fontsize=10)
    ax2.tick_params(axis='y', colors=PAL['amber'])
    ax1.set_ylim(bottom=0)

    # Legend patches
    legend_items = [
        mpatches.Patch(color=PAL['green'],  label='50+'),
        mpatches.Patch(color=PAL['amber'],  label='20–49'),
        mpatches.Patch(color=PAL['red'],    label='<20'),
        plt.Line2D([0], [0], color=PAL['amber'], linewidth=2,
                   marker='o', label='SR'),
    ]
    ax1.legend(handles=legend_items, loc='upper left', fontsize=8,
               framealpha=0.7, facecolor=PAL['bg'])

    ax1.set_title('Runs Per Innings', fontsize=13, color=PAL['text'],
                  pad=10, fontweight='bold')
    fig.tight_layout()
    return _to_b64(fig)


def player_bowling_trend(bowling_history: list[dict]) -> str:
    """
    Bar chart of wickets per spell with economy-rate line overlay.
    """
    if not bowling_history:
        return _no_data_chart('No bowling spells recorded yet')

    labels = [f"Sp {i+1}" for i in range(len(bowling_history))]
    wkts   = [h['wickets'] for h in bowling_history]
    eco    = [h['economy'] for h in bowling_history]

    fig, ax1 = _base_fig(max(7, len(labels) * 0.72), 4)
    ax2 = ax1.twinx()

    x = np.arange(len(labels))
    bar_colours = [
        PAL['red'] if w >= 3 else PAL['red_lt'] if w >= 1 else PAL['muted']
        for w in wkts
    ]
    ax1.bar(x, wkts, color=bar_colours, width=0.55, zorder=3,
            edgecolor=PAL['bg'], linewidth=0.5)
    ax2.plot(x, eco, color=PAL['blue'], linewidth=2, marker='s',
             markersize=5, markerfacecolor=PAL['blue_lt'],
             markeredgecolor=PAL['blue'], zorder=4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax1.set_ylabel('Wickets', color=PAL['text2'], fontsize=10)
    ax2.set_ylabel('Economy', color=PAL['blue'], fontsize=10)
    ax2.tick_params(axis='y', colors=PAL['blue'])
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.set_ylim(bottom=0)

    legend_items = [
        mpatches.Patch(color=PAL['red'],    label='3+ wkts'),
        mpatches.Patch(color=PAL['red_lt'], label='1–2 wkts'),
        mpatches.Patch(color=PAL['muted'],  label='0 wkts'),
        plt.Line2D([0], [0], color=PAL['blue'], linewidth=2,
                   marker='s', label='Economy'),
    ]
    ax1.legend(handles=legend_items, loc='upper right', fontsize=8,
               framealpha=0.7, facecolor=PAL['bg'])

    ax1.set_title('Wickets & Economy Per Spell', fontsize=13,
                  color=PAL['text'], pad=10, fontweight='bold')
    fig.tight_layout()
    return _to_b64(fig)


def player_scoring_mix(batting_history: list[dict]) -> str:
    """
    Horizontal stacked bar showing how the player scores their runs.
    """
    if not batting_history:
        return _no_data_chart('No batting data')

    total_balls = sum(h['balls_faced'] for h in batting_history)
    fours       = sum(h['fours'] for h in batting_history)
    sixes       = sum(h['sixes'] for h in batting_history)
    boundary_b  = fours + sixes          # balls that went for 4/6
    # approximate non-boundary scoring balls as ~25 % of remaining
    other_b     = max(0, total_balls - boundary_b)

    values = [fours, sixes, other_b]
    labels = ['Fours', 'Sixes', 'Other balls']
    colours = [PAL['blue'], PAL['amber'], PAL['panel']]

    fig, ax = _base_fig(6, 1.8)
    left = 0
    for val, lbl, col in zip(values, labels, colours):
        if val > 0:
            ax.barh(0, val, left=left, color=col, edgecolor=PAL['bg'],
                    linewidth=0.8, height=0.45)
            if val / total_balls > 0.07:
                ax.text(left + val / 2, 0, f'{lbl}\n{val}',
                        ha='center', va='center', fontsize=9,
                        color=PAL['text'], fontweight='bold')
        left += val

    ax.set_xlim(0, total_balls)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.set_title('Scoring Breakdown (balls)', fontsize=11,
                 color=PAL['text'], pad=8, fontweight='bold')
    fig.tight_layout()
    return _to_b64(fig)


def player_radar(career_stats: dict, breakdown: dict, stat_type: str = 'batting') -> str:
    """
    Radar / spider chart of a normalised batting or bowling profile.
    """
    if stat_type == 'bowling':
        dimensions = ['Eco\nControl', 'Wicket\nEfficiency', 'Dot\n%', 'Control\nScore', 'Bowl\nIndex']
        raw = [
            max(0.0, 1.0 - min(float(breakdown.get('eco_norm', 0)), 1.0)),
            max(0.0, 1.0 - min(float(breakdown.get('avg_norm', 0)), 1.0)),
            min(float(career_stats.get('dot_pct', 0)), 1.0),
            min(float(breakdown.get('dot_entropy_score', 0)), 1.0),
            min(float(career_stats.get('bowling_index', 0)), 1.0),
        ]
        title = 'Career Bowling Profile'
    else:
        dimensions = ['Average', 'Strike\nRate', 'Consistency', 'Boundary\n%', 'Bat\nIndex']
        raw = [
            min(float(breakdown.get('avg_norm', 0)), 1.0),
            min(float(breakdown.get('sr_norm',  0)), 1.0),
            min(float(breakdown.get('consistency', 0)), 1.0),
            min(float(career_stats.get('boundary_pct', 0)), 1.0),
            min(float(career_stats.get('batting_index', 0)), 1.0),
        ]
        label = career_stats.get('consistency_label', '')
        title = f'Career Profile - {label.title()}'

    values = [v * 100 for v in raw]
    N = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot  = angles + [angles[0]]

    fig = plt.figure(figsize=(4.5, 4.5))
    fig.patch.set_facecolor(PAL['bg'])
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(PAL['panel'])

    ax.plot(angles_plot, values_plot, color=PAL['amber'], linewidth=2, zorder=3)
    ax.fill(angles_plot, values_plot, color=PAL['amber'], alpha=0.22, zorder=2)
    ax.scatter(angles, values, color=PAL['amber'], s=40, zorder=4)

    ax.set_xticks(angles)
    ax.set_xticklabels(dimensions, fontsize=9, color=PAL['text'])
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=7, color=PAL['muted'])
    ax.grid(color=PAL['border'], linewidth=0.7)
    ax.spines['polar'].set_color(PAL['border'])

    ax.set_title(title, fontsize=11, color=PAL['text'], pad=14, fontweight='bold')
    fig.tight_layout()
    return _to_b64(fig)
# ── Match charts ───────────────────────────────────────────────────────────

def match_run_progression(innings_list: list[dict]) -> str:
    """
    Line chart showing cumulative run progression for both innings side-by-side.
    Wicket overs are marked with a red ×.
    """
    if not innings_list:
        return _no_data_chart('No innings data')

    colours = [PAL['amber'], PAL['blue']]
    fig, ax = _base_fig(9, 4)

    for idx, inn in enumerate(innings_list):
        prog  = inn.get('run_progression', [])
        if not prog:
            continue
        overs  = [p['over'] for p in prog]
        cumul  = [p['cumulative_runs'] for p in prog]
        wkt_ov = [p['over'] for p in prog if p.get('wickets_this_over', 0) > 0]
        wkt_sc = [p['cumulative_runs'] for p in prog if p.get('wickets_this_over', 0) > 0]

        col = colours[idx % len(colours)]
        label = f"Innings {inn['innings_number']} ({inn['total_runs']}/{inn['total_wickets']})"
        ax.plot(overs, cumul, color=col, linewidth=2.2,
                marker='o', markersize=3.5, label=label, zorder=3)
        ax.fill_between(overs, cumul, alpha=0.08, color=col)
        if wkt_ov:
            ax.scatter(wkt_ov, wkt_sc, color=PAL['red'], marker='x',
                       s=60, zorder=5, linewidths=1.5)

    ax.set_xlabel('Over', fontsize=10)
    ax.set_ylabel('Cumulative Runs', fontsize=10)
    ax.set_title('Run Progression', fontsize=13, color=PAL['text'],
                 pad=10, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.7, facecolor=PAL['bg'])

    # Mark wickets legend
    ax.scatter([], [], color=PAL['red'], marker='x', s=60,
               linewidths=1.5, label='Wicket over')
    ax.legend(fontsize=9, framealpha=0.7, facecolor=PAL['bg'])

    fig.tight_layout()
    return _to_b64(fig)


def match_batting_contributions(batsmen: list[dict], innings_num: int) -> str:
    """
    Horizontal bar chart of individual batting contributions.
    Bars split into 4s, 6s, and other runs via stacked colouring.
    """
    if not batsmen:
        return _no_data_chart(f'No batting data for innings {innings_num}')

    top = batsmen[:10]
    names  = [b['player_name'].split()[-1] for b in top]   # surname only
    runs   = [b['runs'] for b in top]
    fours  = [b['fours'] * 4 for b in top]
    sixes  = [b['sixes'] * 6 for b in top]
    other  = [max(0, r - f - s) for r, f, s in zip(runs, fours, sixes)]

    fig, ax = _base_fig(8, max(3.5, len(top) * 0.52))
    y = np.arange(len(names))

    ax.barh(y, other,  color=PAL['green'],  height=0.5, label='Singles/2s', zorder=3)
    ax.barh(y, fours,  color=PAL['blue'],   height=0.5, left=other, label='4s', zorder=3)
    ax.barh(y, sixes,  color=PAL['amber'],  height=0.5,
            left=[o+f for o,f in zip(other,fours)], label='6s', zorder=3)

    for i, b in enumerate(top):
        sr_txt = f"{b['strike_rate']:.0f} SR"
        ax.text(b['runs'] + 0.5, i, sr_txt, va='center', fontsize=8,
                color=PAL['muted'])

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Runs', fontsize=10)
    ax.set_title(f'Batting Contributions — Innings {innings_num}',
                 fontsize=12, color=PAL['text'], pad=10, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.7, facecolor=PAL['bg'])
    ax.invert_yaxis()
    fig.tight_layout()
    return _to_b64(fig)


def match_bowling_figures(bowlers: list[dict], innings_num: int) -> str:
    """
    Grouped bar chart — wickets and economy per bowler.
    """
    if not bowlers:
        return _no_data_chart(f'No bowling data for innings {innings_num}')

    top   = bowlers[:8]
    names = [b['player_name'].split()[-1] for b in top]
    wkts  = [b['wickets'] for b in top]
    eco   = [b['economy'] for b in top]

    fig, ax1 = _base_fig(8, max(3.5, len(top) * 0.52))
    ax2 = ax1.twinx()

    y = np.arange(len(names))
    bar_w = 0.38
    ax1.barh(y - bar_w/2, wkts, height=bar_w, color=PAL['red'],
             label='Wickets', zorder=3)
    ax2.barh(y + bar_w/2, eco,  height=bar_w, color=PAL['blue'],
             alpha=0.75, label='Economy', zorder=3)

    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel('Wickets', color=PAL['red'], fontsize=10)
    ax2.set_xlabel('Economy', color=PAL['blue'], fontsize=10)
    ax1.tick_params(axis='x', colors=PAL['red'])
    ax2.tick_params(axis='x', colors=PAL['blue'])
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax1.set_title(f'Bowling Figures — Innings {innings_num}',
                  fontsize=12, color=PAL['text'], pad=10, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               framealpha=0.7, facecolor=PAL['bg'])
    ax1.invert_yaxis()
    fig.tight_layout()
    return _to_b64(fig)


def match_boundary_breakdown(boundaries: dict, innings_num: int) -> str:
    """
    Seaborn-styled pie/donut chart for ball-type breakdown.
    """
    data = {
        'Fours':   boundaries.get('fours', 0),
        'Sixes':   boundaries.get('sixes', 0),
        'Dots':    boundaries.get('dots',  0),
        'Singles': boundaries.get('ones',  0),
        'Twos':    boundaries.get('twos',  0),
        'Extras':  boundaries.get('extras',0),
    }
    data = {k: v for k, v in data.items() if v > 0}
    if not data:
        return _no_data_chart('No ball data')

    colours_map = {
        'Fours':   PAL['blue'],
        'Sixes':   PAL['amber'],
        'Dots':    PAL['muted'],
        'Singles': PAL['green'],
        'Twos':    PAL['green_lt'],
        'Extras':  PAL['red_lt'],
    }
    labels  = list(data.keys())
    values  = list(data.values())
    colours = [colours_map.get(l, PAL['muted']) for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(PAL['bg'])
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colours,
        autopct='%1.0f%%',
        pctdistance=0.78,
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=PAL['bg'], linewidth=1.5),
    )
    for t in texts:
        t.set_fontsize(9)
        t.set_color(PAL['text'])
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color(PAL['text'])

    ax.set_title(f'Ball Type Breakdown — Innings {innings_num}',
                 fontsize=11, color=PAL['text'], pad=10, fontweight='bold')
    fig.tight_layout()
    return _to_b64(fig)


def match_over_run_rate(innings_list: list[dict]) -> str:
    """
    Bar chart — runs per over for each innings (side by side if two innings).
    """
    if not innings_list:
        return _no_data_chart('No data')

    all_overs = set()
    for inn in innings_list:
        for p in inn.get('run_progression', []):
            all_overs.add(p['over'])
    if not all_overs:
        return _no_data_chart('No over data')

    max_over = max(all_overs)
    x = np.arange(1, max_over + 1)
    colours = [PAL['amber'], PAL['blue']]
    width   = 0.38 if len(innings_list) > 1 else 0.55

    fig, ax = _base_fig(10, 4)

    for idx, inn in enumerate(innings_list):
        prog = {p['over']: p['runs_this_over'] for p in inn.get('run_progression', [])}
        runs = [prog.get(ov, 0) for ov in range(1, max_over + 1)]
        offset = (idx - (len(innings_list)-1)/2) * width
        col = colours[idx % len(colours)]
        label = f"Innings {inn['innings_number']}"
        bar_colours = [
            PAL['red'] if (p.get('wickets_this_over', 0) > 0 and
                           any(p['over'] == ov for p in inn.get('run_progression', [])))
            else col
            for ov in range(1, max_over + 1)
        ]
        ax.bar(x + offset, runs, width=width, color=col, alpha=0.82,
               label=label, zorder=3, edgecolor=PAL['bg'], linewidth=0.4)

    ax.set_xlabel('Over', fontsize=10)
    ax.set_ylabel('Runs', fontsize=10)
    ax.set_title('Runs Per Over', fontsize=13, color=PAL['text'],
                 pad=10, fontweight='bold')
    ax.set_xticks(x[::2])
    ax.legend(fontsize=9, framealpha=0.7, facecolor=PAL['bg'])
    fig.tight_layout()
    return _to_b64(fig)

