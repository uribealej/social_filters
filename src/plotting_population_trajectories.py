"""Figures for within-fish shared PCA and equal-fish population comparisons."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_control_similarity_curves(similarity, stimuli_by_side, *, colors=None,
        combine_sides=False, show_sem=True, show_repeatability=True,
        figsize=(12, 5), font_size=9):
    """Overlay same-time population-pattern Pearson correlation curves.

    Select from prepared stimulus labels; Control self-identity is omitted.
    Means/SEM weight finite fish equally. Optional matched split-half Control
    repeatability is gray/dashed; it is descriptive, not a threshold or ceiling.
    Full r range [-1,1] is shared across axes. No recalculation or significance.
    """
    if not stimuli_by_side:
        raise ValueError('Select at least one side for similarity curves.')
    colors = colors or {}
    with plt.rc_context({'font.size': font_size}):
        fig, axes = plt.subplots(1, 1 if combine_sides else len(stimuli_by_side),
                                 figsize=figsize, squeeze=False, sharey=True)
        for side_index, (side, labels) in enumerate(stimuli_by_side.items()):
            ax = axes[0, 0 if combine_sides else side_index]
            control = similarity['controls'][side]
            if len(set(labels)) != len(labels):
                raise ValueError('Selected similarity stimuli must be unique.')
            for index, stimulus in enumerate(label for label in labels if label != control):
                curve = similarity['curve_summary'].query('side == @side & stimulus == @stimulus')
                if curve.empty:
                    raise ValueError(f'{side}/{stimulus}: similarity has not been prepared.')
                color = colors.get(stimulus, f'C{index}')
                style = '--' if combine_sides and side_index % 2 else '-'
                ax.plot(curve.time_s, curve.mean_r, color=color, linestyle=style, lw=1.8, label=stimulus)
                if show_sem:
                    ax.fill_between(curve.time_s, curve.mean_r - curve.sem_r,
                                    curve.mean_r + curve.sem_r, color=color, alpha=0.12)
            reference = similarity['repeatability_summary']
            reference = reference.loc[reference.side.eq(side)]
            if show_repeatability and len(reference):
                label = f'{side}: Control repeatability' if combine_sides else 'Control repeatability'
                ax.plot(reference.time_s, reference.mean_r, color='0.4', linestyle=':', lw=2.1, label=label)
                if show_sem:
                    ax.fill_between(reference.time_s, reference.mean_r - reference.sem_r,
                                    reference.mean_r + reference.sem_r, color='0.5', alpha=0.08)
            ax.axhline(0, color='0.6', lw=0.7)
            ax.set_xlim(*similarity['time_window_s'])
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel('Time from movement-program start (s)')
            ax.set_ylabel('Same-time population-pattern correlation (r)')
            ax.set_title('Left + right' if combine_sides else side.capitalize())
            ax.legend(frameon=False, fontsize=font_size - 1)
            ax.spines[['top', 'right']].set_visible(False)
        method = 'matched split-half means' if similarity['estimator'] == 'split_half' else 'all-repetition means'
        fig.suptitle(f'Pattern similarity to same-side Control · {method}\n'
                     'Equal fish means ± SEM', fontsize=font_size + 1)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
    return dict(figure=fig, axes=axes)


def plot_pooled_population_trajectories(analysis, side, groups, *, dimensions=(0, 1),
        time_window_s=(0, 4), focus_window_s=(0, 4), colors=None, figsize=(14, 4.8),
        font_size=9, marker_interval_s=2.0, pca_fit=None, line_styles=None,
        legend_labels=None, legend_columns=1, title=None, legend_font_size=None,
        legend_loc=None, legend_bbox_to_anchor=None):
    """Plot count-balanced pooled mean paths in flexible panels with fixed axes.

    ``groups`` is an ordered title->condition-list mapping. All panels reuse one
    pooled PCA and common plot limits within this figure; changing displayed
    conditions/windows never refits it. Display window bounds are inclusive native
    sample centers. The focus interval is thicker; circles/squares mark start/end
    and triangles its end when inside the path. Dots mark regular native samples;
    2D times are labeled on the first path in each panel. No individual trials or
    fish paths are fabricated. Dimensions select two or three zero-based PCs.
    Return figure/axes without saving or mutating analysis.

    ``pca_fit`` may supply build_joint_population_pca's shared fit, with condition
    keys (side,label), enabling physically comparable left/right overlays.
    ``line_styles``/``legend_labels`` map condition keys to display styles/text.
    ``legend_font_size`` sets legend text independently of axis text; None uses
    font_size - 1. ``legend_loc`` overrides the dimensionality-specific location;
    optional ``legend_bbox_to_anchor`` positions it in axes coordinates.
    Set ``focus_window_s=None`` for uniform full-path emphasis without an early
    window; a single entry in groups creates one selectable overlay axes.
    """
    fit = analysis['pooled_fits'][side] if pca_fit is None else pca_fit
    focus_bounds = () if focus_window_s is None else tuple(focus_window_s)
    if (not groups or len(dimensions) not in (2, 3)
            or len(set(dimensions)) != len(dimensions)
            or any(d < 0 or d >= len(fit['explained_variance_ratio']) for d in dimensions)
            or not np.isfinite([*time_window_s, *focus_bounds, marker_interval_s]).all()
            or time_window_s[0] >= time_window_s[1] or marker_interval_s <= 0
            or (focus_window_s is not None and focus_window_s[0] > focus_window_s[1])):
        raise ValueError('Invalid pooled trajectory groups, PCs, display/focus window or marker spacing.')
    for group_title, labels in groups.items():
        if not labels or len(set(labels)) != len(labels) or not set(labels) <= set(fit['conditions']):
            raise ValueError(f'{group_title}: display labels must be unique fitted conditions.')
    times = fit['time_s']
    keep = (times >= time_window_s[0]) & (times <= time_window_s[1])
    if keep.sum() < 2:
        raise ValueError('Pooled display window must contain at least two native samples.')
    times, colors = times[keep], colors or {}
    focus = ((times >= focus_window_s[0]) & (times <= focus_window_s[1])
             if focus_window_s is not None else np.zeros(len(times), dtype=bool))
    line_styles, legend_labels = line_styles or {}, legend_labels or {}
    paths = {label: path[keep][:, dimensions] for label, path in fit['means'].items()}
    shown = list(dict.fromkeys(label for labels in groups.values() for label in labels))
    all_points = np.concatenate([paths[label] for label in shown])
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    padding = np.maximum(hi - lo, 0.1) * 0.12
    with plt.rc_context({'font.size': font_size}):
        fig = plt.figure(figsize=figsize)
        axes = []
        for col, (panel_title, labels) in enumerate(groups.items()):
            ax = fig.add_subplot(1, len(groups), col + 1,
                                projection='3d' if len(dimensions) == 3 else None)
            axes.append(ax)
            for label_index, label in enumerate(labels):
                path = paths[label]
                color = colors.get(label, f'C{fit["conditions"].index(label)}')
                style = line_styles.get(label, '-')
                ax.plot(*path.T, color=color, linewidth=1.8 if focus_window_s is None else 1.2,
                        alpha=0.9, linestyle=style, label=legend_labels.get(label, str(label)))
                if focus.sum() >= 2:
                    ax.plot(*path[focus].T, color=color, linewidth=2.4, linestyle=style)
                ax.scatter(*path[0], color=color, marker='o', s=30, edgecolor='black', linewidth=0.4)
                ax.scatter(*path[-1], color=color, marker='s', s=26, edgecolor='black', linewidth=0.4)
                if focus_window_s is not None and times[0] < focus_window_s[1] < times[-1]:
                    index = np.argmin(abs(times - focus_window_s[1]))
                    ax.scatter(*path[index], color=color, marker='^', s=42, edgecolor='black', linewidth=0.4)
                targets = np.arange(np.ceil(times[0] / marker_interval_s) * marker_interval_s,
                                    times[-1] + 1e-8, marker_interval_s)
                for target in targets:
                    index = np.argmin(abs(times - target))
                    ax.scatter(*path[index], color=color, s=12)
                    if len(dimensions) == 2 and label_index == 0:
                        ax.annotate(f'{times[index]:g}', path[index], xytext=(5, 6),
                                    textcoords='offset points', fontsize=font_size - 1, color=color)
                if len(dimensions) == 2 and len(path) >= 4:
                    index = min(max(1, int(len(path) / 3)), len(path) - 2)
                    ax.annotate('', xy=path[index + 1], xytext=path[index],
                                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
            axis_labels = [f'PC {d + 1} ({100 * fit["explained_variance_ratio"][d]:.1f}%)' for d in dimensions]
            ax.set_xlabel(axis_labels[0]); ax.set_ylabel(axis_labels[1])
            ax.set_xlim(lo[0] - padding[0], hi[0] + padding[0])
            ax.set_ylim(lo[1] - padding[1], hi[1] + padding[1])
            if len(dimensions) == 3:
                ax.set_zlabel(axis_labels[2])
                ax.set_zlim(lo[2] - padding[2], hi[2] + padding[2])
            ax.set_title(panel_title)
            ax.legend(loc=legend_loc or ('best' if len(dimensions) == 2 else 'upper left'),
                      frameon=False, fontsize=font_size - 1 if legend_font_size is None else legend_font_size,
                      ncols=legend_columns, bbox_to_anchor=legend_bbox_to_anchor)
            if len(dimensions) == 2:
                ax.spines[['top', 'right']].set_visible(False)
        emphasis = (f'; thicker path: {focus_window_s[0]:g}–{focus_window_s[1]:g} s'
                    if focus_window_s is not None else '')
        fig.suptitle(f'{side.capitalize()} pooled means · {time_window_s[0]:g}–{time_window_s[1]:g} s · '
                     f'{len(fit["neuron_metadata"])} neurons / {len(analysis["fish_ids"])} fish\n'
                     f'{title or "Shared PCA"}; circle: start, square: end{emphasis}',
                     fontsize=font_size + 1)
        if len(dimensions) == 3:
            fig.subplots_adjust(left=0.03, right=0.94, bottom=0.10, top=0.83, wspace=0.28)
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.94))
    return dict(figure=fig, axes=np.asarray(axes))


def plot_control_distance_overlay(analysis, controls, stimuli_by_side, *, significance=None,
        colors=None, time_window_s=(0, 14), program_end_s=None, combine_sides=False,
        show_sem=True, y_scale='linear', symlog_linthresh=0.5, figsize=(12, 5.5), font_size=9):
    """Overlay each selected stimulus's full-space distance from same-side Control.

    Input metrics must contain the ordered (control,stimulus) pair for every
    non-control selection. Omit the control's self-comparison; zero is the visual
    reference, not independent-trial reliability. Means and descriptive SEM give
    each fish equal weight. First corrected significant times are returned in
    detection_table without markers or evidence annotations on the plot. They
    are not biological change-points or proof of earlier equality. Selecting
    display curves does not change corrections.
    ``combine_sides`` overlays sides with solid/dashed lines on one axes. Otherwise
    use one axes per selected side. Colors map stimulus labels to shared family
    colors. Optional symlog preserves negative estimates while compressing range.
    """
    if (not stimuli_by_side or y_scale not in {'linear', 'symlog'}
            or time_window_s[0] >= time_window_s[1] or symlog_linthresh <= 0):
        raise ValueError('Invalid control-distance selections, time window, or scale.')
    sides, colors, detections = list(stimuli_by_side), colors or {}, []
    with plt.rc_context({'font.size': font_size}):
        fig, axes = plt.subplots(1, 1 if combine_sides else len(sides), figsize=figsize, squeeze=False)
        for side_index, side in enumerate(sides):
            ax = axes[0, 0 if combine_sides else side_index]
            control = controls[side]
            selected = list(stimuli_by_side[side])
            if len(set(selected)) != len(selected) or not selected:
                raise ValueError('Require unique, nonempty selected stimuli.')
            for index, stimulus in enumerate(label for label in selected if label != control):
                table = analysis['metrics'].query(
                    'side == @side & condition_a == @control & condition_b == @stimulus')
                table = table[table.time_s.between(*time_window_s)]
                if table.empty:
                    raise ValueError(f'{side}: prepare control pair ({control}, {stimulus}) before plotting.')
                wide = table.pivot(index='time_s', columns='fish_id', values='cv_squared_distance')
                mean, sem = wide.mean(axis=1), wide.sem(axis=1)
                color = colors.get(stimulus, f'C{index}')
                style = '--' if combine_sides and side_index % 2 else '-'
                if show_sem:
                    ax.fill_between(wide.index, mean - sem, mean + sem, color=color, alpha=0.12)
                ax.plot(wide.index, mean, color=color, linestyle=style, lw=1.8,
                        label=f'{stimulus} vs {control}')
                first = np.nan
                if significance is not None:
                    sig = significance.query(
                        'side == @side & condition_a == @control & condition_b == @stimulus & significant')
                    sig = sig[sig.time_s.between(*time_window_s)]
                    if len(sig):
                        first = float(sig.time_s.min())
                detections.append(dict(side=side, control=control, stimulus=stimulus,
                                       first_corrected_evidence_s=first))
            ax.set_title('Left + right' if combine_sides else side.capitalize())
        for ax in axes.ravel():
            ax.axhline(0, color='0.5', lw=0.8)
            if program_end_s is not None and time_window_s[0] <= program_end_s <= time_window_s[1]:
                ax.axvline(program_end_s, color='0.5', linestyle=':', lw=0.9)
            ax.set_xlim(*time_window_s)
            ax.set_xlabel('Time from stimulus motion onset (s)')
            ax.set_ylabel('Crossvalidated distance² from Control\n(z² per neuron)')
            if y_scale == 'symlog':
                ax.set_yscale('symlog', linthresh=symlog_linthresh)
            ax.legend(frameon=False, fontsize=font_size - 1)
            ax.spines[['top', 'right']].set_visible(False)
        caption = 'Population distance from same-side Control · equal fish means' + (' ± SEM' if show_sem else '')
        if program_end_s is not None and time_window_s[0] <= program_end_s <= time_window_s[1]:
            caption += '\nDotted line: movement-program end'
        fig.suptitle(caption, fontsize=font_size + 1)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
    return dict(figure=fig, axes=axes, detection_table=pd.DataFrame(detections))


def plot_population_distance(analysis, side, *, pairs=None, significance=None,
        stimulus_events=None, focus_window_s=(0, 4), context_window_s=(0, 14),
        program_end_s=None, show_fish=False, figsize=(13, 6), font_size=9):
    """Early/context views of full-space crossvalidated squared distance only.

    Fish are weighted equally. Shading around the mean is descriptive SEM, not
    neuron-sampling uncertainty. Optional corrected evidence is reused from the
    complete configured family; display selection does not rerun tests. Focus and
    context endpoints include native sample centers. The context panel shades the
    focus window and can mark program end. Negative distance estimates are kept.
    Return figure/axes without calculating new scientific measurements.
    """
    pairs = analysis['side_conditions'][side]['pairs'] if pairs is None else pairs
    if not pairs or any(tuple(pair) not in map(tuple, analysis['side_conditions'][side]['pairs']) for pair in pairs):
        raise ValueError('Display pairs must be present in the prepared analysis.')
    windows = [('First movement seconds', focus_window_s), ('Movement and calcium tail', context_window_s)]
    with plt.rc_context({'font.size': font_size}):
        fig, axes = plt.subplots(2, len(pairs), figsize=figsize, squeeze=False)
        for col, (a, b) in enumerate(pairs):
            table = analysis['metrics'].query('side == @side & condition_a == @a & condition_b == @b')
            for row, (title, window) in enumerate(windows):
                ax = axes[row, col]
                selected = table[table.time_s.between(*window)]
                wide = selected.pivot(index='time_s', columns='fish_id', values='cv_squared_distance')
                if len(wide) < 2:
                    raise ValueError('Distance display needs at least two samples in each window.')
                mean, sem = wide.mean(axis=1), wide.sem(axis=1)
                if show_fish:
                    ax.plot(wide.index, wide.to_numpy(), color='0.6', alpha=0.35, lw=0.7)
                ax.fill_between(wide.index, mean - sem, mean + sem, color='#385F73', alpha=0.2)
                ax.plot(wide.index, mean, color='#385F73', lw=2)
                ax.axhline(0, color='0.6', lw=0.7)
                if row == 1:
                    ax.axvspan(*focus_window_s, color='#DCE7E4', alpha=0.45, zorder=-1)
                    if program_end_s is not None and window[0] <= program_end_s <= window[1]:
                        ax.axvline(program_end_s, color='0.5', linestyle=':', lw=1)
                event = (stimulus_events or {}).get((a, b), np.nan)
                if np.isfinite(event) and window[0] <= event <= window[1]:
                    ax.axvline(event, color='#AD713F', linestyle='--', lw=1)
                if significance is not None:
                    sig = significance.query('side == @side & condition_a == @a & condition_b == @b & significant')
                    sig = sig[sig.time_s.between(*window)]
                    ax.plot(sig.time_s, np.full(len(sig), 0.97), '|', color='#B44444',
                            transform=ax.get_xaxis_transform(), markersize=6)
                if row == 0:
                    ax.set_title(f'{a} vs {b}')
                if col == 0:
                    ax.set_ylabel(f'{title}\nCrossvalidated distance²\n(z² per neuron)')
                ax.set_xlabel('Time from movement-program start (s)')
                ax.set_xlim(*window)
                ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle(f'{side.capitalize()} · full neuron space · equal fish means ± SEM\n'
                     'Brown dashed: asset difference; red: corrected evidence; dotted: program end',
                     fontsize=font_size + 1)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    return dict(figure=fig, axes=axes)


def plot_population_trajectories(analysis, side, fish_id, *, dimensions=(0, 1),
        colors=None, figsize=(8, 6), show_trials=True, marker_interval_s=2.0,
        event_times_s=None, font_size=9):
    """Plot shared PCA paths for one fish/side; dimensions are zero-based PCs.

    Means and optional faint single trials use the same fitted projection. Dots
    mark regular native samples, annotated in seconds relative to program start;
    start/end use circle/square. Optional label->seconds event markers are drawn
    on every path. Return figure and axes. No refit, saving or data mutation.
    """
    fit = analysis['fits'][side][fish_id]
    if (len(dimensions) not in (2, 3) or len(set(dimensions)) != len(dimensions)
            or any(i < 0 or i >= len(fit['explained_variance_ratio']) for i in dimensions)
            or not np.isfinite(marker_interval_s) or marker_interval_s <= 0):
        raise ValueError('Select two or three distinct available PCs and a positive marker interval.')
    colors = colors or {}
    with plt.rc_context({'font.size': font_size}):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d' if len(dimensions) == 3 else None)
        times = fit['time_s']
        for i, label in enumerate(fit['conditions']):
            color = colors.get(label, f'C{i}')
            path = fit['means'][label][:, dimensions]
            if show_trials:
                for trial in fit['trials'][label].transpose(1, 0, 2):
                    ax.plot(*trial[:, dimensions].T, color=color, alpha=0.15, linewidth=0.7)
            ax.plot(*path.T, color=color, linewidth=2, label=label)
            marker_times = np.arange(np.ceil(times[0] / marker_interval_s) * marker_interval_s,
                                     times[-1] + 1e-8, marker_interval_s)
            for target in marker_times:
                index = np.argmin(abs(times - target))
                point = path[index]
                ax.scatter(*point, color=color, s=13)
                if len(dimensions) == 2 and i == 0:
                    ax.annotate(f'{times[index]:g}', point, xytext=(4, 4),
                                textcoords='offset points', color=color, fontsize=font_size - 1)
            ax.scatter(*path[0], color=color, marker='o', s=45, edgecolor='black', linewidth=0.6)
            ax.scatter(*path[-1], color=color, marker='s', s=35, edgecolor='black', linewidth=0.6)
            for event, value in (event_times_s or {}).items():
                if times[0] <= value <= times[-1]:
                    point = path[np.argmin(abs(times - value))]
                    ax.scatter(*point, color=color, marker='X', s=65, edgecolor='black', linewidth=0.5)
        labels = [f'PC {d + 1} ({100 * fit["explained_variance_ratio"][d]:.1f}%)' for d in dimensions]
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
        if len(dimensions) == 3:
            ax.set_zlabel(labels[2])
        time_key = (f'seconds labeled on {fit["conditions"][0]}' if len(dimensions) == 2
                    else f'dots every {marker_interval_s:g} s')
        ax.set_title(f'{fish_id} · {side} · {len(fit["neuron_ids"])} neurons\n'
                     f'Circle: start; square: end; X: event; {time_key}')
        ax.legend(loc='best' if len(dimensions) == 2 else 'upper left', frameon=False)
        fig.tight_layout()
    return dict(figure=fig, axes=ax)


def plot_population_separation(analysis, side, *, significance=None, stimulus_events=None,
        figsize=(13, 8), show_fish=True, font_size=9):
    """Plot full-space RMS, crossvalidated squared distance and pattern correlation.

    Columns follow configured pairs. Faint lines are fish; thick curves are equal
    fish means with descriptive SEM. Dashed vertical lines mark CSV-derived first
    stimulus differences (pair->seconds); solid vertical lines mark program start.
    Red ticks show corrected permutation evidence on the squared-distance row.
    Return figure/axes without refitting, saving, or mutating the analysis.
    """
    pairs = analysis['side_conditions'][side]['pairs']
    metrics = [('rms_distance', 'RMS distance (baseline z)'),
               ('cv_squared_distance', 'Crossvalidated distance²\n(z² per neuron)'),
               ('pattern_correlation', 'Population pattern correlation')]
    with plt.rc_context({'font.size': font_size}):
        fig, axes = plt.subplots(3, len(pairs), figsize=figsize, sharex=True, squeeze=False)
        for col, (a, b) in enumerate(pairs):
            table = analysis['metrics'].query('side == @side & condition_a == @a & condition_b == @b')
            for row, (metric, ylabel) in enumerate(metrics):
                ax = axes[row, col]
                wide = table.pivot(index='time_s', columns='fish_id', values=metric)
                mean, sem = wide.mean(axis=1), wide.sem(axis=1)
                if show_fish:
                    ax.plot(wide.index, wide.to_numpy(), color='0.65', alpha=0.4, linewidth=0.7)
                ax.fill_between(wide.index, mean - sem, mean + sem, color='#385F73', alpha=0.18)
                ax.plot(wide.index, mean, color='#385F73', linewidth=2)
                ax.axvline(0, color='0.4', linewidth=0.8)
                event = (stimulus_events or {}).get((a, b), np.nan)
                if np.isfinite(event) and wide.index.min() <= event <= wide.index.max():
                    ax.axvline(event, color='#AD713F', linestyle='--', linewidth=1)
                ax.axhline(0, color='0.7', linewidth=0.6)
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == 0:
                    event_text = f'Asset difference: {event:g} s' if np.isfinite(event) else 'Identical assets'
                    ax.set_title(f'{a} vs {b}\n{event_text}', fontsize=font_size)
                if row == 1 and significance is not None:
                    sig = significance.query('side == @side & condition_a == @a & condition_b == @b & significant')
                    ax.plot(sig.time_s, np.full(len(sig), 0.97), '|', color='#B44444',
                            transform=ax.get_xaxis_transform(), markersize=6)
                if row == 2:
                    ax.set_ylim(-1.05, 1.05)
                    ax.set_xlabel('Time from program start (s)')
                ax.set_xlim(wide.index.min(), wide.index.max())
                ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle(f'{side.capitalize()} · full neuron space · equal fish weight\n'
                     'Gray: fish; blue: mean ± SEM; dashed: first stimulus difference; red: corrected evidence',
                     fontsize=font_size + 1)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    return dict(figure=fig, axes=axes)
