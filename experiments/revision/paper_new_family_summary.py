"""Concise main-text evidence from completed, validation-selected task searches."""

def build_summary(complete, methods, display):
    if not complete:
        return (
            r'\paragraph{Prospective task status.} '
            'The six declared settings are running; complete comparisons will '
            'be reported after each five-method search finishes.\n'
        )
    wins = []
    for case in complete:
        selected = case['selected']
        better = (lambda x, y: x < y) if case['lower'] else (lambda x, y: x > y)
        if better(selected['strata']['test'], selected['single']['test']) or better(
            selected['strata']['test'], selected['adamw']['test']
        ):
            wins.append(case)
    sg_wins = sum(
        int(c['selected']['strata']['test'] < c['selected']['single']['test']
            if c['lower'] else c['selected']['strata']['test'] > c['selected']['single']['test'])
        for c in complete
    )
    intro = (
        'All 180 configurations across six settings have completed their 1000-epoch budgets.'
        if len(complete) == 6 and not any(c['failed'] for c in complete)
        else f'{len(complete)} of six settings have exhausted their thirty-candidate searches.'
    )
    text = r'\paragraph{Transfer to forecasting, recommendation, and sensor drift.} ' + intro + ' '
    text += (
        'Each of five methods selects among six candidates using terminal-100 validation metrics '
        f'(Appendix~\\ref{{app:prospective-real-tasks}}). STRATA improves on same-rank '
        f'Single-Grassmann in {sg_wins}/{len(complete)} completed settings. '
        'Table~\\ref{tab:new-task-positive} highlights these positive cases after observing the complete matrix; '
        'all settings, including ETTh1, appear in Table~\\ref{tab:new-families-complete}. '
        'The results use seed 42 and describe the tested configurations.\n'
    )
    if wins:
        text += (
            r'\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}' '\n'
            r'\caption{\textbf{Completed new-task positive cases.} Terminal-100 test metrics after equal-count validation search. '
            r'All five methods are shown; bold marks the best value. MSE/RMSE are lower-is-better and accuracy is higher-is-better. '
            r'The full six-setting matrix and selected losses are in Appendix~\ref{app:new-family-results}.}' '\n'
            r'\label{tab:new-task-positive}\begin{adjustbox}{max width=\linewidth}' '\n'
            r'\begin{tabular}{llrrrrr}\toprule Setting & Metric & STRATA & Single-G & AdamW & GaLore & Indep.\\\midrule' '\n'
        )
        for case in wins:
            _, task, base = case['key']
            selected = case['selected']
            best = (min if case['lower'] else max)(selected[m]['test'] for m in methods)
            cells = []
            for method in methods:
                value = selected[method]['test']
                cell = f'{value:.5f}'
                if value == best:
                    cell = r'\textbf{' + cell + '}'
                if method == 'strata':
                    cell = r'\cellcolor{oursbg}' + cell
                cells.append(cell)
            text += display(task, base) + ' & ' + selected['strata']['metric'] + ' & ' + ' & '.join(cells) + r' \\' + '\n'
        text += r'\bottomrule\end{tabular}\end{adjustbox}\end{table*}' + '\n'
    cases = {c['key'][1:]: c for c in complete}
    if ('ETTh2', 'static') in cases:
        s = cases[('ETTh2', 'static')]['selected']
        if all(s['strata']['test'] < s[m]['test'] for m in methods if m != 'strata'):
            gain = {m: 100 * (1 - s['strata']['test'] / s[m]['test']) for m in ['single', 'adamw', 'galore']}
            text += (
                r'\paragraph{Forecasting.} '
                f"On ETTh2/H24, STRATA attains MSE {s['strata']['test']:.5f}, the lowest among the five methods, "
                f"reducing error by {gain['single']:.1f}\\%, {gain['adamw']:.1f}\\%, and {gain['galore']:.1f}\\% "
                'relative to Single-Grassmann, AdamW, and GaLore, respectively. '
                f"It stores {100*s['strata']['state_bytes']/s['adamw']['state_bytes']:.1f}\\% of AdamW's optimizer tensor state. "
                'The common forecaster is an MLP; this comparison tests the optimizer under the specified chronological split.\n'
            )
    if all((task, 'static') in cases for task in ['ml100k', 'ml1m']):
        a, b = [cases[(task, 'static')]['selected'] for task in ['ml100k', 'ml1m']]
        if all(s['strata']['test'] < s['single']['test'] for s in [a, b]):
            text += (
                r'\paragraph{Recommendation.} '
                f"STRATA reduces RMSE relative to Single-Grassmann by {100*(1-a['strata']['test']/a['single']['test']):.2f}\\% on MovieLens100K "
                f"and {100*(1-b['strata']['test']/b['single']['test']):.2f}\\% on MovieLens1M. "
                'AdamW and GaLore remain stronger on both datasets.\n'
            )
    if all(('gas_drift', base) in cases for base in ['finetune', 'replay']):
        a, b = [cases[('gas_drift', base)]['selected'] for base in ['finetune', 'replay']]
        text += (
            r'\paragraph{Sensor drift.} '
            f"Fine-tuning gives {100*a['strata']['test']:.3f}\\% accuracy, {100*(a['strata']['test']-a['adamw']['test']):.3f} "
            'percentage points above AdamW, while GaLore is stronger. '
            f"With replay, STRATA reaches {100*b['strata']['test']:.3f}\\%, versus AdamW's {100*b['adamw']['test']:.3f}\\%, "
            f"using {100*b['strata']['state_bytes']/b['adamw']['state_bytes']:.1f}\\% of its stored optimizer state. "
            'State counts include probes and buffers. These same-rank comparisons do not equalize geometric-method state, '
            'and stored-state savings do not imply peak-memory or throughput gains.\n'
        )
    if wins:
        # Declare the double-column float before the prose so it stays with
        # the main-text discussion when adjacent experiment paragraphs grow.
        start = text.index(r'\begin{table*}')
        end = text.index(r'\end{table*}', start) + len(r'\end{table*}')
        text = text[start:end] + '\n' + text[:start] + text[end:].lstrip('\n')
    return text
