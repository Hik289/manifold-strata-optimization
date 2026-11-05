"""Shared WWW-style tables; green identifies STRATA, bold identifies a stated optimum."""
import re

def format_tex(text):
    # Linewise row styling is intentionally restricted to table environments.
    def style(match):
        block = match.group(0).replace(r'\small', r'\footnotesize')
        block = block.replace(r'\rowcolor{stratablue!9}', r'\rowcolor{oursbg}')
        lines = block.splitlines()
        ours_column = None
        for line in lines:
            if '&' in line and re.search(r'&\s*(?:\\textbf\{)?STRATA\b', line):
                cells = line.split('&')
                ours_column = next((i for i,c in enumerate(cells) if re.search(r'\bSTRATA\b',c)), None)
                break
        output = []
        for line in lines:
            if '&' in line and r'\\' in line and r'\caption' not in line:
                cells = line.split('&')
                first = cells[0]
                is_ours = (bool(re.search(r'\bSTRATA\b', first)) or first.strip() in ['full', 'Full STRATA']
                           or any(c.strip()=='strata' for c in cells[1:]))
                if is_ours and not any(x in first for x in [r'\toprule', r'\begin', r'\textbf{Method}']):
                    if not (output and r'\rowcolor' in output[-1]) and r'\rowcolor' not in line:
                        line = r'\rowcolor{oursbg}' + '\n' + line
                elif ours_column is not None and ours_column < len(cells):
                    if re.search(r'\d', cells[ours_column]) and r'\cellcolor' not in cells[ours_column]:
                        cells[ours_column] = r' \cellcolor{oursbg}' + cells[ours_column].lstrip()
                        line = '&'.join(cells)
            output.append(line)
        block = '\n'.join(output)
        if any(label in block for label in [r'\label{tab:positive-cases}',r'\label{tab:pubmed-resources}']):
            # Recompute highlights from all rows, including previously bold optima.
            block=re.sub(r'\\textbf\{([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\}',r'\1',block)
        number = re.compile(r'^(\s*(?:\\cellcolor\{oursbg\})?)([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)(\s*(?:\\\\)?)$')
        # Bold has a numerical meaning; green alone identifies the proposed method.
        if r'\label{tab:positive-cases}' in block:
            result=[]
            for line in block.splitlines():
                cells=line.split('&');values=[number.fullmatch(c) for c in cells[1:]]
                if len(cells)==6 and all(values):
                    best=max(float(m[2]) for m in values)
                    cells[1:]=[m[1]+r'\textbf{'+m[2]+'}'+m[3] if float(m[2])==best else c for m,c in zip(values,cells[1:])]
                    line='&'.join(cells)
                result.append(line)
            block='\n'.join(result)
        if r'\label{tab:pubmed-resources}' in block:
            lines=block.splitlines();records=[]
            for i,line in enumerate(lines):
                cells=line.split('&');values=[number.fullmatch(c) for c in cells[1:]]
                if len(cells)==6 and all(values):records.append((i,cells,values))
            if records:
                best=[(max if j<2 else min)(float(v[j][2]) for _,_,v in records) for j in range(5)]
                for i,cells,values in records:
                    cells[1:]=[m[1]+r'\textbf{'+m[2]+'}'+m[3] if float(m[2])==best[j] else c for j,(m,c) in enumerate(zip(values,cells[1:]))]
                    lines[i]='&'.join(cells)
                block='\n'.join(lines)
        # Scale down only when required; never stretch a small table.
        if r'\begin{tabular}' in block and not any(x in block for x in [r'\resizebox', r'\begin{adjustbox}']):
            block = block.replace(r'\begin{tabular}', r'\begin{adjustbox}{max width=\linewidth}'+'\n'+r'\begin{tabular}')
            block = block.replace(r'\end{tabular}', r'\end{tabular}'+'\n'+r'\end{adjustbox}')
        return block
    return re.sub(r'\\begin\{(table\*?|longtable)\}(?:\[[^\]]*\])?[\s\S]*?\\end\{\1\}', style, text)

def loss_style(plt):
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':8, 'axes.titlesize':9,
        'axes.labelsize':8, 'legend.fontsize':7, 'axes.spines.top':False,
        'axes.spines.right':False, 'pdf.fonttype':42, 'ps.fonttype':42,
        'savefig.bbox':'tight', 'grid.alpha':.18,
    })
