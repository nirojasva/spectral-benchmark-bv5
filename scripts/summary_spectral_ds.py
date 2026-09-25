import pandas as pd
from pathlib import Path

current_dir = Path.cwd()
target_paths = [
    current_dir / "datasets" / "raw" / "ScenariosV5_test",
]

MAX_VALUE_CURRENT = 8.25
MIN_VALUE_CURRENT = 7.75

scenario_order = ["SAR1", "SAR2", "SAR3", "TNH", "TAR", "HNH", "HAR", "LNH", "LAR", "RNH", "RAR", "RMX"]

condition_map = {
    "SAR1": "C", "SAR2": "C", "SAR3": "C",
    "TNH": "C", "TAR": "C",
    "HNH": "D", "HAR": "D", "LNH": "D", "LAR": "D",
    "RNH": "D", "RAR": "D", "RMX": "D"
}

def generate_detailed_report(folder_path):
    report_data = []

    for file_path in folder_path.glob("*.csv"):
        try:
            df = pd.read_csv(
                file_path,
                sep=',',
                low_memory=False,
                dtype={'CURRENTTIMESTAMP': str}
            )

            current_col = "CURRENT"
            df[current_col] = pd.to_numeric(df[current_col], errors='coerce')
            
            mask_in_range = (df[current_col] >= MIN_VALUE_CURRENT) & (df[current_col] <= MAX_VALUE_CURRENT)

            target_col = 'ANOMALY?' if 'ANOMALY?' in df.columns else df.columns[-1]
            
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
            
            target_series_overall = df[target_col].fillna(0).astype(int)
            target_series_narrowed = (target_series_overall & mask_in_range).astype(int)
            
            total_rows = mask_in_range.sum()
            total_ones = target_series_narrowed.sum()
            
            condition_changes = (target_series_narrowed != target_series_narrowed.shift()).cumsum()
            group_sums = target_series_narrowed.groupby(condition_changes).sum()
            unique_events = len(group_sums[group_sums > 0])
            
            raw_pct_sf = (total_ones / total_rows * 100) if total_rows > 0 else 0
            raw_pct = (len(df[df['ANOMALY?'] == 1]) / len(df) * 100) if len(df) > 0 else 0
            
            stem = file_path.stem
            scenario_name = stem.split("_")[0]
            condition_val = condition_map.get(scenario_name, "-")

            report_data.append({
                'Scenario': scenario_name,
                '\\makecell{Total Dur.\\\\Anom. (sec.)}': total_ones,
                '\\makecell{\\#\\\\Anom.}': unique_events,
                '\\makecell{\\# Instances\\\\(sec.)}': total_rows,
                '\\makecell{Condition\\\\(Constant or\\\\Degassing)}': condition_val,
                '\\makecell{\\% Anom.\\\\SF}': f"{raw_pct_sf:.2f}\\%",
                '\\makecell{\\% Anom.\\\\Total}': f"{raw_pct:.2f}\\%"
            })
                
        except Exception:
            pass

    if report_data:
        results_df = pd.DataFrame(report_data)
        
        results_df['Scenario_Cat'] = pd.Categorical(results_df['Scenario'], categories=scenario_order, ordered=True)
        results_df = results_df.sort_values('Scenario_Cat').drop(columns=['Scenario_Cat'])
        
        latex_str = results_df.to_latex(index=False, escape=False, column_format="l cc c c c c")
        
        lines = latex_str.split('\n')
        new_lines = []
        for line in lines:
            if line.strip() == '\\toprule':
                new_lines.append(line)
                new_lines.append('& \\multicolumn{2}{c}{Anomaly Characteristics} & & & & \\\\')
                new_lines.append('\\cmidrule(r){2-3}')
                continue
            if line.lstrip().startswith('TNH &') or line.lstrip().startswith('HNH &') or line.lstrip().startswith('RNH &'):
                new_lines.append('    \\midrule')
            new_lines.append(line)
            
        latex_str = '\n'.join(new_lines)
        latex_str = "\\resizebox{\\textwidth}{!}{\n" + latex_str + "}"
        
        final_table = (
            "\\begin{table}[!h]\n"
            "\\centering\n"
            "\\caption{Effective data used with SF, with the current set between 7.75 and 8.25~mA.}\n"
            "\\label{tab:dataset_spectra_bv5_with_sf}\n"
            f"{latex_str}\n"
            "\\end{table}"
        )
        
        print(final_table)

if __name__ == "__main__":
    for path in target_paths:
        if path.exists() and path.is_dir():
            generate_detailed_report(path)