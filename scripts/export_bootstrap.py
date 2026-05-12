import pandas as pd
from pathlib import Path

def export_bootstrap(base_dir: str):
    base_path = Path(base_dir)
    in_file = base_path / "per_species" / "model_comparison.csv"

    if in_file.exists():
        df = pd.read_csv(in_file)

        cols_to_extract = ['species', 'mode', 'variant']
        format_cols = []

        if 'boot_ci_lo' in df.columns and 'boot_ci_hi' in df.columns:
            cols_to_extract.extend(['boot_ci_lo', 'boot_ci_hi'])
            format_cols.extend(['boot_ci_lo', 'boot_ci_hi'])

        if 'block_ci_lo' in df.columns and 'block_ci_hi' in df.columns:
            cols_to_extract.extend(['block_ci_lo', 'block_ci_hi'])
            format_cols.extend(['block_ci_lo', 'block_ci_hi'])

        if len(cols_to_extract) > 3:
            # Select only identifiers and bootstrap columns
            df_boot = df[cols_to_extract].copy()

            # Format to 4 decimal places for readability
            for col in format_cols:
                df_boot[col] = df_boot[col].map(lambda x: f"{x:+.4f}")

            # Save to a new document
            out_file = base_path / "bootstrap_summary.csv"
            df_boot.to_csv(out_file, index=False)
            print(f"Successfully generated: {out_file}")

            # Formatted text table version for easy reading
            out_txt = base_path / "bootstrap_summary.txt"
            with open(out_txt, 'w') as f:
                f.write(df_boot.to_string(index=False))
            print(f"Successfully generated: {out_txt}")

            # Formatted markdown table version for easy reading
            out_md = base_path / "bootstrap_summary.md"
            with open(out_md, 'w') as f:
                try:
                    f.write(df_boot.to_markdown(index=False))
                    print(f"Successfully generated: {out_md}")
                except ImportError:
                    print("Markdown output skipped (tabulate not installed).")
        else:
            print(f"Bootstrap columns not found in {in_file}")
    else:
        print(f"File not found: {in_file}")

if __name__ == "__main__":
    export_bootstrap("outputs")
    export_bootstrap("outputs_year")
