import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import MDS
import re


def parse_output_file(output_path):
    """Extract GS-score and RMSD from a gsalign_output.txt file"""
    gs_score = rmsd = None
    try:
        with open(output_path, 'r') as f:
            content = f.read()

            # Use regex to robustly find values
            gs_match = re.search(r'GS-score:\s*([0-9.]+)', content)
            rmsd_match = re.search(r'RMSD.*?:\s*([0-9.]+)', content)

            if gs_match:
                gs_score = float(gs_match.group(1))
            if rmsd_match:
                rmsd = float(rmsd_match.group(1))
    except Exception as e:
        print(f"Error parsing {output_path}: {str(e)}")

    return gs_score, rmsd


def create_distance_matrices(output_dir):
    """Create distance matrices from all pairwise results"""
    # Find all PDB structures
    pdb_files = set()
    pair_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    for pair_dir in pair_dirs:
        if "_vs_" in pair_dir.name:
            pdb1, pdb2 = pair_dir.name.split("_vs_")
            pdb_files.add(pdb1)
            pdb_files.add(pdb2)

    pdb_list = sorted(pdb_files)
    n = len(pdb_list)
    pdb_index = {name: i for i, name in enumerate(pdb_list)}

    # Initialize matrices
    gs_matrix = np.zeros((n, n))
    rmsd_matrix = np.zeros((n, n))

    # Fill matrices with values from output files
    for pair_dir in pair_dirs:
        if "_vs_" not in pair_dir.name:
            continue

        try:
            pdb1, pdb2 = pair_dir.name.split("_vs_")
            i = pdb_index[pdb1]
            j = pdb_index[pdb2]

            output_file = pair_dir / "gsalign_output.txt"
            if output_file.exists():
                gs_score, rmsd = parse_output_file(output_file)

                if gs_score is not None:
                    gs_matrix[i, j] = gs_matrix[j, i] = 1 - gs_score
                if rmsd is not None:
                    rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd
        except Exception as e:
            print(f"Error processing {pair_dir}: {str(e)}")

    return gs_matrix, rmsd_matrix, pdb_list


def load_metadata(metadata_path, pdb_names):
    """Load metadata from CSV and validate against PDB names"""
    try:
        metadata = pd.read_csv(metadata_path, index_col=0)
        missing = set(pdb_names) - set(metadata.index)
        if missing:
            print(f"Warning: Missing metadata for {len(missing)} structures: {missing}")
        return metadata
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return None


def visualize_mds(gs_matrix, rmsd_matrix, pdb_names, output_dir, metadata=None, color_by=None):
    """Create MDS visualizations with discrete metadata coloring"""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate both plots
    for name, matrix in [('GS_Score', gs_matrix), ('RMSD', rmsd_matrix)]:
        try:
            # Skip if matrix contains only zeros
            if np.all(matrix == 0):
                print(f"Skipping {name} MDS - all distances are zero")
                continue

            # Add small value to avoid zero distances
            matrix += 1e-9

            # Perform MDS
            mds = MDS(n_components=2,
                      dissimilarity='precomputed',
                      normalized_stress='auto',
                      random_state=42)
            coords = mds.fit_transform(matrix)

            # Create plot
            plt.figure(figsize=(10, 8))

            # Add colored points if metadata is provided
            if metadata is not None and color_by in metadata.columns:
                # Get unique categories and assign colors
                categories = metadata.loc[pdb_names, color_by]
                unique_categories = categories.unique()
                color_map = {cat: color for cat, color in
                             zip(unique_categories, plt.cm.tab20.colors)}

                # Plot each category separately
                for cat, color in color_map.items():
                    mask = categories == cat
                    plt.scatter(coords[mask, 0], coords[mask, 1],
                                color=color, label=cat, s=100, alpha=0.7)

                # Add legend
                plt.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                plt.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.7)

            # Add labels
            for i, label in enumerate(pdb_names):
                plt.text(coords[i, 0], coords[i, 1], label,
                         fontsize=8, ha='center', va='bottom')

            plt.title(f'MDS Visualization ({name} Distance)\nStress: {mds.stress_:.2f}')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.grid(alpha=0.3)
            plt.tight_layout()

            # Save plot
            plot_path = output_dir / f"mds_{name.lower()}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {name} MDS plot to {plot_path}")

        except Exception as e:
            print(f"Error creating {name} MDS plot: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GS-align results with discrete metadata coloring")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Main output directory containing pairwise result folders")
    parser.add_argument("-a", "--analysis_dir", required=True,
                        help="Directory to save analysis results")
    parser.add_argument("-m", "--metadata",
                        help="Path to metadata CSV file")
    parser.add_argument("-c", "--color_by",
                        help="Column name in metadata to use for coloring")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir)

    # Create analysis directory if it doesn't exist
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Create distance matrices
    gs_matrix, rmsd_matrix, pdb_names = create_distance_matrices(output_dir)

    # Save matrices as CSV
    pd.DataFrame(gs_matrix, index=pdb_names, columns=pdb_names
                 ).to_csv(analysis_dir / "gs_distance_matrix.csv")
    pd.DataFrame(rmsd_matrix, index=pdb_names, columns=pdb_names
                 ).to_csv(analysis_dir / "rmsd_distance_matrix.csv")

    # Load metadata if provided
    metadata = None
    if args.metadata:
        metadata = load_metadata(args.metadata, pdb_names)

    # Generate MDS plots
    visualize_mds(gs_matrix, rmsd_matrix, pdb_names, analysis_dir,
                  metadata=metadata, color_by=args.color_by)
    print(f"Analysis results saved to: {analysis_dir}")


if __name__ == "__main__":
    main()