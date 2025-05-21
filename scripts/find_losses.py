import os
import json
import csv
import pandas as pd
import re
import fnmatch

PATTERNS = {
            "*airbnb*_emb*": "AirBnB",
            "*walmart*_emb*": "Walmart",
            "*rossmann*_emb*": "Rossmann",
            "*imdb*_emb*": "IMDB MovieLens"
            }

for PATTERN in PATTERNS.keys():

    root_dir = "artifacts/models"
    output_csv = "losses_min.csv"
    results = []


    def extract_graph_embedding_dim(config_path):
        try:
            with open(config_path, "r") as f:
                for line in f:
                    m = re.search(r"GRAPH_EMBEDDING_DIM\s*=\s*(\d+)", line)
                    if m:
                        return int(m.group(1))
        except Exception as e:
            print(f"Error reading {config_path}: {e}")
        return None

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "losses.json" in filenames:
            rel_path = os.path.relpath(dirpath, root_dir)
            # Apply fnmatch filter to the relative path
            if not fnmatch.fnmatch(rel_path, PATTERN):
                continue
            losses_path = os.path.join(dirpath, "losses.json")
            try:
                with open(losses_path, "r") as f:
                    losses = json.load(f)
                val_min = losses["validation"]["min"]
                # Extract exp_name: get the folder name after the timestamp
                m = re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}_\d+_(.+?)(/|$)", rel_path)
                if m:
                    exp_name = m.group(1)
                    date_str = rel_path.split("_")[0]
                else:
                    exp_name = rel_path
                    date_str = ""
                # Always look for config.py in the experiment root (the first part of rel_path)
                exp_root = os.path.join(root_dir, rel_path.split(os.sep)[0])
                config_path = os.path.join(exp_root, "config.py")
                graph_emb_dim = extract_graph_embedding_dim(config_path)
                results.append((date_str, exp_name, val_min, graph_emb_dim))
            except Exception as e:
                print(f"Error reading {losses_path}: {e}")

    # Sort by date
    results.sort(key=lambda x: x[0])

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["exp_name", "Val", "GRAPH_EMBEDDING_DIM"])
        for _, exp_name, val_min, graph_emb_dim in results:
            writer.writerow([exp_name, val_min, graph_emb_dim])

    # Load and print with pandas
    df = pd.read_csv(output_csv).sort_values(by="GRAPH_EMBEDDING_DIM")
    print(df)

    import matplotlib.pyplot as plt


    # Extract a short name for the pattern for the filename
    pattern_name = PATTERNS[PATTERN]

    # Plot GRAPH_EMBEDDING_DIM vs Val
    height = 3
    width = height*1.5
    plt.figure(figsize=(width, height))
    plt.plot(df["GRAPH_EMBEDDING_DIM"], df["Val"], marker='.', linestyle='-', alpha=0.95)  # Line + points
    plt.xlabel("GNN Embedding Dimension")
    plt.ylabel("Validation Loss")
    plt.title(pattern_name)
    plt.grid(True)
    plt.tight_layout()
    #plt.xticks(sorted(df["GRAPH_EMBEDDING_DIM"].unique()))  # Only x values as ticks

    #plt.xticks(sorted(df["GRAPH_EMBEDDING_DIM"].unique()))  # Show actual x values

    for i in range(len(df)):

        emb_dim = df.iloc[i]["GRAPH_EMBEDDING_DIM"]
        val_loss = df.iloc[i]["Val"]
        plt.annotate(
            str(emb_dim),
            (emb_dim, val_loss),
            textcoords="offset points",
            xytext=(5-1.2*i, -2+i),  # Small shift right and up
            ha='left',
            va='bottom',
            fontsize=8
        )

    plt.savefig(f"embedding_vs_val_{pattern_name}.pdf")
    plt.show()
    plt.close()