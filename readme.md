# Link Analysis Project

## Overview
This project is a comparative study of link analysis algorithms, specifically PageRank and HITS, along with their variations. The analysis is performed on a directed graph representing political blogs, where each blog belongs to either a liberal or conservative category.

## Requirements
- Python 3.x
- Required Libraries:
  - `networkx`
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `os`

Install dependencies using:
```bash
pip install networkx numpy matplotlib scipy
```

## Dataset
The dataset used is from Adamic and Glance's political blogs, which can be downloaded from:
[Political Blogs Dataset](https://websites.umich.edu/~mejn/netdata/polblogs.zip)

The dataset is in `.gml` format and represents a directed graph where:
- Nodes are blogs labeled as either liberal (0) or conservative (1).
- Edges represent hyperlinks between blogs.

## Implemented Algorithms
### PageRank Variations
1. **Weighted PageRank by Category**: Assigns double weight to edges between blogs of the same category.
2. **Weighted PageRank by Reciprocity**: Assigns double weight to reciprocal links.

### HITS Variations
1. **Modified HITS**: Instead of distributing the entire hub (or authority) score to each linked node, it is divided by the out-degree (or in-degree) of the node.

### Grouped PageRank and HITS
- Aggregates PageRank and HITS scores per category (liberal/conservative) to determine category influence.

## Project Structure
```
├── main.py                  # Main script for running experiments
├── pagerank_variations.py    # PageRank variations implementation
├── hits_variations.py        # HITS variations implementation
├── results/                  # Folder for storing results
├── histograms/               # Folder for storing histograms
```

## Execution
Run the project with:
```bash
python main.py
```

## Experiments and Analysis
The project performs the following tasks:
1. **Basic Network Analysis**:
   - Number of nodes, edges, and their degree statistics.
   - Number of blogs per category.

2. **PageRank with Different Parameters**:
   - Computes PageRank for different damping factors (`β = 0.5, 0.7, 0.9`).
   - Compares results using Spearman rank correlation for the entire directed graph, and overlap of top 10 nodes.

3. **Comparison of PageRank and HITS Variations**:
   - Computes and compares different variations.
   - Analyzes rank correlation and overlap among results.

4. **Score Distribution Analysis**:
   - Histograms of PageRank and HITS scores.

5. **Category-Based Influence**:
   - Aggregated scores for each category.

## Results and Visualization
- The results are stored in the `results/` folder.
- Histograms are saved in the `histograms/` folder.

## Author
**Thanos Moschou**

