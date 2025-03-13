"""
Author: Thanos Moschou
Description: This is a programming project for Text Analysis and Search Engines course of UoM.

We use networkx library.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr
import os
from pagerank_variations import *
from hits_variations import *

# Helping function for printing histograms
def plot_histogram(scores, title, color):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Frequency')


# Task I
# General graph network analysis
def taskI(G):
    # Calculate liberal nodes and conservative ones
    liberalNodes = 0
    conservativeNodes = 0

    for node, data in G.nodes(data=True):
        if data['value'] == 0:
            liberalNodes += 1
        else:
            conservativeNodes += 1

    # Calculate in-degree and out-degree of each node and then do some statistics.
    inDegrees = [G.in_degree(node) for node in G]
    outDegrees = [G.out_degree(node) for node in G]

    minIn = np.min(inDegrees)
    maxIn = np.max(inDegrees)
    avgIn = np.mean(inDegrees)

    minOut = np.min(outDegrees)
    maxOut = np.max(outDegrees)
    avgOut = np.mean(outDegrees)

    # Save the results
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'results')
    os.makedirs(file_path, exist_ok = True)

    with open(f"{file_path}/taskI.txt", "w") as filename:
        filename.write("Task I:\n\n")

        filename.write(f"Total nodes: {G.number_of_nodes()}\n\n")

        filename.write(f"Liberal nodes: {liberalNodes}\n")
        filename.write(f"Conservative nodes: {conservativeNodes}\n\n")

        filename.write(f"Number of edges: {G.number_of_edges()}\n\n")

        filename.write(f"Min in-degree: {minIn}\n")
        filename.write(f"Max in-degree: {maxIn}\n\n")
        filename.write(f"Min out-degree: {minOut}\n")
        filename.write(f"Max out-degree: {maxOut}\n\n")
        filename.write(f"Avg in-degree: {avgIn}\n")
        filename.write(f"Avg out-degree: {avgOut}\n")
    
    print("Task I results are saved successfully in file: taskI.txt inside results folder")


# Task II
# Apply built-in pagerank algorithm with different b parameters
# and calculate spearman rank correlation and overlap 
def taskII(G):
    alphaValues = [0.5, 0.7, 0.9] # b values
    pageRankResults = {} # key is a b value and value is a list with ranks
    nodeLabels = {} # key is a b value and value is a list with node labels

    for alpha in alphaValues:
        pr = nx.pagerank(G, alpha)

        # pr.items() returns a list of tuples. 
        # Index 1 contains the rank of the node inside each tuple. 
        # Sort this list of tuples based on this rank.
        sortedItems = sorted(pr.items(), key=lambda item: item[1], reverse=True)

        top10List = sortedItems[:10]
        
        sortedLabels = [item[0] for item in top10List] # Top 10 labels
        sortedRanks = [item[1] for item in sortedItems]
        
        # Store the results
        nodeLabels[str(alpha)] = sortedLabels
        pageRankResults[str(alpha)] = sortedRanks

    spearmanRankCorr_05_07 = spearmanr(pageRankResults["0.5"], pageRankResults["0.7"])
    spearmanRankCorr_05_09 = spearmanr(pageRankResults["0.5"], pageRankResults["0.9"])
    spearmanRankCorr_07_09 = spearmanr(pageRankResults["0.7"], pageRankResults["0.9"])

    overlap_05_07 = len(set(nodeLabels["0.5"]) & set(nodeLabels["0.7"]))
    overlap_05_09 = len(set(nodeLabels["0.5"]) & set(nodeLabels["0.9"]))
    overlap_07_09 = len(set(nodeLabels["0.7"]) & set(nodeLabels["0.9"]))

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'results')
    
    # Save the results
    with open(f"{file_path}/taskII.txt", "w") as filename:
        filename.write("Task II:\n\n")

        filename.write(f"Spearman rank correlation between alpha = 0.5 and alpha = 0.7: {spearmanRankCorr_05_07.statistic} with p-value: {spearmanRankCorr_05_07.pvalue}\n")
        filename.write(f"Spearman rank correlation between alpha = 0.5 and alpha = 0.9: {spearmanRankCorr_05_09.statistic} with p-value: {spearmanRankCorr_05_09.pvalue}\n")
        filename.write(f"Spearman rank correlation between alpha = 0.7 and alpha = 0.9: {spearmanRankCorr_07_09.statistic} with p-value: {spearmanRankCorr_07_09.pvalue}\n\n")

        filename.write(f"Overlap between alpha = 0.5 and alpha = 0.7: {overlap_05_07}\n")
        filename.write(f"Overlap between alpha = 0.5 and alpha = 0.9: {overlap_05_09}\n")
        filename.write(f"Overlap between alpha = 0.7 and alpha = 0.9: {overlap_07_09}\n")

    print("Task II results are saved successfully in file: taskII.txt inside results folder")

    # Task IV
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'histograms')
    os.makedirs(file_path, exist_ok = True)

    pr_05 = list(pageRankResults["0.5"])
    pr_07 = list(pageRankResults["0.7"])
    pr_09 = list(pageRankResults["0.9"])

    plot_histogram(pr_05, 'Scores of Original PageRank with β = 0.5', 'skyblue')
    plt.savefig(f"{file_path}/taskIIpagerankOrig05.png", format = "png")
    plt.clf()

    plot_histogram(pr_07, 'Scores of Original PageRank with β = 0.7', 'lightgreen')
    plt.savefig(f"{file_path}/taskIIpagerankOrig07.png", format = "png")
    plt.clf()

    plot_histogram(pr_09, 'Scores of Original PageRank with β = 0.9', 'salmon')
    plt.savefig(f"{file_path}/taskIIpagerankOrig09.png", format = "png")
    plt.clf()

    print("Histograms for task II are under histograms folder.")


# Task III
# Apply pagerank, hits and all their variations and calculate spearman rank correlation and overlap.
def taskIII(G):
    # Pagerank and HITS 
    prVariation1 = pagerank_variation_double_weight_for_same_category(G.copy(), alpha = 0.8)
    prVariation2 = pagerank_variation_double_weight_for_reciprocal_edges(G.copy(), alpha = 0.8)
    prOriginal = nx.pagerank(G.copy(), 0.8)

    hitsVariation = hits_variation_in_out_degree(G)
    hitsOriginal = nx.hits(G)

    hubsVar, authVar = hitsVariation # Destructure the tuple
    hubsOrig, authOrig = hitsOriginal

    # Sort the items and get the top 10
    sortedPrVar1 = sorted(prVariation1.items(), key=lambda item: item[1], reverse=True)
    top10ListPrVariation1 = sortedPrVar1[:10]

    sortedPrVar2 = sorted(prVariation2.items(), key=lambda item: item[1], reverse=True)
    top10ListPrVariation2 = sortedPrVar2[:10]

    sortedPrOrig = sorted(prOriginal.items(), key=lambda item: item[1], reverse=True)
    top10ListPrOriginal = sortedPrOrig[:10]

    sortedHubsHitsVar = sorted(hubsVar.items(), key=lambda item: item[1], reverse=True)
    top_10_hubsHitsVariation = sortedHubsHitsVar[:10]

    sortedAuthHitsVar = sorted(authVar.items(), key=lambda item: item[1], reverse=True)
    top_10_authoritiesHitsVariation = sortedAuthHitsVar[:10]

    sortedHubsHitsOrig = sorted(hubsOrig.items(), key=lambda item: item[1], reverse=True)
    top_10_hubsHitsOriginal = sortedHubsHitsOrig[:10]

    sortedAuthHitsOrig = sorted(authOrig.items(), key=lambda item: item[1], reverse=True)
    top_10_authoritiesHitsOriginal = sortedAuthHitsOrig[:10]

    # Get the labels and the ranks
    labelsPrVariation1 = [item[0] for item in top10ListPrVariation1] # Top 10 labels of the first variation
    ranksPrVariation1 = [item[1] for item in sortedPrVar1]

    labelsPrVariation2 = [item[0] for item in top10ListPrVariation2] # Top 10 labels of the second variation
    ranksPrVariation2 = [item[1] for item in sortedPrVar2]

    labelsPrOriginal = [item[0] for item in top10ListPrOriginal] # Top 10 labels of the original implementation
    ranksPrOriginal = [item[1] for item in sortedPrOrig]

    hVarLabels = [item[0] for item in top_10_hubsHitsVariation]
    hVarScores = [item[1] for item in sortedHubsHitsVar]

    aVarLabels = [item[0] for item in top_10_authoritiesHitsVariation]
    aVarScores = [item[1] for item in sortedAuthHitsVar]

    hOriginalLabels = [item[0] for item in top_10_hubsHitsOriginal] 
    hOriginalScores = [item[1] for item in sortedHubsHitsOrig]

    aOriginalLabels = [item[0] for item in top_10_authoritiesHitsOriginal]
    aOriginalScores = [item[1] for item in sortedAuthHitsOrig]

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'results')

    # Calculate spearman rank correlation and overlap by pairs. Finally save the result to a file
    with open(f"{file_path}/taskIII.txt", "w") as filename:
        filename.write(f"1st Variation of Pagerank vs 2nd Variation of Pagerank\nSpearman: {spearmanr(ranksPrVariation1, ranksPrVariation2)}\nOverlap: {len(set(labelsPrVariation1) & set(labelsPrVariation2))}\n\n")
        filename.write(f"1st Variation of Pagerank vs Original Pagerank\nSpearman: {spearmanr(ranksPrVariation1, ranksPrOriginal)}\nOverlap: {len(set(labelsPrVariation1) & set(labelsPrOriginal))}\n\n")
        filename.write(f"1st Variation of Pagerank vs Hub Scores of HITS Variation\nSpearman: {spearmanr(ranksPrVariation1, hVarScores)}\nOverlap: {len(set(labelsPrVariation1) & set(hVarLabels))}\n\n")
        filename.write(f"1st Variation of Pagerank vs Authority Scores of HITS Variation\nSpearman: {spearmanr(ranksPrVariation1, aVarScores)}\nOverlap: {len(set(labelsPrVariation1) & set(aVarLabels))}\n\n")
        filename.write(f"1st Variation of Pagerank vs Hub Scores of HITS Original\nSpearman: {spearmanr(ranksPrVariation1, hOriginalScores)}\nOverlap: {len(set(labelsPrVariation1) & set(hOriginalLabels))}\n\n")
        filename.write(f"1st Variation of Pagerank vs Authority Scores of HITS Original\nSpearman: {spearmanr(ranksPrVariation1, aOriginalScores)}\nOverlap: {len(set(labelsPrVariation1) & set(aOriginalLabels))}\n\n")
        
        filename.write(f"2nd Variation of Pagerank vs Original Pagerank\nSpearman: {spearmanr(ranksPrVariation2, ranksPrOriginal)}\nOverlap: {len(set(labelsPrVariation2) & set(labelsPrOriginal))}\n\n")
        filename.write(f"2nd Variation of Pagerank vs Hub Scores of HITS Variation\nSpearman: {spearmanr(ranksPrVariation2, hVarScores)}\nOverlap: {len(set(labelsPrVariation2) & set(hVarLabels))}\n\n")
        filename.write(f"2nd Variation of Pagerank vs Authority Scores of HITS Variation\nSpearman: {spearmanr(ranksPrVariation2, aVarScores)}\nOverlap: {len(set(labelsPrVariation2) & set(aVarLabels))}\n\n")
        filename.write(f"2nd Variation of Pagerank vs Hub Scores of HITS Original\nSpearman: {spearmanr(ranksPrVariation2, hOriginalScores)}\nOverlap: {len(set(labelsPrVariation2) & set(hOriginalLabels))}\n\n")
        filename.write(f"2nd Variation of Pagerank vs Authority Scores of HITS Original\nSpearman: {spearmanr(ranksPrVariation2, aOriginalScores)}\nOverlap: {len(set(labelsPrVariation2) & set(aOriginalLabels))}\n\n")

        filename.write(f"Original Pagerank vs Hub Scores of HITS Variation\nSpearman: {spearmanr(ranksPrOriginal, hVarScores)}\nOverlap: {len(set(labelsPrOriginal) & set(hVarLabels))}\n\n")
        filename.write(f"Original Pagerank vs Authority Scores of HITS Variation\nSpearman: {spearmanr(ranksPrOriginal, aVarScores)}\nOverlap: {len(set(labelsPrOriginal) & set(aVarLabels))}\n\n")
        filename.write(f"Original Pagerank vs Hub Scores of HITS Original\nSpearman: {spearmanr(ranksPrOriginal, hOriginalScores)}\nOverlap: {len(set(labelsPrOriginal) & set(hOriginalLabels))}\n\n")
        filename.write(f"Original Pagerank vs Authority Scores of HITS Original\nSpearman: {spearmanr(ranksPrOriginal, aOriginalScores)}\nOverlap: {len(set(labelsPrOriginal) & set(aOriginalLabels))}\n\n")

        filename.write(f"Hub Scores of HITS Variation vs Authority Scores of HITS Variation\nSpearman: {spearmanr(hVarScores, aVarScores)}\nOverlap: {len(set(hVarLabels) & set(aVarLabels))}\n\n")
        filename.write(f"Hub Scores of HITS Variation vs Hub Scores of HITS Original\nSpearman: {spearmanr(hVarScores, hOriginalScores)}\nOverlap: {len(set(hVarLabels) & set(hOriginalLabels))}\n\n")
        filename.write(f"Hub Scores of HITS Variation vs Authority Scores of HITS Original\nSpearman: {spearmanr(hVarScores, aOriginalScores)}\nOverlap: {len(set(hVarLabels) & set(aOriginalLabels))}\n\n")

        filename.write(f"Authority Scores of HITS Variation vs Hub Scores of HITS Original\nSpearman: {spearmanr(aVarScores, hOriginalScores)}\nOverlap: {len(set(aVarLabels) & set(hOriginalLabels))}\n\n")
        filename.write(f"Authority Scores of HITS Variation vs Authority Scores of HITS Original\nSpearman: {spearmanr(aVarScores, aOriginalScores)}\nOverlap: {len(set(aVarLabels) & set(aOriginalLabels))}\n\n")

        filename.write(f"Hub Scores of HITS Original vs Authority Scores of HITS Original\nSpearman: {spearmanr(hOriginalScores, aOriginalScores)}\nOverlap: {len(set(hOriginalLabels) & set(aOriginalLabels))}\n\n")

    print("Task III results are saved successfully in file: taskIII.txt inside results folder")

    # Task IV
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'histograms')

    plot_histogram(ranksPrOriginal, 'Scores of Original PageRank with β = 0.8', 'salmon')
    plt.savefig(f"{file_path}/taskIIIpagerankOrig08.png", format = "png")
    plt.clf()

    plot_histogram(ranksPrVariation1, 'Scores of 1st Variation of PageRank with β = 0.8', 'salmon')
    plt.savefig(f"{file_path}/taskIIIpagerankVar1.png", format = "png")
    plt.clf()

    plot_histogram(ranksPrVariation2, 'Scores of 2nd Variation of PageRank with β = 0.8', 'salmon')
    plt.savefig(f"{file_path}/taskIIIpagerankVar2.png", format = "png")
    plt.clf()

    plot_histogram(hOriginalScores, 'Hub Scores of Original HITS', 'salmon')
    plt.savefig(f"{file_path}/taskIIIhubshitsOrig.png", format = "png")
    plt.clf()

    plot_histogram(aOriginalScores, 'Authority Scores of Original HITS', 'salmon')
    plt.savefig(f"{file_path}/taskIIIauthoritieshitsOrig.png", format = "png")
    plt.clf()

    plot_histogram(hVarScores, 'Hub Scores of HITS Variation', 'salmon')
    plt.savefig(f"{file_path}/taskIIIhubshitsVar.png", format = "png")
    plt.clf()

    plot_histogram(aVarScores, 'Authority Scores of HITS Variation', 'salmon')
    plt.savefig(f"{file_path}/taskIIIauthoritieshitsVar.png", format = "png")
    plt.clf()

    print("Histograms for task III are under histograms folder.")


# Task V
def taskV(G):
    # Apply group pagerank and group hits
    resPrOrig = group_pagerank_original(G, 0.8)
    resPrVar1 = group_pagerank_variation1(G, 0.8)
    resPrVar2 = group_pagerank_variation2(G, 0.8)

    hGOrig, aGOrig = group_hits_original(G)
    hGVar, aGVar = group_hits_variation(G)

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'results')

    # Save the results
    with open(f"{file_path}/taskV.txt", "w") as filename:
        filename.write("Task V:\n\n")

        filename.write("Group Pagerank Results:\n\n")

        filename.write("Group Pagerank Original:\n")
        filename.write(f"Liberal group: {resPrOrig["liberal"]} Conservative group: {resPrOrig["conservative"]}\n\n")

        filename.write("Group Pagerank Variation 1:\n")
        filename.write(f"Liberal group: {resPrVar1["liberal"]} Conservative group: {resPrVar1["conservative"]}\n\n")

        filename.write("Group Pagerank Variation 2:\n")
        filename.write(f"Liberal group: {resPrVar2["liberal"]} Conservative group: {resPrVar2["conservative"]}\n\n")

        filename.write("Group HITS Results:\n\n")

        filename.write("Group HITS Original:\n")
        filename.write(f"Hubs Results:\nLiberal: {hGOrig["liberal"]} Conservative: {hGOrig["conservative"]}\n")
        filename.write(f"Authorities Results:\nLiberal: {aGOrig["liberal"]} Conservative: {aGOrig["conservative"]}\n\n")

        filename.write("Group HITS Variation:\n")
        filename.write(f"Hubs Results:\nLiberal: {hGVar["liberal"]} Conservative: {hGVar["conservative"]}\n")
        filename.write(f"Authorities Results:\nLiberal: {aGVar["liberal"]} Conservative: {aGVar["conservative"]}\n\n")

    print("Task V results are saved successfully in file: taskV.txt inside results folder")


def main():
    # Change defaults to be less ugly (for charts)
    mpl.rc('xtick', labelsize=14, color="#222222")
    mpl.rc('ytick', labelsize=14, color="#222222")
    mpl.rc('font', **{'family':'sans-serif','sans-serif':['Arial']})
    mpl.rc('font', size=16)
    mpl.rc('xtick.major', size=6, width=1)
    mpl.rc('xtick.minor', size=3, width=1)
    mpl.rc('ytick.major', size=6, width=1)
    mpl.rc('ytick.minor', size=3, width=1)
    mpl.rc('axes', linewidth=1, edgecolor="#222222", labelcolor="#222222")
    mpl.rc('text', usetex=False, color="#222222")

    # Parse the gml file and convert the graph to a DiGraph object.
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'polblogs.gml')

    G = nx.read_gml(file_path)
    G = nx.DiGraph(G)

    taskI(G)

    taskII(G)
    
    taskIII(G)
    
    taskV(G)


if __name__ == "__main__":
    main()