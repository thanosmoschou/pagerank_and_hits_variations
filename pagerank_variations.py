"""
Author: Thanos Moschou
Description: This is a programming project for Text Analysis and Search Engines course of UoM.

We use networkx library.
"""

import networkx as nx

# Pagerank first variation
def pagerank_variation_double_weight_for_same_category(G, alpha = 0.85, iterations = 1000, tol = 1e-6):
    if len(G) == 0:
        return {}

    ranks = dict.fromkeys(G, 1 / G.number_of_nodes())
    
    for i in range(iterations):
        prev_ranks = ranks.copy()
        ranks = dict.fromkeys(G, 0)

        for node in G.nodes():
            edges = G.out_edges(node)
            if len(edges) == 0:
                ranks[node] += prev_ranks[node]
            else:
                in_edges = G.in_edges(node)
                for nbr, _ in in_edges:
                    if G.nodes[node]["value"] == G.nodes[nbr]["value"]:
                        ranks[node] += ( alpha * (prev_ranks[nbr] / len(G.out_edges(nbr))) + (1 - alpha) / G.number_of_nodes() )
                    else:
                        ranks[node] += ( alpha * (prev_ranks[nbr] / (2 * len(G.out_edges(nbr)))) + (1 - alpha) / G.number_of_nodes() )
        
        # convergence check
        diff = sum(abs(ranks[node] - prev_ranks[node]) for node in G.nodes())
        if diff < tol:
            break
    
    # Normalization so all ranks add up to 1
    total_rank = sum(ranks.values())
    ranks = {node: rank / total_rank for node, rank in ranks.items()}
    
    return ranks
    

# Pagerank second variation
def pagerank_variation_double_weight_for_reciprocal_edges(G, alpha = 0.85, iterations = 1000, tol = 1e-6):
    if len(G) == 0:
        return {}

    ranks = dict.fromkeys(G, 1 / G.number_of_nodes())
    
    for i in range(iterations):
        prev_ranks = ranks.copy()
        ranks = dict.fromkeys(G, 0)

        for node in G.nodes():
            edges = G.out_edges(node)
            if len(edges) == 0:
                ranks[node] += prev_ranks[node]
            else:
                in_edges = G.in_edges(node)
                for nbr, _ in in_edges:
                    if G.has_edge(node, nbr) and G.has_edge(nbr, node):
                        ranks[node] += ( alpha * (prev_ranks[nbr] / len(G.out_edges(nbr))) + (1 - alpha) / G.number_of_nodes() )
                    else:
                        ranks[node] += ( alpha * (prev_ranks[nbr] / (2 * len(G.out_edges(nbr)))) + (1 - alpha) / G.number_of_nodes() )
        
        # convergence check
        diff = sum(abs(ranks[node] - prev_ranks[node]) for node in G.nodes())
        if diff < tol:
            break
    
    # Normalization so all ranks add up to 1
    total_rank = sum(ranks.values())
    ranks = {node: rank / total_rank for node, rank in ranks.items()}
    
    return ranks
    

# Group Pagerank
def group_pagerank_original(G, alpha = 0.85):
    pagerankRes = nx.pagerank(G, alpha)

    res = {"liberal": 0, "conservative": 0}

    for key, value in pagerankRes.items():
        if G.nodes[key]['value'] == 0:
            res["liberal"] += value
        else:
            res["conservative"] += value

    return res


def group_pagerank_variation1(G, alpha = 0.85):
    pagerankRes = pagerank_variation_double_weight_for_same_category(G, alpha)

    res = {"liberal": 0, "conservative": 0}

    for key, value in pagerankRes.items():
        if G.nodes[key]['value'] == 0:
            res["liberal"] += value
        else:
            res["conservative"] += value

    return res


def group_pagerank_variation2(G, alpha = 0.85):
    pagerankRes = pagerank_variation_double_weight_for_reciprocal_edges(G, alpha)

    res = {"liberal": 0, "conservative": 0}

    for key, value in pagerankRes.items():
        if G.nodes[key]['value'] == 0:
            res["liberal"] += value
        else:
            res["conservative"] += value

    return res
    