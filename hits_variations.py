"""
Author: Thanos Moschou
Description: This is a programming project for Text Analysis and Search Engines course of UoM.

We use networkx library.
"""

import networkx as nx

# HITS variation
def hits_variation_in_out_degree(G, iterations = 1000, tol = 1e-6):
    if len(G) == 0:
        return ({}, {})
    
    # Initialize the hubs and authorities scores vectors with 1's
    h = dict.fromkeys(G, 1)
    a = dict.fromkeys(G, 1)

    for i in range(iterations):
        hLast = h.copy() # To check convergence
        # Hub calculation first
        for n in h:
            for _, nbr in G.out_edges(n):
                h[n] += a[nbr] / (G.in_degree(nbr) if G.in_degree(nbr) != 0 else 1) # This authority does not give its whole value to this hub. It divides its value with the number of hubs that show to it.
        
        # Now authorities
        for n in a:
            for nbr, _ in G.in_edges(n):
                a[n] += h[nbr] / (G.out_degree(nbr) if G.out_degree(nbr) != 0 else 1) # This hub does not give its whole score to this authority. It divides its score with the number of authorities that shows to.
        
        # Normalization
        hMax = sum(h.values())
        for n in h: 
            h[n] /= hMax

        aMax = sum(a.values())
        for n in a:
            a[n] /= aMax

        # Convergence check
        if sum(abs(h[n] - hLast[n]) for n in h) < tol:
            break


    return h, a


# Group HITS
def group_hits_original(G):
    if len(G) == 0:
        return ({}, {})
    
    h, a = nx.hits(G)

    retH = {"liberal": 0, "conservative": 0}
    retA = {"liberal": 0, "conservative": 0}

    for key, value in h.items():
        if G.nodes[key]['value'] == 1:
            retH["liberal"] += value
        else:
            retH["conservative"] += value

    for key, value in a.items():
        if G.nodes[key]['value'] == 1:
            retA["liberal"] += value
        else:
            retA["conservative"] += value

    return retH, retA


def group_hits_variation(G):
    if len(G) == 0:
        return ({}, {})
    
    h, a = hits_variation_in_out_degree(G)

    retH = {"liberal": 0, "conservative": 0}
    retA = {"liberal": 0, "conservative": 0}

    for key, value in h.items():
        if G.nodes[key]['value'] == 1:
            retH["liberal"] += value
        else:
            retH["conservative"] += value

    for key, value in a.items():
        if G.nodes[key]['value'] == 1:
            retA["liberal"] += value
        else:
            retA["conservative"] += value

    return retH, retA
