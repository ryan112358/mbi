"""Utilities for constructing and working with junction trees.

This module provides functions for building junction trees from a given domain
and set of cliques. Junction trees are fundamental structures in graphical model
inference, enabling efficient message passing algorithms. Functions include
finding maximal cliques, determining message passing orders, graph triangulation,
computing greedy elimination orders, and estimating model size.
"""

import itertools
from collections import OrderedDict
from collections.abc import Collection
from typing import TypeAlias

import networkx as nx
import numpy as np

from .domain import Domain

Clique: TypeAlias = tuple[str, ...]


def maximal_cliques(junction_tree: nx.Graph) -> list[Clique]:
    """Return the list of maximal cliques in the model."""
    return list(nx.dfs_preorder_nodes(junction_tree))


def message_passing_order(junction_tree: nx.Graph) -> list[tuple[Clique, Clique]]:
    """Return a valid message passing order."""
    edges = set()
    messages = list(junction_tree.edges()) + [(b, a) for a, b in junction_tree.edges()]
    for m1 in messages:
        for m2 in messages:
            if m1[1] == m2[0] and m1[0] != m2[1]:
                edges.add((m1, m2))
    graph = nx.DiGraph()
    graph.add_nodes_from(messages)
    graph.add_edges_from(edges)
    return list(nx.topological_sort(graph))


def _make_graph(domain: Domain, cliques: Collection[Clique]) -> nx.Graph:
    """Create a graph from the domain and cliques."""
    graph = nx.Graph()
    graph.add_nodes_from(domain.attributes)
    for cl in cliques:
        graph.add_edges_from(itertools.combinations(cl, 2))
    return graph


def _triangulated(graph: nx.Graph, order: list[str]) -> nx.Graph:
    """Triangulate the graph using the given elimination order."""
    edges = set()
    graph2 = nx.Graph(graph)
    for node in order:
        tmp = set(itertools.combinations(graph2.neighbors(node), 2))
        edges |= tmp
        graph2.add_edges_from(tmp)
        graph2.remove_node(node)
    tri = nx.Graph(graph)
    tri.add_edges_from(edges)
    return tri


def greedy_order(
    domain: Domain,
    cliques: list[Clique],
    stochastic: bool = False,
    elim: list[str] | None = None,
) -> tuple[list[str], int]:
    """Compute a greedy elimination order."""
    order = []
    unmarked = elim if elim is not None else list(domain.attributes)
    cliques = set(cliques)
    total_cost = 0
    for _ in range(len(unmarked)):
        cost = OrderedDict()
        for a in unmarked:
            neighbors = [cl for cl in cliques if a in cl]
            variables = tuple(set.union(set(), *[set(n) for n in neighbors]))
            newdom = domain.project(variables)
            cost[a] = newdom.size()

        if stochastic:
            choices = list(unmarked)
            costs = np.array([cost[a] for a in choices], dtype=float)
            probas = np.max(costs) - costs + 1
            probas /= probas.sum()
            i = np.random.choice(probas.size, p=probas)
            a = choices[i]
        else:
            a = min(cost, key=lambda a: cost[a])

        order.append(a)
        unmarked.remove(a)
        neighbors = [cl for cl in cliques if a in cl]
        variables = tuple(set.union(set(), *[set(n) for n in neighbors]) - {a})
        cliques -= set(neighbors)
        cliques.add(variables)
        total_cost += cost[a]

    return order, total_cost


def make_junction_tree(
    domain: Domain,
    cliques: Collection[Clique],
    elimination_order: list[str] | int | None = None,
) -> tuple[nx.Graph, list[str]]:
    """Create a junction tree."""
    cliques = [tuple(cl) for cl in cliques]
    graph = _make_graph(domain, cliques)

    if elimination_order is None and nx.is_tree(graph):
        elimination_order = list(nx.dfs_postorder_nodes(graph))
    elif elimination_order is None:
        elimination_order = greedy_order(domain, cliques, stochastic=False)[0]
    elif isinstance(elimination_order, int):
        orders = [greedy_order(domain, cliques, stochastic=False)] + [
            greedy_order(domain, cliques, stochastic=True)
            for _ in range(elimination_order)
        ]
        elimination_order = min(orders, key=lambda x: x[1])[0]

    tri = _triangulated(graph, elimination_order)
    cliques = sorted([domain.canonical(c) for c in nx.find_cliques(tri)])
    complete = nx.Graph()
    complete.add_nodes_from(cliques)
    for c1, c2 in itertools.combinations(cliques, 2):
        wgt = len(set(c1) & set(c2))
        complete.add_edge(c1, c2, weight=-wgt)
    spanning = nx.minimum_spanning_tree(complete)
    return spanning, elimination_order


def hypothetical_model_size(domain: Domain, cliques: list[Clique]) -> float:
    """Size of the full junction tree parameters, measured in megabytes."""
    jtree, _ = make_junction_tree(domain, cliques)
    max_cliques = maximal_cliques(jtree)
    cells = sum(domain.size(cl) for cl in max_cliques)
    size_mb = cells * 8 / 2**20
    return size_mb
