"""Utility functions for manipulating cliques (subsets of attributes).

This module provides helper functions for common operations on cliques,
such as finding maximal subsets and creating mappings between cliques and
their maximal counterparts. Cliques are typically represented as tuples of
attribute names (strings).
"""
from typing import TypeAlias

Clique: TypeAlias = tuple[str, ...]


def reverse_clique_mapping(
    maximal_cliques: list[Clique], all_cliques: list[Clique]
) -> dict[Clique, list[Clique]]:
    """Creates a mapping from maximal cliques to a list of cliques they contain.

    Args:
      maximal_cliques: A list of maximal cliques.
      all_cliques: A list of all cliques.

    Returns:
      A mapping from maximal cliques to cliques they contain.
    """
    mapping = {cl: [] for cl in maximal_cliques}
    for cl in all_cliques:
        for cl2 in maximal_cliques:
            if set(cl) <= set(cl2):
                mapping[cl2].append(cl)
                break
    return mapping


def maximal_subset(cliques: list[Clique]) -> list[Clique]:
    """Given a list of cliques, finds a maximal subset of non-nested cliques.

    A clique is considered nested in another if all its vertices are a subset
    of the other's vertices.

    Example Usage:
    >>> maximal_subset([('A', 'B'), ('B',), ('C',), ('B', 'A')])
    [('A', 'B'), ('C',)]

    Args:
      cliques: A list of cliques.

    Returns:
      A new list containing a maximal subset of non-nested cliques.
    """
    cliques = sorted(cliques, key=len, reverse=True)
    result = []
    for cl in cliques:
        if not any(set(cl) <= set(cl2) for cl2 in result):
            result.append(cl)
    return result


def clique_mapping(
    maximal_cliques: list[Clique], all_cliques: list[Clique]
) -> dict[Clique, Clique]:
    """Creates a mapping from cliques to their corresponding maximal clique.

    Example Usage:
    >>> maximal_cliques = [('A', 'B'), ('B', 'C')]
    >>> all_cliques = [('B', 'A'), ('B',), ('C',), ('B', 'C')]
    >>> mapping = clique_mapping(maximal_cliques, all_cliques)
    >>> print(mapping)
    {('B', 'A'): ('A', 'B'), ('B',): ('A', 'B'), ('C',): ('B', 'C'), ('B', 'C'): ('B', 'C')}

    Args:
      maximal_cliques: A list of maximal cliques.
      all_cliques: A list of all cliques.

    Returns:
      A mapping from cliques to their maximal clique.

    """
    mapping = {}
    for cl in all_cliques:
        for cl2 in maximal_cliques:
            if set(cl) <= set(cl2):
                mapping[cl] = cl2
                break
    return mapping
