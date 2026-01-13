import unittest
from mbi.domain import Domain
from mbi.junction_tree import make_junction_tree, maximal_cliques

class TestJunctionTree(unittest.TestCase):
    def test_clique_support(self):
        attrs = ['a', 'b', 'c', 'd', 'e']
        shape = [2] * 5
        domain = Domain(attrs, shape)

        # Case 1: Simple chain
        cliques = [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')]
        jtree, _ = make_junction_tree(domain, cliques)
        max_cliques = maximal_cliques(jtree)
        # Check support
        for cl in cliques:
            self.assertTrue(any(set(cl) <= set(mx) for mx in max_cliques), f"Clique {cl} not supported by any maximal clique in chain case")

        # Case 2: Cycle
        cliques = [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')]
        jtree, _ = make_junction_tree(domain, cliques)
        max_cliques = maximal_cliques(jtree)
        for cl in cliques:
            self.assertTrue(any(set(cl) <= set(mx) for mx in max_cliques), f"Clique {cl} not supported by any maximal clique in cycle case")

        # Case 3: More complex
        # A structure where triangulation will add edges.
        cliques = [('a', 'b', 'c'), ('c', 'd', 'e'), ('a', 'e')]
        jtree, _ = make_junction_tree(domain, cliques)
        max_cliques = maximal_cliques(jtree)
        for cl in cliques:
            self.assertTrue(any(set(cl) <= set(mx) for mx in max_cliques), f"Clique {cl} not supported in complex case")

        # Case 4: Disconnected components
        cliques = [('a', 'b'), ('d', 'e')]
        jtree, _ = make_junction_tree(domain, cliques)
        max_cliques = maximal_cliques(jtree)
        for cl in cliques:
            self.assertTrue(any(set(cl) <= set(mx) for mx in max_cliques), f"Clique {cl} not supported in disconnected case")


        # Null graphs
        _ = make_junction_tree(domain, [(a,) for a in attrs])
        _ = make_junction_tree(domain, [])
        _ = make_junction_tree(Domain([], []), [])

if __name__ == '__main__':
    unittest.main()
