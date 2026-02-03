import unittest
from mbi.clique_utils import clique_mapping, reverse_clique_mapping
from mbi.domain import Domain

class TestCliqueUtils(unittest.TestCase):
    def test_clique_mapping_no_domain(self):
        # M1: Length 3
        # M2: Length 4
        # Target: ('A', 'B') supported by both.
        # Should pick M1 (fewest elements).

        M1 = ('A', 'B', 'C')
        M2 = ('A', 'B', 'D', 'E')
        maximal_cliques = [M2, M1] # M2 comes first
        all_cliques = [('A', 'B')]

        # clique_mapping
        mapping = clique_mapping(maximal_cliques, all_cliques)
        self.assertEqual(mapping[('A', 'B')], M1, "Should pick M1 (len 3) over M2 (len 4)")

        # reverse_clique_mapping
        rev_mapping = reverse_clique_mapping(maximal_cliques, all_cliques)
        self.assertEqual(rev_mapping[M1], [('A', 'B')], "M1 should contain the clique")
        self.assertEqual(rev_mapping[M2], [], "M2 should not contain the clique")

    def test_clique_mapping_with_domain(self):
        # M1: Length 3, Size 400
        # M2: Length 4, Size 16
        # Target: ('A', 'B')
        # If domain provided, should pick M2 (Size 16 < 400).
        # If domain NOT provided, should pick M1 (Length 3 < 4).

        M1 = ('A', 'B', 'X')
        M2 = ('A', 'B', 'C', 'D')

        attrs = ['A', 'B', 'C', 'D', 'X']
        shape = [2, 2, 2, 2, 100]
        config = dict(zip(attrs, shape))
        domain = Domain.fromdict(config)

        # Check sizes
        self.assertEqual(domain.size(M1), 400)
        self.assertEqual(domain.size(M2), 16)

        maximal_cliques = [M1, M2] # M1 comes first and is shorter.
        all_cliques = [('A', 'B')]

        # Test WITH domain
        try:
            mapping = clique_mapping(maximal_cliques, all_cliques, domain=domain)
            self.assertEqual(mapping[('A', 'B')], M2, "Should pick M2 (size 16) over M1 (size 400)")

            rev_mapping = reverse_clique_mapping(maximal_cliques, all_cliques, domain=domain)
            self.assertEqual(rev_mapping[M2], [('A', 'B')])
            self.assertEqual(rev_mapping[M1], [])
        except TypeError:
            self.fail("clique_mapping or reverse_clique_mapping raised TypeError, likely due to missing domain argument support")

if __name__ == '__main__':
    unittest.main()
