
import unittest
import time
import signal
import mbi.marginal_oracles
import mbi.domain
import mbi.clique_vector

class TestTimeoutFix(unittest.TestCase):
    def test_message_passing_fast_timeout(self):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        domain = mbi.domain.Domain(list(letters), [2]*len(letters))
        cliques = [()] + [('a', b) for b in letters[1:]]
        theta = mbi.clique_vector.CliqueVector.zeros(domain, cliques)

        def handler(signum, frame):
            raise TimeoutError("Timed out!")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(60) # Should be enough time

        start = time.time()
        try:
            mbi.marginal_oracles.message_passing_fast.lower(theta)
            elapsed = time.time() - start
            print(f"Compilation took {elapsed:.2f}s")
        except TimeoutError:
            self.fail("message_passing_fast.lower timed out")
        finally:
            signal.alarm(0)

if __name__ == '__main__':
    unittest.main()
