import unittest
import jax
import jax.numpy as jnp
from mbi.einsum import scan_einsum, custom_dot_general
import numpy as np

np.random.seed(0)


class TestCustomDotGeneral(unittest.TestCase):

    def test_standard_vector_vector_inner_product(self):
        lhs = np.array([1.0, 2.0, 3.0])
        rhs = np.array([4.0, 5.0, 6.0])
        expected = np.array(32.0)
        dims = (([0], [0]), ([], []))
        res = custom_dot_general(lhs, rhs, dims)
        np.testing.assert_allclose(
            res, expected, err_msg="Std Vec-Vec Inner Product"
        )

    def test_standard_matrix_vector_product(self):
        lhs = np.array([[1.0, 2.0], [3.0, 4.0]])
        rhs = np.array([5.0, 6.0])
        expected = np.array([17.0, 39.0])
        dims = (([1], [0]), ([], []))
        res = custom_dot_general(lhs, rhs, dims)
        np.testing.assert_allclose(res, expected, err_msg="Std Mat-Vec Product")

    def test_standard_matrix_matrix_product(self):
        lhs = np.array([[1.0, 2.0], [3.0, 4.0]])
        rhs = np.array([[5.0, 6.0], [7.0, 8.0]])
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        dims = (([1], [0]), ([], []))
        res = custom_dot_general(lhs, rhs, dims)
        np.testing.assert_allclose(res, expected, err_msg="Std Mat-Mat Product")

    def test_standard_batch_matrix_matrix(self):
        lhs = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        rhs = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
        expected = np.array(
            [[[1.0, 2.0], [3.0, 4.0]], [[11.0, 11.0], [15.0, 15.0]]]
        )
        dims = (([2], [1]), ([0], [0]))
        res = custom_dot_general(lhs, rhs, dims)
        np.testing.assert_allclose(res, expected, err_msg="Std Batch Mat-Mat")

    def test_log_space_vector_vector(self):
        lhs = np.log(np.array([1.0, 2.0]))
        rhs = np.log(np.array([3.0, 4.0]))
        expected = np.log(11.0)
        dims = (([0], [0]), ([], []))
        res = custom_dot_general(
            lhs,
            rhs,
            dims,
            combine_fn=np.add,
            reduce_fn=jax.scipy.special.logsumexp,
        )
        np.testing.assert_allclose(res, expected, err_msg="Log-Space Vec-Vec")

    def test_custom_max_min_product(self):
        lhs = np.array([[1, 6], [5, 2]])
        rhs = np.array([[4, 3], [0, 7]])
        expected = np.array([[4, 3], [2, 5]])
        dims = (([1], [0]), ([], []))
        res = custom_dot_general(
            lhs, rhs, dims, combine_fn=np.maximum, reduce_fn=np.min
        )
        np.testing.assert_equal(res, expected, err_msg="Custom Max-Min Product")

    def test_outer_product_no_contraction(self):
        lhs = np.array([1, 2])
        rhs = np.array([3, 4, 5])
        expected = np.array([[3, 4, 5], [6, 8, 10]])
        dims = (([], []), ([], []))
        res = custom_dot_general(lhs, rhs, dims)
        np.testing.assert_equal(res, expected, err_msg="Outer Product")

    def test_scalar_inputs_default_ops(self):
        lhs_s = np.array(2.0)
        rhs_s = np.array(3.0)
        rhs_v = np.array([3.0, 4.0])

        expected_ss = np.array(6.0)
        dims_ss = (([], []), ([], []))
        res_ss = custom_dot_general(lhs_s, rhs_s, dims_ss)
        np.testing.assert_allclose(
            res_ss, expected_ss, err_msg="Scalar-Scalar Outer"
        )
        assert res_ss.ndim == 0, "Scalar-Scalar output not scalar ndim"

        expected_sv = np.array([6.0, 8.0])
        res_sv = custom_dot_general(lhs_s, rhs_v, dims_ss)
        np.testing.assert_allclose(
            res_sv, expected_sv, err_msg="Scalar-Vector Outer"
        )

        expected_vs = np.array([6.0, 8.0])
        res_vs = custom_dot_general(rhs_v, lhs_s, dims_ss)
        np.testing.assert_allclose(
            res_vs, expected_vs, err_msg="Vector-Scalar Outer"
        )

        lhs_v_contract = np.array([2.0, 3.0])
        rhs_v_contract = np.array([4.0, 5.0])
        expected_contract_s = np.array(23.0)
        dims_contract = (([0], [0]), ([], []))
        res_contract_s = custom_dot_general(
            lhs_v_contract, rhs_v_contract, dims_contract
        )
        np.testing.assert_allclose(
            res_contract_s,
            expected_contract_s,
            err_msg="Scalar output from vector contraction",
        )
        assert (
            res_contract_s.ndim == 0
        ), "Scalar contraction output not scalar ndim"

    def test_empty_contract_batch_dims(self):
        lhs = np.array([[1, 2], [3, 4]])
        rhs = np.array([[5, 6], [7, 8]])
        # For default multiply, this is equivalent to np.multiply.outer(lhs,rhs)
        # The output of outer(a,b) is a.ravel()[:, newaxis] * b.ravel()[newaxis, :], then reshaped.
        # Our function's output shape: (lhs_free_dims, rhs_free_dims) -> (2,2,2,2)
        # The actual outer product `np.outer` is for vectors.
        # A more general tensor product is:
        expected = np.einsum(
            "ab,cd->abcd", lhs, rhs
        )  # This should be the correct comparison

        dims = (([], []), ([], []))
        res = custom_dot_general(lhs, rhs, dims)
        np.testing.assert_equal(
            res, expected, err_msg="Tensor product via empty contract/batch"
        )

    def test_complex_case_multiple_dims(self):
        B, M, N, C1, C2, K_ = 2, 3, 4, 2, 2, 3
        rng = np.random.default_rng(0)
        lhs = rng.random((B, M, C1, C2, K_))
        rhs = rng.random(
            (B, N, C1, C2, K_)
        )  # Note: original JAX test might have different shape for rhs free dim

        dims = (
            ((2, 3, 4), (2, 3, 4)),
            ((0,), (0,)),
        )  # Contract C1,C2,K; Batch B

        # Manual computation: res[b, m, n] = sum_{c1,c2,k} (lhs[b,m,c1,c2,k] * rhs[b,n,c1,c2,k])
        # The free dimension of rhs is N at original index 1.
        # After permutation, lhs_p: (B, M, C1,C2,K), rhs_p: (B, N, C1,C2,K)
        # combine_fn (lhs_expanded(B,M,1,C1,C2,K), rhs_expanded(B,1,N,C1,C2,K))
        # -> combined (B,M,N,C1,C2,K)
        # reduce over last 3 dims (C1,C2,K) -> (B,M,N)

        # This is equivalent to:
        # einsum_path = 'bmcij,bncij->bmn' (if C1,C2,K are c,i,j respectively)
        expected = np.einsum("bmcde,bncde->bmn", lhs, rhs)

        res = custom_dot_general(lhs, rhs, dims)
        np.testing.assert_allclose(
            res, expected, rtol=1e-6, err_msg="Complex multiple dim contraction"
        )


class TestScanEinsum(unittest.TestCase):

    def assertArraysAllClose(self, arr1, arr2, msg=None, rtol=1e-5, atol=1e-5):
        self.assertTrue(jnp.allclose(arr1, arr2, rtol=rtol, atol=atol), msg=msg)

    def test_no_sequential(self):
        A = jnp.arange(6, dtype=jnp.float64).reshape(2, 3)
        B = jnp.arange(12, dtype=jnp.float64).reshape(3, 4)
        # Matrix multiplication
        expected = jnp.einsum("ij,jk->ik", A, B)
        actual = scan_einsum("ij,jk->ik", A, B, sequential="")
        self.assertArraysAllClose(expected, actual)

        # Dot product
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        expected_dot = jnp.einsum("i,i->", x, y)
        actual_dot = scan_einsum("i,i->", x, y, sequential="")
        self.assertArraysAllClose(expected_dot, actual_dot)

        # Outer product
        expected_outer = jnp.einsum("i,j->ij", x, y)
        actual_outer = scan_einsum("i,j->ij", x, y, sequential="")
        self.assertArraysAllClose(expected_outer, actual_outer)

        # Transpose
        expected_T = jnp.einsum("ij->ji", A)
        actual_T = scan_einsum("ij->ji", A, sequential="")
        self.assertArraysAllClose(expected_T, actual_T)

        # More complex
        C = jnp.arange(8, dtype=jnp.float64).reshape(2, 4)
        expected_complex = jnp.einsum(
            "ij,jk,il->kl", A, B, C
        )  # A(2,3) B(3,4) C(2,4) -> (4,4)
        actual_complex = scan_einsum("ij,jk,il->kl", A, B, C, sequential="")
        self.assertArraysAllClose(expected_complex, actual_complex)

    def test_single_sequential_summed_out(self):
        A = jnp.arange(2 * 3 * 4, dtype=jnp.float64).reshape(2, 3, 4)
        B = jnp.arange(3 * 4 * 5, dtype=jnp.float64).reshape(3, 4, 5)

        # Sum over 'j': "ijk,jkl->il"
        # Sequential 'j'
        expected = jnp.einsum("ijk,jkl->il", A, B)
        actual = scan_einsum("ijk,jkl->il", A, B, sequential="j")
        self.assertArraysAllClose(expected, actual, msg="Sequential 'j'")

        # Sequential 'k'
        actual_k = scan_einsum("ijk,jkl->il", A, B, sequential="k")
        self.assertArraysAllClose(expected, actual_k, msg="Sequential 'k'")

        # Test with a dimension of size 1
        X = jnp.arange(2 * 1 * 3, dtype=jnp.float64).reshape(2, 1, 3)
        Y = jnp.arange(1 * 3 * 4, dtype=jnp.float64).reshape(1, 3, 4)
        expected_xy = jnp.einsum("ijk,jkl->il", X, Y)  # j is size 1
        actual_xy_j = scan_einsum("ijk,jkl->il", X, Y, sequential="j")
        self.assertArraysAllClose(
            expected_xy, actual_xy_j, msg="Sequential 'j' with size 1"
        )
        actual_xy_k = scan_einsum(
            "ijk,jkl->il", X, Y, sequential="k"
        )  # k is size 3
        self.assertArraysAllClose(
            expected_xy, actual_xy_k, msg="Sequential 'k' with size 3"
        )

    def test_single_sequential_in_output(self):
        A = jnp.arange(2 * 3, dtype=jnp.float64).reshape(2, 3)
        B = jnp.arange(2 * 3, dtype=jnp.float64).reshape(2, 3)

        # Element-wise product like: "ij,ij->ij"
        # Sequential 'i'
        expected = jnp.einsum("ij,ij->ij", A, B)
        actual_i = scan_einsum("ij,ij->ij", A, B, sequential="i")
        self.assertArraysAllClose(expected, actual_i, msg="Sequential 'i'")

        # Sequential 'j'
        actual_j = scan_einsum("ij,ij->ij", A, B, sequential="j")
        self.assertArraysAllClose(expected, actual_j, msg="Sequential 'j'")

        # Broadcasting like: "i,j->ij" (sequential 'i')
        x = jnp.array([1.0, 2.0])
        y = jnp.array([10.0, 20.0, 30.0])
        expected_broadcast = jnp.einsum("i,j->ij", x, y)
        actual_broadcast_i = scan_einsum("i,j->ij", x, y, sequential="i")
        self.assertArraysAllClose(
            expected_broadcast, actual_broadcast_i, msg="Broadcast seq 'i'"
        )
        actual_broadcast_j = scan_einsum("i,j->ij", x, y, sequential="j")
        self.assertArraysAllClose(
            expected_broadcast, actual_broadcast_j, msg="Broadcast seq 'j'"
        )

        # Batch matrix mul: "bij,bjk->bik"
        # Sequential 'b' (batch dimension)
        P = jnp.arange(2 * 3 * 4, dtype=jnp.float64).reshape(
            2, 3, 4
        )  # b=2, i=3, j=4
        Q = jnp.arange(2 * 4 * 5, dtype=jnp.float64).reshape(
            2, 4, 5
        )  # b=2, j=4, k=5
        expected_bmm = jnp.einsum("bij,bjk->bik", P, Q)
        actual_bmm_b = scan_einsum("bij,bjk->bik", P, Q, sequential="b")
        self.assertArraysAllClose(expected_bmm, actual_bmm_b, msg="BMM seq 'b'")

        # Sequential 'j' (summed out, but 'b' is in output) - covered by other test
        # actual_bmm_j = scan_einsum("bij,bjk->bik", P, Q, sequential="j")
        # self.assertArraysAllClose(expected_bmm, actual_bmm_j, msg="BMM seq 'j'")

    def test_multiple_sequential_axes(self):
        A = jnp.arange(2 * 3 * 4, dtype=jnp.float64).reshape(2, 3, 4)  # ijk
        B = jnp.arange(3 * 4 * 5, dtype=jnp.float64).reshape(3, 4, 5)  # jkl
        C = jnp.arange(2 * 5 * 6, dtype=jnp.float64).reshape(2, 5, 6)  # ilm

        # Formula: "ijk,jkl,ilm->km"
        # i,j,l are summed out. k,m are output.
        # Shapes: A(i=2,j=3,k=4), B(j=3,k=4,l=5), C(i=2,l=5,m=6) -> Output(k=4,m=6)
        expected = jnp.einsum("ijk,jkl,ilm->km", A, B, C)

        # Seq "ij" (both summed out)
        actual_ij = scan_einsum("ijk,jkl,ilm->km", A, B, C, sequential="ij")
        self.assertArraysAllClose(expected, actual_ij, msg="Sequential 'ij'")

        # Seq "ji" (both summed out, order matters for scan)
        actual_ji = scan_einsum("ijk,jkl,ilm->km", A, B, C, sequential="ji")
        self.assertArraysAllClose(expected, actual_ji, msg="Sequential 'ji'")

        # Seq "il" (both summed out)
        actual_il = scan_einsum("ijk,jkl,ilm->km", A, B, C, sequential="il")
        self.assertArraysAllClose(expected, actual_il, msg="Sequential 'il'")

        # Seq "jl" (both summed out)
        actual_jl = scan_einsum("ijk,jkl,ilm->km", A, B, C, sequential="jl")
        self.assertArraysAllClose(expected, actual_jl, msg="Sequential 'jl'")

        # Formula: "ijk,j->ik" (B is vector)
        X = jnp.arange(2 * 3 * 4, dtype=jnp.float64).reshape(2, 3, 4)  # ijk
        Y = jnp.arange(3, dtype=jnp.float64)  # j
        expected_xy = jnp.einsum("ijk,j->ik", X, Y)

        # Seq "j" (summed out) - i,k in output
        actual_xy_j = scan_einsum("ijk,j->ik", X, Y, sequential="j")
        self.assertArraysAllClose(
            expected_xy, actual_xy_j, msg="Sequential 'j' for ijk,j->ik"
        )

        # Formula: "aij,bjk,ckl->abil" (a,b,c are batch dims, i,j,k are summed)
        # Let a=2, b=2, c=2, i=3,j=4,k=5,l=3
        T1 = jnp.array(np.random.rand(2, 3, 4), dtype=jnp.float64)  # aij
        T2 = jnp.array(np.random.rand(2, 4, 5), dtype=jnp.float64)  # bjk
        T3 = jnp.array(np.random.rand(2, 5, 3), dtype=jnp.float64)  # ckl
        expected_abc = jnp.einsum("aij,bjk,ckl->abil", T1, T2, T3)

        # Seq "jk" (summed out, a,b,c,l in output)
        actual_abc_jk = scan_einsum(
            "aij,bjk,ckl->abil", T1, T2, T3, sequential="jk"
        )
        self.assertArraysAllClose(
            expected_abc,
            actual_abc_jk,
            msg="Sequential 'jk' for aij,bjk,ckl->abil",
        )

        # Seq "ab" (in output)
        actual_abc_ab = scan_einsum(
            "aij,bjk,ckl->abil", T1, T2, T3, sequential="ab"
        )
        self.assertArraysAllClose(
            expected_abc,
            actual_abc_ab,
            msg="Sequential 'ab' for aij,bjk,ckl->abil",
        )

        # Seq "aj" (a in output, j summed out)
        actual_abc_aj = scan_einsum(
            "aij,bjk,ckl->abil", T1, T2, T3, sequential="aj"
        )
        self.assertArraysAllClose(
            expected_abc,
            actual_abc_aj,
            msg="Sequential 'aj' for aij,bjk,ckl->abil",
        )

        # Test all sequential axes "ijk"
        D1 = jnp.arange(2 * 2 * 2, dtype=jnp.float64).reshape(2, 2, 2)  # ijk
        D2 = jnp.arange(2 * 2 * 2, dtype=jnp.float64).reshape(2, 2, 2)  # jkl
        D3 = jnp.arange(2 * 2 * 2, dtype=jnp.float64).reshape(2, 2, 2)  # kli
        # ijk,jkl,kli -> (no output axes, scalar sum)
        expected_all_seq = jnp.einsum("ijk,jkl,kli->", D1, D2, D3)
        actual_all_seq_ijk = scan_einsum(
            "ijk,jkl,kli->", D1, D2, D3, sequential="ijk"
        )
        self.assertArraysAllClose(
            expected_all_seq, actual_all_seq_ijk, msg="Seq 'ijk' all summed"
        )
        actual_all_seq_ikj = scan_einsum(
            "ijk,jkl,kli->", D1, D2, D3, sequential="ikj"
        )  # order change
        self.assertArraysAllClose(
            expected_all_seq, actual_all_seq_ikj, msg="Seq 'ikj' all summed"
        )

    def test_readme_example(self):
        # From the original problem description/context if any or a typical use case
        # Example: trace of batch matrix products
        # (B, N, M) @ (B, M, K) -> (B, N, K), then trace over N, K if N=K.
        # Or contract more: A(b,i,j), C(b,j,k), D(b,k,l) -> E(b,i,l)
        A = jnp.array(np.random.rand(2, 3, 4), dtype=jnp.float64)  # bij
        B = jnp.array(np.random.rand(2, 4, 5), dtype=jnp.float64)  # bjk
        C = jnp.array(np.random.rand(2, 5, 6), dtype=jnp.float64)  # bkl
        # Result E(b,i,l) -> (2,3,6)
        expected = jnp.einsum("bij,bjk,bkl->bil", A, B, C)
        # Sequential over b
        actual_b = scan_einsum("bij,bjk,bkl->bil", A, B, C, sequential="b")
        self.assertArraysAllClose(
            expected, actual_b, msg="Chain product seq 'b'"
        )
        # Sequential over j
        actual_j = scan_einsum("bij,bjk,bkl->bil", A, B, C, sequential="j")
        self.assertArraysAllClose(
            expected, actual_j, msg="Chain product seq 'j'"
        )
        # Sequential over k
        actual_k = scan_einsum("bij,bjk,bkl->bil", A, B, C, sequential="k")
        self.assertArraysAllClose(
            expected, actual_k, msg="Chain product seq 'k'"
        )
        # Sequential over "jk"
        actual_jk = scan_einsum("bij,bjk,bkl->bil", A, B, C, sequential="jk")
        self.assertArraysAllClose(
            expected, actual_jk, msg="Chain product seq 'jk'"
        )
        # Sequential over "bj"
        actual_bj = scan_einsum("bij,bjk,bkl->bil", A, B, C, sequential="bj")
        self.assertArraysAllClose(
            expected, actual_bj, msg="Chain product seq 'bj'"
        )

    def test_output_only_formula(self):

        # Case: "a->a" where 'a' is sequential
        X = jnp.arange(5, dtype=jnp.float64)
        expected = jnp.einsum("a->a", X)
        actual = scan_einsum("a->a", X, sequential="a")
        self.assertArraysAllClose(expected, actual, msg="a->a sequential a")

        # Case: ",a->a" with one array (implicit ellipsis for first array)
        # jnp.einsum interprets ",a->a" as "...,a->a"
        # Our current _infer_shapes expects explicit labels for all arrays
        # To support this, _infer_shapes or the parsing logic would need adjustment.
        # For now, assume explicit labels.
        # If this means "a,b->b" with arrays X, Y and seq='a'
        # For "a->", seq 'a'
        expected_sum = jnp.einsum("a->", X)
        actual_sum = scan_einsum("a->", X, sequential="a")
        self.assertArraysAllClose(
            expected_sum, actual_sum, msg="a-> sequential a"
        )

    def test_sequential_axis_not_in_all_arrays(self):
        A = jnp.arange(2 * 3, dtype=jnp.float64).reshape(2, 3)  # ij
        B = jnp.arange(3 * 4, dtype=jnp.float64).reshape(3, 4)  # jk
        C = jnp.arange(2 * 4, dtype=jnp.float64).reshape(
            2, 4
        )  # ik (output like)

        # Formula: "ij,jk->ik"
        expected = jnp.einsum("ij,jk->ik", A, B)
        # Seq "i": A is sliced, B is passed as whole.
        actual_i = scan_einsum("ij,jk->ik", A, B, sequential="i")
        self.assertArraysAllClose(
            expected, actual_i, msg="Seq 'i' for ij,jk->ik"
        )
        # Seq "j": A and B are sliced.
        actual_j = scan_einsum("ij,jk->ik", A, B, sequential="j")
        self.assertArraysAllClose(
            expected, actual_j, msg="Seq 'j' for ij,jk->ik"
        )
        # Seq "k": B is sliced, A is passed as whole.
        actual_k = scan_einsum("ij,jk->ik", A, B, sequential="k")
        self.assertArraysAllClose(
            expected, actual_k, msg="Seq 'k' for ij,jk->ik"
        )

        # More complex: "ab,bc,cd->ad"
        X = jnp.arange(2 * 3, dtype=jnp.float64).reshape(2, 3)  # ab
        Y = jnp.arange(3 * 4, dtype=jnp.float64).reshape(3, 4)  # bc
        Z = jnp.arange(4 * 5, dtype=jnp.float64).reshape(4, 5)  # cd
        expected_xyz = jnp.einsum("ab,bc,cd->ad", X, Y, Z)

        # Seq "a": X sliced, Y,Z whole
        actual_a = scan_einsum("ab,bc,cd->ad", X, Y, Z, sequential="a")
        self.assertArraysAllClose(
            expected_xyz, actual_a, msg="Seq 'a' for ab,bc,cd->ad"
        )
        # Seq "b": X,Y sliced, Z whole
        actual_b = scan_einsum("ab,bc,cd->ad", X, Y, Z, sequential="b")
        self.assertArraysAllClose(
            expected_xyz, actual_b, msg="Seq 'b' for ab,bc,cd->ad"
        )
        # Seq "c": Y,Z sliced, X whole
        actual_c = scan_einsum("ab,bc,cd->ad", X, Y, Z, sequential="c")
        self.assertArraysAllClose(
            expected_xyz, actual_c, msg="Seq 'c' for ab,bc,cd->ad"
        )
        # Seq "d": Z sliced, X,Y whole
        actual_d = scan_einsum("ab,bc,cd->ad", X, Y, Z, sequential="d")
        self.assertArraysAllClose(
            expected_xyz, actual_d, msg="Seq 'd' for ab,bc,cd->ad"
        )
        # Seq "ac": a in X, c in Y,Z
        actual_ac = scan_einsum("ab,bc,cd->ad", X, Y, Z, sequential="ac")
        self.assertArraysAllClose(
            expected_xyz, actual_ac, msg="Seq 'ac' for ab,bc,cd->ad"
        )

    def test_custom_einsum_max(self):
        from mbi.einsum import custom_einsum

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        # 'ij,jk->ik' with (+, max)
        actual = custom_einsum(
            "ij,jk->ik", x, y, combine_fn=jnp.add, reduce_fn=jnp.max
        )
        expected = jnp.array([[9.0, 10.0], [11.0, 12.0]])
        self.assertArraysAllClose(
            expected, actual, msg="custom_einsum with jnp.max"
        )

    def test_custom_einsum_logsumexp(self):
        from mbi.einsum import custom_einsum
        from jax.scipy.special import logsumexp

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        actual = custom_einsum(
            "ij,jk->ik", x, y, combine_fn=jnp.add, reduce_fn=logsumexp
        )
        expected = jnp.array([
            [
                logsumexp(jnp.array([6.0, 9.0])),
                logsumexp(jnp.array([7.0, 10.0])),
            ],
            [
                logsumexp(jnp.array([8.0, 11.0])),
                logsumexp(jnp.array([9.0, 12.0])),
            ],
        ])
        self.assertArraysAllClose(
            expected, actual, msg="custom_einsum with logsumexp"
        )

    def test_custom_einsum_vmap_jit(self):
        from mbi.einsum import custom_einsum

        def f(x, y):
            return custom_einsum(
                "ij,jk->ik", x, y, combine_fn=jnp.add, reduce_fn=jnp.max
            )

        f = jax.jit(jax.vmap(f, in_axes=(0, 0)))
        x = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
        y = jnp.array([[[5.0, 6.0], [7.0, 8.0]]])
        actual = f(x, y)
        expected = jnp.array([[[9.0, 10.0], [11.0, 12.0]]])
        self.assertArraysAllClose(
            expected, actual, msg="custom_einsum vmap and jit"
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
