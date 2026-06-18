"""
Tests pour le projet TFHE.
Organisation : une classe par fichier source.

Bugs a corriger avant de lancer :
  1. tore.py    : ajouter  "import numpy as np"  en haut
  2. gadget.py  : remplacer "from torus import Torus" par "from tore import Torus"
"""

import unittest
import numpy as np
import sys
import os

# Ajuste le path si les fichiers ne sont pas dans un package tfhe
sys.path.insert(0, os.path.dirname(__file__))


# ===========================================================================
# 1. TORE
# ===========================================================================

class TestTorus(unittest.TestCase):
    """Verifie que Torus respecte la structure mathematique R/Z."""

    def setUp(self):
        from tore import Torus
        self.T = Torus

    def test_modulo(self):
        """Un element du tore est toujours dans [0, 1[."""
        self.assertAlmostEqual(self.T(1.7).val, 0.7)
        self.assertAlmostEqual(self.T(-0.3).val, 0.7)
        self.assertAlmostEqual(self.T(1.0).val, 0.0)

    def test_addition_boucle(self):
        """Addition modulo 1."""
        a = self.T(0.8)
        b = self.T(0.5)
        self.assertAlmostEqual((a + b).val, 0.3, places=9)

    def test_soustraction(self):
        a = self.T(0.3)
        b = self.T(0.5)
        self.assertAlmostEqual((a - b).val, 0.8, places=9)

    def test_neg(self):
        """-t + t = 0 dans T."""
        t = self.T(0.3)
        self.assertTrue(t + (-t) == self.T(0.0))

    def test_produit_externe_entier(self):
        """Seul Z x T -> T est autorise."""
        t = self.T(0.25)
        self.assertAlmostEqual((2 * t).val, 0.5)
        self.assertAlmostEqual((3 * t).val, 0.75)
        self.assertAlmostEqual((4 * t).val, 0.0)  # boucle

    def test_produit_externe_interdit(self):
        """T x T -> TypeError."""
        t = self.T(0.25)
        with self.assertRaises(TypeError):
            _ = t * t  # pas de __mul__ defini

    def test_addition_interdit_float(self):
        """T + float -> TypeError."""
        t = self.T(0.25)
        with self.assertRaises(TypeError):
            _ = t + 0.5

    def test_egalite(self):
        self.assertTrue(self.T(0.3) == self.T(0.3))
        self.assertFalse(self.T(0.3) == self.T(0.4))

    def test_roundtrip_int32(self):
        """Torus -> int32 -> Torus doit etre presque identique."""
        t = self.T(0.3)
        t2 = self.T.from_int32(t.to_int32())
        self.assertAlmostEqual(t.val, t2.val, places=5)

    def test_from_int32_quart(self):
        """2^30 / 2^32 = 0.25 dans T."""
        t = self.T.from_int32(2**30)
        self.assertAlmostEqual(t.val, 0.25, places=9)


# ===========================================================================
# 2. GADGET DECOMPOSITION
# ===========================================================================

class TestGadgetDecomposition(unittest.TestCase):
    """Verifie le Lemma 3.7 du papier."""

    def setUp(self):
        from gadget import GadgetDecomposition
        from tore import Torus
        self.GD = GadgetDecomposition
        self.T = Torus

    def test_digits_bornes(self):
        """||u||_inf <= Bg/2 (qualite beta)."""
        gd = self.GD(Bg=4, l=3)
        for val in [0.0, 0.1, 0.3, 0.5, 0.7, 0.99]:
            digits = gd.decompose(self.T(val))
            self.assertEqual(len(digits), 3)
            for d in digits:
                self.assertLessEqual(abs(d), gd.beta)

    def test_erreur_reconstruction(self):
        """||u.H - v||_inf <= epsilon."""
        gd = self.GD(Bg=4, l=3)
        for val in [0.0, 0.1, 0.25, 0.5, 0.75, 0.99]:
            t = self.T(val)
            digits = gd.decompose(t)
            t_approx = gd.recompose(digits)
            err = abs(t.val - t_approx.val)
            # gere le cas ou l'erreur boucle pres de 0/1
            err = min(err, 1 - err)
            self.assertLessEqual(err, gd.epsilon + 1e-10)

    def test_bg_puissance_2(self):
        """Bg doit etre une puissance de 2."""
        with self.assertRaises(AssertionError):
            self.GD(Bg=3, l=2)

    def test_decompose_array_shape(self):
        """decompose_array retourne shape (l, N)."""
        gd = self.GD(Bg=4, l=3)
        arr = np.array([0, 2**30, 2**31 - 1], dtype=np.int32)
        result = gd.decompose_array(arr)
        self.assertEqual(result.shape, (3, 3))

    def test_decompose_array_bornes(self):
        """Chaque digit reste dans [-beta, beta]."""
        gd = self.GD(Bg=8, l=4)
        arr = np.random.randint(
            np.iinfo(np.int32).min,
            np.iinfo(np.int32).max,
            size=50,
            dtype=np.int32
        )
        result = gd.decompose_array(arr)
        self.assertTrue(np.all(np.abs(result) <= gd.beta))

    def test_repr(self):
        gd = self.GD(Bg=4, l=3)
        self.assertIn("Bg=4", repr(gd))
        self.assertIn("l=3", repr(gd))


# ===========================================================================
# 3. POLYNOMIAL
# ===========================================================================

class TestPolynomial(unittest.TestCase):
    """Verifie l'arithmetique negacyclique dans Z[X]/(X^N + 1)."""

    def setUp(self):
        from polynomial import (
            Polynomial, polynomial_add, polynomial_subtract,
            polynomial_multiply, polynomial_constant_multiply,
            zero_polynomial, build_monomial
        )
        self.P = Polynomial
        self.add = polynomial_add
        self.sub = polynomial_subtract
        self.mul = polynomial_multiply
        self.cmul = polynomial_constant_multiply
        self.zero = zero_polynomial
        self.mono = build_monomial

    def test_add(self):
        p1 = self.P(N=4, coeff=np.array([1, 2, 3, 4], dtype=np.int32))
        p2 = self.P(N=4, coeff=np.array([1, 0, 0, 0], dtype=np.int32))
        result = self.add(p1, p2)
        np.testing.assert_array_equal(result.coeff, [2, 2, 3, 4])

    def test_sub(self):
        p1 = self.P(N=4, coeff=np.array([3, 2, 1, 0], dtype=np.int32))
        p2 = self.P(N=4, coeff=np.array([1, 1, 1, 0], dtype=np.int32))
        result = self.sub(p1, p2)
        np.testing.assert_array_equal(result.coeff, [2, 1, 0, 0])

    def test_negacyclique_xN_egal_moins1(self):
        """X^N = -1 dans Z[X]/(X^N+1)."""
        N = 4
        # X^3 * X = X^4 = -1 -> coefficients : -1, 0, 0, 0
        x3 = self.mono(1, 3, N)   # X^3
        x1 = self.mono(1, 1, N)   # X
        result = self.mul(x3, x1)
        # X^4 = -X^0 = -1
        np.testing.assert_array_equal(result.coeff, [-1, 0, 0, 0])

    def test_mul_constant(self):
        p = self.P(N=4, coeff=np.array([1, 2, 3, 4], dtype=np.int32))
        result = self.cmul(3, p)
        np.testing.assert_array_equal(result.coeff, [3, 6, 9, 12])

    def test_zero_polynomial(self):
        z = self.zero(8)
        np.testing.assert_array_equal(z.coeff, np.zeros(8))

    def test_build_monomial_signe(self):
        """X^(N+1) = -X dans Z[X]/(X^N+1)."""
        N = 4
        m = self.mono(1, N + 1, N)  # X^5 = X^4 * X = -X
        np.testing.assert_array_equal(m.coeff, [0, -1, 0, 0])

    def test_overflow_int32(self):
        """L'overflow int32 est intentionnel (modulo 2^32 = tore)."""
        p = self.P(N=2, coeff=np.array([2**31 - 1, 0], dtype=np.int32))
        result = self.cmul(2, p)
        # 2 * (2^31 - 1) overflow en int32
        self.assertEqual(result.coeff.dtype, np.int32)


# ===========================================================================
# 4. LWE
# ===========================================================================

class TestLwe(unittest.TestCase):
    """Verifie chiffrement/dechiffrement et operations homomorphes LWE."""

    def setUp(self):
        import lwe, config
        self.lwe = lwe
        self.key = lwe.generate_lwe_key(config.LWE_CONFIG)

    def test_encode_decode_roundtrip(self):
        for i in range(-4, 4):
            pt = self.lwe.lwe_encode(i)
            self.assertEqual(self.lwe.lwe_decode(pt), i)

    def test_encrypt_decrypt(self):
        for i in range(-4, 4):
            pt = self.lwe.lwe_encode(i)
            ct = self.lwe.lwe_encrypt(pt, self.key)
            dec = self.lwe.lwe_decrypt(ct, self.key)
            self.assertEqual(self.lwe.lwe_decode(dec), i)

    def test_encrypt_bool(self):
        for b in [True, False]:
            pt = self.lwe.lwe_encode_bool(b)
            ct = self.lwe.lwe_encrypt(pt, self.key)
            dec = self.lwe.lwe_decrypt(ct, self.key)
            self.assertEqual(self.lwe.lwe_decode_bool(dec), b)

    def test_homomorphic_add(self):
        pt1 = self.lwe.lwe_encode(2)
        pt2 = self.lwe.lwe_encode(1)
        ct1 = self.lwe.lwe_encrypt(pt1, self.key)
        ct2 = self.lwe.lwe_encrypt(pt2, self.key)
        ct_sum = self.lwe.lwe_add(ct1, ct2)
        dec = self.lwe.lwe_decrypt(ct_sum, self.key)
        self.assertEqual(self.lwe.lwe_decode(dec), 3)

    def test_homomorphic_subtract(self):
        pt1 = self.lwe.lwe_encode(3)
        pt2 = self.lwe.lwe_encode(1)
        ct1 = self.lwe.lwe_encrypt(pt1, self.key)
        ct2 = self.lwe.lwe_encrypt(pt2, self.key)
        ct_sub = self.lwe.lwe_subtract(ct1, ct2)
        dec = self.lwe.lwe_decrypt(ct_sub, self.key)
        self.assertEqual(self.lwe.lwe_decode(dec), 2)

    def test_plaintext_multiply(self):
        pt = self.lwe.lwe_encode(2)
        ct = self.lwe.lwe_encrypt(pt, self.key)
        ct2 = self.lwe.lwe_plaintext_multiply(3, ct)
        dec = self.lwe.lwe_decrypt(ct2, self.key)
        self.assertEqual(self.lwe.lwe_decode(dec), 6 % 8 - (6 >= 4) * 8)

    def test_trivial_ciphertext(self):
        import config as cfg
        pt = self.lwe.lwe_encode(3)
        ct = self.lwe.lwe_trivial_ciphertext(pt, cfg.LWE_CONFIG)
        dec = self.lwe.lwe_decrypt(ct, self.key)
        self.assertEqual(self.lwe.lwe_decode(dec), 3)

    def test_bruit_frais_petit(self):
        """Le bruit d'un ciphertext frais doit etre << 2^28."""
        pt = self.lwe.lwe_encode(0)
        errors = []
        for _ in range(100):
            ct = self.lwe.lwe_encrypt(pt, self.key)
            dec = self.lwe.lwe_decrypt(ct, self.key)
            errors.append(abs(int(dec.message)))
        self.assertLess(max(errors), 2**20)


# ===========================================================================
# 5. RLWE
# ===========================================================================

class TestRlwe(unittest.TestCase):

    def setUp(self):
        import rlwe, config
        from polynomial import Polynomial, build_monomial
        self.rlwe = rlwe
        self.key = rlwe.generate_rlwe_key(config.RLWE_CONFIG)
        self.rlwe_key = self.key  # manquait
        self.config = config.RLWE_CONFIG
        self.P = Polynomial
        self.mono = build_monomial

    def test_encrypt_decrypt(self):
        N = self.config.degree
        p = self.P(N=N, coeff=np.array(
            [2, -1, 0, 3] + [0] * (N - 4), dtype=np.int32
        ))
        pt = self.rlwe.rlwe_encode(p, self.config)
        ct = self.rlwe.rlwe_encrypt(pt, self.key)
        dec = self.rlwe.rlwe_decrypt(ct, self.key)
        decoded = self.rlwe.rlwe_decode(dec)
        np.testing.assert_array_equal(decoded.coeff[:4], [2, -1, 0, 3])

    def test_homomorphic_add(self):
        N = self.config.degree
        p1 = self.P(N=N, coeff=np.array([1] + [0]*(N-1), dtype=np.int32))
        p2 = self.P(N=N, coeff=np.array([2] + [0]*(N-1), dtype=np.int32))
        ct1 = self.rlwe.rlwe_encrypt(self.rlwe.rlwe_encode(p1, self.config), self.key)
        ct2 = self.rlwe.rlwe_encrypt(self.rlwe.rlwe_encode(p2, self.config), self.key)
        ct_sum = self.rlwe.rlwe_add(ct1, ct2)
        dec = self.rlwe.rlwe_decode(self.rlwe.rlwe_decrypt(ct_sum, self.key))
        self.assertEqual(dec.coeff[0], 3)

    def test_trivial_ciphertext(self):
        import config as cfg
        from polynomial import Polynomial
        import utils
        N = cfg.RLWE_CONFIG.degree
        coeff = np.zeros(N, dtype=np.int32)
        coeff[0] = utils.encode(2)
        p = Polynomial(N=N, coeff=coeff)
        ct = self.rlwe.rlwe_trivial_ciphertext(p, cfg.RLWE_CONFIG)
        dec = self.rlwe.rlwe_decode(self.rlwe.rlwe_decrypt(ct, self.rlwe_key))
        self.assertEqual(dec.coeff[0], 2)

    def test_degree_mismatch(self):
        from polynomial import Polynomial
        p = Polynomial(N=8, coeff=np.zeros(8, dtype=np.int32))
        with self.assertRaises(ValueError):
            self.rlwe.rlwe_trivial_ciphertext(p, self.config)


# ===========================================================================
# 6. GSW / CMUX
# ===========================================================================

class TestGsw(unittest.TestCase):

    def setUp(self):
        import gsw, rlwe, lwe, config
        from polynomial import build_monomial
        self.gsw = gsw
        self.rlwe = rlwe
        self.lwe_key = lwe.generate_lwe_key(config.LWE_CONFIG)
        self.gsw_key = gsw.convert_lwe_key_to_gsw(self.lwe_key, config.GSW_CONFIG)
        self.rlwe_key = rlwe.convert_lwe_key_to_rlwe(self.lwe_key)
        self.config = config
        self.mono = build_monomial

    def _encrypt_rlwe(self, val, idx=0):
        N = self.config.RLWE_CONFIG.degree
        from polynomial import build_monomial
        p = build_monomial(val, idx, N)
        pt = self.rlwe.rlwe_encode(p, self.config.RLWE_CONFIG)
        return self.rlwe.rlwe_encrypt(pt, self.rlwe_key)

    def test_cmux_select_0(self):
        """CMux(0, l0, l1) = l0."""
        N = self.config.RLWE_CONFIG.degree
        selector = self.mono(0, 0, N)
        sel_pt = self.gsw.GswPlaintext(
            config=self.config.GSW_CONFIG, message=selector
        )
        sel_ct = self.gsw.gsw_encrypt(sel_pt, self.gsw_key)

        ct0 = self._encrypt_rlwe(1)
        ct1 = self._encrypt_rlwe(2)

        result = self.gsw.cmux(sel_ct, ct0, ct1)
        dec = self.rlwe.rlwe_decode(self.rlwe.rlwe_decrypt(result, self.rlwe_key))
        self.assertEqual(dec.coeff[0], 1)

    def test_cmux_select_1(self):
        """CMux(1, l0, l1) = l1."""
        N = self.config.RLWE_CONFIG.degree
        selector = self.mono(1, 0, N)
        sel_pt = self.gsw.GswPlaintext(
            config=self.config.GSW_CONFIG, message=selector
        )
        sel_ct = self.gsw.gsw_encrypt(sel_pt, self.gsw_key)

        ct0 = self._encrypt_rlwe(1)
        ct1 = self._encrypt_rlwe(2)

        result = self.gsw.cmux(sel_ct, ct0, ct1)
        dec = self.rlwe.rlwe_decode(self.rlwe.rlwe_decrypt(result, self.rlwe_key))
        self.assertEqual(dec.coeff[0], 2)


# ===========================================================================
# 7. BOOTSTRAP + NAND
# ===========================================================================

class TestBootstrap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Genere la bootstrap key une seule fois (lent)."""
        import bootstrap, gsw, lwe, config
        cls.lwe = lwe
        cls.bootstrap = bootstrap
        cls.lwe_key = lwe.generate_lwe_key(config.LWE_CONFIG)
        gsw_key = gsw.convert_lwe_key_to_gsw(cls.lwe_key, config.GSW_CONFIG)
        cls.bk = bootstrap.generate_bootstrap_key(cls.lwe_key, gsw_key)

    def test_bootstrap_true(self):
        import utils
        pt = self.lwe.lwe_encode_bool(True)
        for _ in range(3):  # 3 essais independants
            ct = self.lwe.lwe_encrypt(pt, self.lwe_key)
            result = self.bootstrap.bootstrap(ct, self.bk, scale=utils.encode_bool(True))
            dec = self.lwe.lwe_decrypt(result, self.lwe_key)
            if self.lwe.lwe_decode_bool(dec):
                return
        self.fail("bootstrap_true a echoue 3 fois")

    def test_bootstrap_false(self):
        import utils
        pt = self.lwe.lwe_encode_bool(False)
        ct = self.lwe.lwe_encrypt(pt, self.lwe_key)
        result = self.bootstrap.bootstrap(
            ct, self.bk, scale=utils.encode_bool(True)
        )
        dec = self.lwe.lwe_decrypt(result, self.lwe_key)
        self.assertFalse(self.lwe.lwe_decode_bool(dec))

    
class TestNand(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import bootstrap, gsw, lwe, nand, config
        cls.lwe = lwe
        cls.nand = nand
        cls.lwe_key = lwe.generate_lwe_key(config.LWE_CONFIG)
        gsw_key = gsw.convert_lwe_key_to_gsw(cls.lwe_key, config.GSW_CONFIG)
        cls.bk = bootstrap.generate_bootstrap_key(cls.lwe_key, gsw_key)

    def _enc(self, b):
        pt = self.lwe.lwe_encode_bool(b)
        return self.lwe.lwe_encrypt(pt, self.lwe_key)

    def _dec(self, ct):
        return self.lwe.lwe_decode_bool(self.lwe.lwe_decrypt(ct, self.lwe_key))

    def test_nand_table_de_verite(self):
        """NAND(a,b) pour les 4 combinaisons."""
        table = [
            (False, False, True),
            (False, True,  True),
            (True,  False, True),
            (True,  True,  False),
        ]
        for a, b, expected in table:
            result = self.nand.lwe_nand(self._enc(a), self._enc(b), self.bk)
            self.assertEqual(self._dec(result), expected, f"NAND({a},{b}) failed")

    def test_nand_composable(self):
        """Le resultat d'un NAND peut etre utilise comme entree d'un autre NAND."""
        ct_t = self._enc(True)
        ct_f = self._enc(False)
        # NOT(True) = NAND(True, True) = False
        ct_not = self.nand.lwe_nand(ct_t, ct_t, self.bk)
        # NAND(False, False) = True
        result = self.nand.lwe_nand(ct_not, ct_not, self.bk)
        self.assertTrue(self._dec(result))


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    # Lance les tests rapides d'abord, bootstrap en dernier (lent)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestTorus))
    suite.addTests(loader.loadTestsFromTestCase(TestGadgetDecomposition))
    suite.addTests(loader.loadTestsFromTestCase(TestPolynomial))
    suite.addTests(loader.loadTestsFromTestCase(TestLwe))
    suite.addTests(loader.loadTestsFromTestCase(TestRlwe))
    suite.addTests(loader.loadTestsFromTestCase(TestGsw))
    suite.addTests(loader.loadTestsFromTestCase(TestBootstrap))
    suite.addTests(loader.loadTestsFromTestCase(TestNand))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)