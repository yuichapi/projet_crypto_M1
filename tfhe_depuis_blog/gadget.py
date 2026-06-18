# gadget.py
import numpy as np
from tore import Torus


class GadgetDecomposition:
    """
    Lemma 3.7 du papier : decomposition sur le gadget H = (1/Bg, 1/Bg², ..., 1/Bg^l)
    
    Pour v dans T, trouve u dans Z^l tel que :
        u · H ≈ v  avec  ||u||_inf <= Bg/2  (qualite β)
        ||u·H - v||_inf <= 1/2*Bg^l         (precision ε)
    """

    def __init__(self, Bg: int, l: int):
        assert Bg & (Bg - 1) == 0, "Bg doit etre une puissance de 2"
        self.Bg = Bg
        self.l = l
        self.beta = Bg // 2          # borne sur les coefficients
        self.epsilon = 1 / (2 * Bg**l)  # erreur max

    def decompose(self, t: Torus) -> list[int]:
        """
        Algo 1 du papier.
        Retourne [u_1, ..., u_l] avec u_i dans [-Bg/2, Bg/2[
        du plus significatif au moins significatif.
        """
        scale = self.Bg ** self.l
        # arrondir au multiple de 1/Bg^l le plus proche
        approx = round(t.val * scale)

        digits = []
        for _ in range(self.l):
            digit = approx % self.Bg
            approx //= self.Bg
            # recentrer dans [-Bg/2, Bg/2[
            if digit >= self.beta:
                digit -= self.Bg
                approx += 1  # carry
            digits.append(digit)

        return list(reversed(digits))

    def recompose(self, digits: list[int]) -> Torus:
        """Verifie : sum(u_i / Bg^i) ≈ t"""
        val = sum(d / (self.Bg ** (i + 1)) for i, d in enumerate(digits))
        return Torus(val)

    def decompose_array(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Decompose un tableau de TorusInt32 (int32).
        Retourne shape (l, len(coeffs)).
        Utilise par gsw_multiply (produit externe).
        """
        result = np.zeros((self.l, len(coeffs)), dtype=np.int32)
        for j, c in enumerate(coeffs):
            t = Torus.from_int32(int(c))
            for i, d in enumerate(self.decompose(t)):
                result[i, j] = d
        return result

    def __repr__(self):
        return (f"GadgetDecomposition("
                f"Bg={self.Bg}, l={self.l}, "
                f"β={self.beta}, ε={self.epsilon:.2e})")