import numpy as np

class Torus:
    
    def __init__(self, value): # R/Z
        self.val = float(value) % 1.0
    
    def __add__(self, other): #groupe pour +
        if isinstance(other, Torus):
            return Torus(self.val + other.val)
        raise TypeError("Addition tore + tore seulement")
    
    def __sub__(self, other): 
        if isinstance(other, Torus):
            return Torus(self.val - other.val)
        raise TypeError("Soustraction tore - tore seulement")
    
    def __rmul__(self, scalar): # produit externe pour un Z, pas de produit interne (pas anneau)
        if isinstance(scalar, (int, np.integer)):
            return Torus(scalar * self.val)
        raise TypeError("Seul un entier peut multiplier un élément du tore")
    
    def __repr__(self):
        return f"Torus({self.val:.6f})" #arrondis à 6 chiffres
    
    def __neg__(self):
        return Torus(-self.val)

    def __eq__(self, other):
        return abs(self.val - other.val) < 1e-9
    
    def to_int32(self): # dans le papier, ils bossent sur 32 bits
        return int(self.val * (2**32)) % (2**32)
    
    
    
    @classmethod # permet de créer une instance à partir d'un entier
    def from_int32(cls, n):
        return cls(n / 2**32)
    
    