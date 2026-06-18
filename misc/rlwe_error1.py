import numpy as np
from numpy.polynomial import polynomial as poly
import matplotlib.pyplot as plt

# Paramètres RLWE pour 16 bits
n = 512  # Degré du polynôme
q = 2**20  # Module 
std_dev = 8  # Écart-type pour le bruit
delta = q // 2**16  # Échelle pour l'encodage 16 bits

# Génération de polynômes aléatoires
def random_poly():
    return np.round(np.random.normal(0, std_dev, n)).astype(np.int32) % q

# Réduction modulo x^n + 1
def reduce_modulo(poly_coeffs):
    for i in range(n, len(poly_coeffs)):
        poly_coeffs[i - n] = (poly_coeffs[i - n] - poly_coeffs[i]) % q
    return poly_coeffs[:n]

# Initialisation des clés
s = random_poly()
a = random_poly()
e = random_poly()
b = poly.polymul(a, s)
b = reduce_modulo(b)
b = (b + e) % q

# Chiffrement d'un entier 16 bits
def encrypt(m):
    r = random_poly()
    e1, e2 = random_poly(), random_poly()
    c1 = poly.polymul(a, r)
    c1 = reduce_modulo(c1) % q
    c2 = poly.polymul(b, r)
    c2 = reduce_modulo(c2) % q
    c2 = (c2 + e2 + (m * delta)) % q
    return (c1, c2)

# Déchiffrement
def decrypt(c1, c2):
    s_c1 = poly.polymul(s, c1)
    s_c1 = reduce_modulo(s_c1) % q
    m_approx = (c2 - s_c1) % q
    m_approx_float = m_approx.astype(np.float64) / delta
    m_approx_rounded = np.round(m_approx_float).astype(np.int32) % 2**16
    return int(m_approx_rounded[0])



# Exécution des algorithmes 100 fois
sequential_errors = []
binary_tree_errors = []

for iteration in range(100):
    messages = np.random.randint(0, 2**16, 100)
    scalars = np.random.randint(1, 10, 100)
    encrypted = [encrypt(m) for m in messages]

    real_sum = np.sum(messages * scalars) % 2**16
    seq_sum = sequential_sum(encrypted, scalars)
    tree_sum = binary_tree_sum(encrypted, scalars)

    sequential_errors.append(seq_sum - real_sum)
    binary_tree_errors.append(tree_sum - real_sum)

# Histogrammes comparatifs
plt.figure(figsize=(12, 5))

# Histogramme pour la somme en arbre binaire 
plt.subplot(1, 2, 1)
plt.hist(binary_tree_errors, bins=20, color='purple', alpha=0.7, edgecolor='black')
plt.title("Somme en Arbre Binaire\n(Erreurs)")
plt.xlabel("Erreur absolue")
plt.ylabel("Fréquence")
plt.grid(True, alpha=0.3)

# Histogramme pour la somme séquentielle 
plt.subplot(1, 2, 2)
plt.hist(sequential_errors, bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.title("Somme Séquentielle\n(Erreurs)")
plt.xlabel("Erreur absolue")
plt.ylabel("Fréquence")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
