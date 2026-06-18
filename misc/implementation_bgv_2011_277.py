import numpy as np

# ═══════════════════════════════════════════════════════════════
#  Arithmétique dans R_q = Z[x]/(x^d + 1)
# ═══════════════════════════════════════════════════════════════

def poly_mod(poly, d, q):
    """Réduit un polynôme mod (x^d + 1) et mod q."""
    result = np.zeros(d, dtype=object)

    for i, coef in enumerate(poly):
        deg = i % (2 * d)
        if deg < d:
            result[deg] = (result[deg] + coef) % q
        else:
            result[deg - d] = (result[deg - d] - coef) % q
    return result % q

def poly_mul(a, b, d, q):
    return poly_mod(np.convolve(a, b), d, q)

def poly_add(a, b, q):
    n = max(len(a), len(b))
    a = np.pad(a, (0, n - len(a)))
    b = np.pad(b, (0, n - len(b)))
    return (a + b) % q

def poly_neg(a, q):
    return (-a) % q

# ═══════════════════════════════════════════════════════════════
#  Vecteurs / matrices de polynômes
# ═══════════════════════════════════════════════════════════════

def vec_dot(u, v, d, q):
    result = np.zeros(d, dtype=object)
    for a, b in zip(u, v):
        result = poly_add(result, poly_mul(a, b, d, q), q)
    return result

def mat_transpose_vec(M, v, d, q): # M^T·v, avec M matrice de polynômes et v vecteur de polynômes
    n_cols = len(M[0])
    return [vec_dot([M[i][j] for i in range(len(M))], v, d, q)
            for j in range(n_cols)]

# ═══════════════════════════════════════════════════════════════
#  Distributions
# ═══════════════════════════════════════════════════════════════

def sample_uniform_poly(d, q):
    return np.array([np.random.randint(0, q) for _ in range(d)], dtype=object)

def sample_binary_poly(d):
    return np.array([np.random.randint(0, 2) for _ in range(d)], dtype=object)

def sample_error_poly(d, std=1.0):
    return np.array(np.round(np.random.normal(0, std, d)).astype(int), dtype=object)

# ═══════════════════════════════════════════════════════════════
#  GLWE Setup / KeyGen / Enc / Dec
#
# ═══════════════════════════════════════════════════════════════

def setup(lam=40, mu=16, b=1):
    """
    b=0 → LWE  (d=1, n=lam//4)
    b=1 → RLWE (d=64, n=1)
    q = 2^mu — puissance de 2 exacte, indispensable pour la parité.
    """
    q = 2 ** mu
    if b == 0:
        d = 1
        n = lam // 4
    else:
        d = 64
        n = 1
    N   = max(d * (2 * n + 1), 8)
    # N c la taille de la matrice A, c le param de sécu
    std = 1.0
    return {"q": q, "d": d, "n": n, "N": N, "std": std}

def secret_key_gen(params):
    """s = (1, s'₀, …, s'_{n-1}),  s' ← χⁿ."""
    q, d, n, std = params["q"], params["d"], params["n"], params["std"]
    one = np.zeros(d, dtype=object); one[0] = 1
    return [one] + [sample_error_poly(d, std) for _ in range(n)]
# de ce que je comprend le 1 c'est juste pour simplifier le déchiffrement'
# <c,s> = c0*1+c1*s' = c0 + c1*s' 
# d'où c0 = m+e + b*r, c1 = -a*r et c0 + c1·s = m + e

def public_key_gen(params, sk):
    """pk = A = [b | −A'] ∈ R^{N×(n+1)},  vérifie A·s = 2e."""
    q, d, n, N, std = (params["q"], params["d"], params["n"],
                       params["N"], params["std"])
    sp      = sk[1:]
    # on enlève le 1 fixe de la secret key gen
    A_prime = [[sample_uniform_poly(d, q) for _ in range(n)] for _ in range(N)]
    e       = [sample_error_poly(d, std) for _ in range(N)]
    b       = [poly_add(vec_dot(A_prime[i], sp, d, q), (2 * e[i]) % q, q)
               for i in range(N)]
    return [[b[i]] + [poly_neg(A_prime[i][j], q) for j in range(n)]
            for i in range(N)]

def encrypt(params, pk, m_poly):
    """c = m̃ + Aᵀr,  m̃ = (m, 0,…,0),  r ← R_2^N."""
    q, d, n, N = params["q"], params["d"], params["n"], params["N"]
    m_tilde = [m_poly % q] + [np.zeros(d, dtype=object) for _ in range(n)]
    r       = [sample_binary_poly(d) for _ in range(N)]
    At_r    = mat_transpose_vec(pk, r, d, q)
    return [poly_add(m_tilde[j], At_r[j], q) for j in range(n + 1)]

def decrypt(params, sk, c):
    """m = [[⟨c, s⟩]_q]_2."""
    q, d  = params["q"], params["d"]
    inner = vec_dot(c, sk, d, q)
    ic    = np.array([int(x) for x in inner], dtype=int)
    ic    = ((ic + q // 2) % q) - q // 2
    return ic % 2

# ═══════════════════════════════════════════════════════════════
#  Primitives FHE : bit_length, BitDecomp, Powersof2, Scale
# ═══════════════════════════════════════════════════════════════

def bit_length(q):
    return int(np.floor(np.log2(int(q)))) + 1


# retourne l polynomes binaires 
def powersof2_poly(p, q, d):
    """[p, 2p, 4p, …, 2^{l−1}p] mod q — liste de l polynômes."""
    l = bit_length(q)
    return [np.array([(int(c) * (1 << i)) % q for c in p], dtype=object)
            for i in range(l)]

def powersof2_vec(vec, q, d):
    """Powersof2 sur un vecteur : k polys → k·l polys."""
    result = []
    for poly in vec:
        result.extend(powersof2_poly(poly, q, d))
    return result

def bitdecomp_poly(p, q, d):
    """Décomposition bit à bit : 1 poly → l polys binaires."""
    l = bit_length(q)
    # NDLR : >> ça décale d'un bit donc ça divise par 2
    return [np.array([(int(c) >> i) & 1 for c in p], dtype=object)
            for i in range(l)]

def bitdecomp_vec(vec, q, d):
    result = []
    for poly in vec:
        result.extend(bitdecomp_poly(poly, q, d))
    return result

# CAR (cf. le papier !) ⟨bitdecomp(c), powersof2(s)⟩ = ⟨c, s⟩   
# or bitdecomp a des coef petits dans 0,1

def scale_poly(p, q_from, q_to, t, d):
    """
    Scale(p, q_from → q_to) avec correction de parité mod t.
    Choisit floor ou ceil pour que scaled % t == p % t.
    Requiert q_from = 2^k pour garantir la correction.
    """
    result = np.zeros(d, dtype=object)
    for i in range(d):
        c  = int(p[i])
        fv = (c * q_to) // q_from # floor 
        cv = fv + 1 #ceil
        result[i] = (fv if (fv % t) == (c % t) else cv) % q_to
        # on traduit un coef de q_from vers q_to en le scalant. 
        # le problème c'est que ( :) ) ils faut qu'ils restent congruent mod t
        # (cf le papier : q congru à p congru à 1 mod r)
    return result

def scale_vec(vec, q_from, q_to, t, d):
    return [scale_poly(p, q_from, q_to, t, d) for p in vec]

# ═══════════════════════════════════════════════════════════════
#  Tensoring & embed
# ═══════════════════════════════════════════════════════════════

def tensor_vec(s, d, q):
    """s' = s ⊗ s ∈ R^{(n+1)²}_q."""
    n1 = len(s)
    return [poly_mul(s[i], s[j], d, q)
            for i in range(n1) for j in range(n1)]

def ciphertext_tensor(c1, c2, d, q):
    """
    c3 tel que ⟨c3, s⊗s⟩ = ⟨c1,s⟩ · ⟨c2,s⟩.
    Le déchiffrement donne m1·m2 dans R_2 = Z_2[x]/(x^d+1)
    (produit polynomial, pas AND bit à bit).
    """
    n1 = len(c1)
    return [poly_mul(c1[i], c2[j], d, q)
            for i in range(n1) for j in range(n1)]

def embed_in_tensor_space(c, d):
    """
    Plonge c ∈ R^{n+1} dans R^{(n+1)²} tel que ⟨c_tensor, s⊗s⟩ = ⟨c, s⟩.
    Identité : s[0]=1 ⟹ (s⊗s)[i·(n+1)] = s[i]·s[0] = s[i].
    Pose c_tensor[i·(n+1)] = c[i], reste = 0.
    """
    n1     = len(c)
    zero   = np.zeros(d, dtype=object)
    result = [zero.copy() for _ in range(n1 * n1)]
    for i in range(n1):
        result[i * n1] = c[i].copy()
    return result

# ═══════════════════════════════════════════════════════════════
#  Key Switching
#
#  CORRECTION 2 : switch_key_gen génère un sample RLWE direct,
#  pas un appel à encrypt(). Cela évite le doublement du bruit.
#  Pour chaque b ∈ s'' :
#    c0 = a·sk_to[1] + 2e + b  mod q_{j-1}
#    c1 = −a                   mod q_{j-1}
#  ⟹ ⟨(c0,c1), s_{j-1}⟩ = b + 2e  ✓
# ═══════════════════════════════════════════════════════════════

def switch_key_gen(s_pp, sk_to, params_to):
    #s_pp c'est bitdecomp(s⊗s)
    """τ_{s''_j → s_{j-1}} : un chiffré de b sous s_{j-1} pour chaque b ∈ s''."""
    q, d, std = params_to["q"], params_to["d"], params_to["std"]
    tau = []
    for b_poly in s_pp:
        a  = sample_uniform_poly(d, q)
        e  = sample_error_poly(d, std)
        c0 = poly_add(
                 poly_add(poly_mul(a, sk_to[1], d, q), (2 * e) % q, q),
                 b_poly % q, q)
        c1 = poly_neg(a, q)
        tau.append([c0, c1])
    return tau

def switch_key(tau, c2, q, d):
    """
    SwitchKey(τ, c2) = Σ_i c2[i] · τ[i]  mod q.
    ⟨résultat, s_{j-1}⟩ ≈ ⟨c2, s''_j⟩  (à bruit 2e près).
    """
    n_out  = len(tau[0])
    result = [np.zeros(d, dtype=object) for _ in range(n_out)]
    for i, ct_i in enumerate(tau):
        coef = c2[i]
        for k in range(n_out):
            result[k] = poly_add(result[k], poly_mul(coef, ct_i[k], d, q), q)
    return result

# ═══════════════════════════════════════════════════════════════
#  FHE Refresh
# ═══════════════════════════════════════════════════════════════

def fhe_refresh(c, tau, params_j, params_jm1, d):
    """
    Refresh(c, τ, q_j, q_{j-1}) — réduit le modulus et change de clé.

    c sous s_j   (longueur n+1)   → embed dans (n+1)²  [cas Add]
    c sous s'_j  (longueur (n+1)²) → direct             [cas Mult]

    1. Embed si nécessaire
    2. Expand    c1 = Powersof2(c_tensor, q_j)
    3. Scale     c2 = Scale(c1, q_j → q_{j-1}, mod 2)
    4. SwitchKey c3 = SwitchKey(τ, c2, q_{j-1})
    """
    q_j   = params_j["q"]
    q_jm1 = params_jm1["q"]
    n1    = params_j["n"] + 1
    l_j   = bit_length(q_j)

    if len(c) == n1: # n+1 donc c un add
        c_tensor = embed_in_tensor_space(c, d)
    elif len(c) == n1 * n1: # (n+1)² donc c un mult
        c_tensor = c
    else:
        raise ValueError(f"Longueur de c inattendue : {len(c)}, attendu {n1} ou {n1*n1}")

    assert len(c_tensor) * l_j == len(tau), (
        f"Désalignement Powersof2/tau : {len(c_tensor)*l_j} vs {len(tau)}")

    # on connait la taille de c : (n+1)^2
    """
     là c'est totalement banger 
     les n+1)^2 polynomes de c_tensor sont développés en l_j polynomess
     puis tous ces tout petit coef de gros BÉBÉ sont scaled selon la nouvelle clef
     puis BAM on change la clef. 
    """
    c1 = powersof2_vec(c_tensor, q_j, d)
    c2 = scale_vec(c1, q_j, q_jm1, 2, d)
    return switch_key(tau, c2, q_jm1, d)

# ═══════════════════════════════════════════════════════════════
#  FHE Setup / KeyGen / Enc / Dec
# ═══════════════════════════════════════════════════════════════

def fhe_setup(lam=20, L=3, b=1, mu_override=None):
    """
    params_list[0]  = niveau L  (q_L = 2^{(L+1)·μ}, grand modulus)
    params_list[L]  = niveau 0  (q_0 = 2^μ,          petit modulus)
    d et χ uniformisés au niveau L.
    """
    mu = mu_override if mu_override else max(8, int(np.ceil(np.log2(lam) + np.log2(L + 1))) + 4)
    params_list = []
    for j in range(L, -1, -1):
        params_list.append(setup(lam=lam, mu=(j + 1) * mu, b=b))
    d_L, std_L = params_list[0]["d"], params_list[0]["std"]
    for p in params_list:
        p["d"]   = d_L
        p["std"] = std_L
    return params_list, L, mu

def fhe_keygen(params_list, L):
    """
    i=0 → niveau L,  i=L → niveau 0.
    sks[i]=s_{L−i},  pks[i]=A_{L−i},  taus[i]=τ_{s''_{L−i}→s_{L−i−1}}.
    """
    sks, pks, taus = [], [], []
    for params in params_list:
        sks.append(secret_key_gen(params))
        pks.append(public_key_gen(params, sks[-1]))

    for i in range(len(params_list) - 1):
        params_j, params_jm1 = params_list[i], params_list[i + 1]
        q_j, d_j             = params_j["q"], params_j["d"]
        s_prime_j = tensor_vec(sks[i], d_j, q_j)
        s_pp_j    = bitdecomp_vec(s_prime_j, q_j, d_j)
        print(f"  [KeyGen] niveau {L-i}: "
              f"|s'|={len(s_prime_j)}, |s''|={len(s_pp_j)}, "
              f"l={bit_length(q_j)}, q=2^{int(np.log2(q_j))}")
        taus.append(switch_key_gen(s_pp_j, sks[i + 1], params_jm1))

    return sks, pks, taus

def fhe_enc(params_list, pks, m_poly):
    return encrypt(params_list[0], pks[0], m_poly)

def fhe_dec(params_list, sks, ct, level_idx=0):
    return decrypt(params_list[level_idx], sks[level_idx], ct)

# ═══════════════════════════════════════════════════════════════
#  FHE Add / Mult
# ═══════════════════════════════════════════════════════════════

def fhe_add(c1, c2, params_j, tau_j, params_jm1, d):
    """
    XOR homomorphe dans R_2 : c3 = c1 + c2 mod q_j (sous s_j),
    puis Refresh → chiffré sous s_{j-1}.
    """
    q_j = params_j["q"]
    c3  = [poly_add(c1[k], c2[k], q_j) for k in range(len(c1))]
    return fhe_refresh(c3, tau_j, params_j, params_jm1, d)

def fhe_mult(c1, c2, params_j, tau_j, params_jm1, d):
    """
    Produit polynomial homomorphe dans R_2 = Z_2[x]/(x^d+1).
    c3 = c1 ⊗ c2 (sous s'_j), puis Refresh → chiffré sous s_{j-1}.

    NB : calcule m1·m2 dans R_2 (produit de polynômes mod x^d+1 et mod 2),
         PAS le AND bit à bit des coefficients.
    """
    q_j = params_j["q"]
    c3  = ciphertext_tensor(c1, c2, d, q_j)
    return fhe_refresh(c3, tau_j, params_j, params_jm1, d)

# ═══════════════════════════════════════════════════════════════
#  Test bout en bout
# ═══════════════════════════════════════════════════════════════

def measure_noise(params, sk, ct):
    q, d = params["q"], params["d"]
    
    # calcul du produit scalaire brut
    inner = vec_dot(ct, sk, d, q)
    
    # centrage dans (-q/2, q/2]
    ic = np.array([int(x) for x in inner], dtype=int)
    ic = ((ic + q // 2) % q) - q // 2
    
    # le bruit c'est ce qui reste après avoir enlevé le message (0 ou 1)
    msg = ic % 2
    noise = ic - msg  # ou ic - msg*1 si message polynomial
    
    return noise, np.max(np.abs(noise))


import matplotlib.pyplot as plt

def noise_experiment(params_list, sks, pks, taus, d, n_trials=20):
    L = len(params_list) - 1
    
    results = {"fresh": [], "after_add": [], "after_mult": [], "after_chain": []}
    limits  = {
        "fresh":       params_list[0]["q"] / 2,
        "after_add":   params_list[1]["q"] / 2,
        "after_mult":  params_list[1]["q"] / 2,
        "after_chain": params_list[2]["q"] / 2,
    }

    for _ in range(n_trials):
        m1 = sample_binary_poly(d)
        m2 = sample_binary_poly(d)

        ct1 = fhe_enc(params_list, pks, m1.copy())
        ct2 = fhe_enc(params_list, pks, m2.copy())

        _, b = measure_noise(params_list[0], sks[0], ct1)
        results["fresh"].append(b)

        ct_add = fhe_add(ct1, ct2, params_list[0], taus[0], params_list[1], d)
        _, b = measure_noise(params_list[1], sks[1], ct_add)
        results["after_add"].append(b)

        ct1b = fhe_enc(params_list, pks, m1.copy())
        ct2b = fhe_enc(params_list, pks, m2.copy())
        ct_mul = fhe_mult(ct1b, ct2b, params_list[0], taus[0], params_list[1], d)
        _, b = measure_noise(params_list[1], sks[1], ct_mul)
        results["after_mult"].append(b)

        ct3   = fhe_enc(params_list, pks, m1.copy())
        ct4   = fhe_enc(params_list, pks, m2.copy())
        ct_s1 = fhe_add(ct3, ct4, params_list[0], taus[0], params_list[1], d)
        m2_l1 = encrypt(params_list[1], pks[1], m2.copy())
        ct_s2 = fhe_mult(ct_s1, m2_l1, params_list[1], taus[1], params_list[2], d)
        _, b  = measure_noise(params_list[2], sks[2], ct_s2)
        results["after_chain"].append(b)

    # affichage
    labels = ["fresh\n(niveau L)", "après Add\n(niveau L-1)", 
              "après Mult\n(niveau L-1)", "après Chain\n(niveau L-2)"]
    keys   = ["fresh", "after_add", "after_mult", "after_chain"]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, label) in enumerate(zip(keys, labels)):
        vals = results[key]
        ax.scatter([i] * len(vals), vals, alpha=0.5, s=20)
        ax.hlines(np.mean(vals), i - 0.3, i + 0.3, colors="black", linewidth=2)
        ax.hlines(limits[key], i - 0.3, i + 0.3, colors="red", 
                  linewidth=1.5, linestyles="--", label="q/2" if i == 0 else "")

    ax.set_yscale("log")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bruit max (log scale)")
    ax.set_title(f"Évolution du bruit ({n_trials} répétitions)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # résumé numérique
    print(f"\n{'Étape':<25} {'moy':>10} {'max':>10} {'q/2':>15} {'ratio max/q*2':>15}")
    print("-" * 75)
    for key, label in zip(keys, labels):
        vals  = results[key]
        lim   = limits[key]
        label_clean = label.replace("\n", " ")
        print(f"{label_clean:<25} {np.mean(vals):>10.1f} {np.max(vals):>10.1f} "
              f"{lim:>15.0f} {np.max(vals)/lim:>15.4f}")
        

if __name__ == "__main__":
    np.random.seed(0)
    print("=== FHE Setup ===")

    params_list, L, mu = fhe_setup(lam=16, L=2, b=1)
    d = params_list[0]["d"]
    print(f"L={L}, μ={mu}, d={d}")
    for i, p in enumerate(params_list):
        print(f"  niveau {L-i}: q=2^{int(np.log2(p['q']))} = {p['q']}, n={p['n']}")

    print("\n=== FHE KeyGen ===")
    sks, pks, taus = fhe_keygen(params_list, L)
    print(f"  {len(sks)} clés secrètes, {len(taus)} clés de switching")

    m1 = sample_binary_poly(d)
    m2 = sample_binary_poly(d)

    # ── XOR homomorphe ───────────────────────────────────────
    print("\n=== Test XOR homomorphe ===")
    m_xor   = (m1 + m2) % 2
    ct1     = fhe_enc(params_list, pks, m1.copy())
    ct2     = fhe_enc(params_list, pks, m2.copy())
    ct_add  = fhe_add(ct1, ct2, params_list[0], taus[0], params_list[1], d)
    dec_add = fhe_dec(params_list, sks, ct_add, level_idx=1)
    print(f"  m1 XOR m2 (8 premiers): {m_xor[:8]}")
    print(f"  Déchiffré             : {dec_add[:8]}")
    print(f"  Correct               : {np.array_equal(dec_add, m_xor)}")

    # ── Produit polynomial homomorphe ────────────────────────
    print("\n=== Test produit polynomial dans R_2 ===")
    m_prod  = poly_mul(m1, m2, d, 2)      # m1·m2 dans Z_2[x]/(x^d+1)
    ct1b    = fhe_enc(params_list, pks, m1.copy())
    ct2b    = fhe_enc(params_list, pks, m2.copy())
    ct_mul  = fhe_mult(ct1b, ct2b, params_list[0], taus[0], params_list[1], d)
    dec_mul = fhe_dec(params_list, sks, ct_mul, level_idx=1)
    print(f"  m1·m2 dans R_2 (8 premiers): {m_prod[:8]}")
    print(f"  Déchiffré                  : {dec_mul[:8]}")
    print(f"  Correct                    : {np.array_equal(dec_mul, m_prod)}")

    # ── Chaîne 2 niveaux ─────────────────────────────────────
    if L >= 2:
        print("\n=== Chaîne : (m1 XOR m2) · m2  (2 niveaux) ===")
        ct3   = fhe_enc(params_list, pks, m1.copy())
        ct4   = fhe_enc(params_list, pks, m2.copy())
        ct_s1 = fhe_add(ct3, ct4, params_list[0], taus[0], params_list[1], d)
        m2_l1 = encrypt(params_list[1], pks[1], m2.copy())
        ct_s2 = fhe_mult(ct_s1, m2_l1, params_list[1], taus[1], params_list[2], d)

        m_chain   = poly_mul((m1 + m2) % 2, m2, d, 2)
        dec_chain = fhe_dec(params_list, sks, ct_s2, level_idx=2)
        print(f"  (m1 XOR m2)·m2 (8 premiers): {m_chain[:8]}")
        print(f"  Déchiffré                   : {dec_chain[:8]}")
        print(f"  Correct                     : {np.array_equal(dec_chain, m_chain)}")


        # après encrypt
        ct = fhe_enc(params_list, pks, m1)
        _, b_fresh = measure_noise(params_list[0], sks[0], ct)
        print(f"Bruit frais : {b_fresh}")  # doit être petit

        # après un add + refresh
        ct_add = fhe_add(ct1, ct2, params_list[0], taus[0], params_list[1], d)
        _, b_add = measure_noise(params_list[1], sks[1], ct_add)
        print(f"Bruit après Add+Refresh : {b_add}")

        # après un mult + refresh  
        ct_mul = fhe_mult(ct1, ct2, params_list[0], taus[0], params_list[1], d)
        _, b_mul = measure_noise(params_list[1], sks[1], ct_mul)
        print(f"Bruit après Mult+Refresh : {b_mul}")

        params_list, L, mu = fhe_setup(lam=16, L=2, b=1, mu_override=20)
        d = params_list[0]["d"]
        sks, pks, taus = fhe_keygen(params_list, L)
        noise_experiment(params_list, sks, pks, taus, d, n_trials=20)


