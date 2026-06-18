import numpy as np
import matplotlib.pyplot as plt


def measure_noise(ct, lwe_key, expected_encoded):
    import lwe as lwe_mod
    phase = lwe_mod.lwe_decrypt(ct, lwe_key).message
    return int(np.int32(phase) - np.int32(expected_encoded))


def calibrer_sigma(lwe_key, n=2000):
    """Mesure l'ecart-type du bruit d'un chiffre frais (random bit)."""
    import lwe, utils
    rng = np.random.default_rng(1)
    bruits = []
    for _ in range(n):
        bit = bool(rng.integers(0, 2))
        ct = lwe.lwe_encrypt(lwe.lwe_encode_bool(bit), lwe_key)
        bruits.append(measure_noise(ct, lwe_key, utils.encode_bool(bit)))
    bruits = np.array(bruits)
    return bruits.std(), np.abs(bruits).max()


def add_phase_noise(ct, e, lwe_key):

    import copy
    M = 1 << 32 
    ct2 = copy.deepcopy(ct)
    ct2.b = np.uint32((int(ct2.b) + int(e)) % M)
    return ct2
   


def taux_erreur_injection(n_trials=2000, n_points=40, seed=0):
    import lwe, config, utils
    rng = np.random.default_rng(seed)
    lwe_key = lwe.generate_lwe_key(config.LWE_CONFIG_NOISY)
    seuil = 2 ** 28

    sigma, _ = calibrer_sigma(lwe_key)
    print(f"sigma bruit frais : {sigma:.0f}   seuil = 2^28 = {seuil}")

    amplitudes = np.linspace(0, 6 * seuil, n_points).astype(np.int64)

    taux = []
    for amp in amplitudes:
        err = 0
        for _ in range(n_trials):
            bit = bool(rng.integers(0, 2))
            ct = lwe.lwe_encrypt(lwe.lwe_encode_bool(bit), lwe_key)
            # bruit signe d'amplitude controlee
            e = int(rng.integers(-amp, amp + 1)) if amp > 0 else 0
            ct = add_phase_noise(ct, e, lwe_key)
            dec = lwe.lwe_decode_bool(lwe.lwe_decrypt(ct, lwe_key))
            if dec != bit:
                err += 1
        taux.append(err / n_trials)

    amplitudes = np.array(amplitudes)
    taux = np.array(taux)


    # --- figure ---
    fig, ax = plt.subplots(figsize=(9, 5))
    x = amplitudes / seuil
    ax.plot(x, taux, marker='o', color='crimson', label="taux d'erreur empirique")
    ax.axhline(0.5, color='black', linestyle='--', label="1/2 (aleatoire)")
    ax.axvline(1.0, color='steelblue', linestyle=':', label="seuil")
    ax.set_xlabel("amplitude du bruit injecte (en multiples du seuil $2^{28}$)")
    ax.set_ylabel("P(dechiffrement incorrect)")
    ax.set_ylim(-0.02, 0.6)
    ax.set_title(f"Taux d'erreur LWE vs bruit (n={n_trials} essais/point)")
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("taux_erreur_lwe.png", dpi=150)
    plt.show()
    print("Figure : taux_erreur_lwe.png")
    return amplitudes, taux


if __name__ == "__main__":
    taux_erreur_injection(n_trials=2000, n_points=40)