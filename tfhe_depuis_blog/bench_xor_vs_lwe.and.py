import time
import numpy as np
import matplotlib.pyplot as plt



def measure_noise(ct, lwe_key, expected_encoded):
    import lwe as lwe_mod
    phase = lwe_mod.lwe_decrypt(ct, lwe_key).message
    noise = np.int32(phase) - np.int32(expected_encoded)
    return int(noise)


def benchmark_xor_vs_add(n_gates: int = 10, n_add: int = 1000):
    import bootstrap, gsw, lwe, xor, config, utils


    # Setup
    lwe_key = lwe.generate_lwe_key(config.LWE_CONFIG_NOISY)
    gsw_key = gsw.convert_lwe_key_to_gsw(lwe_key, config.GSW_CONFIG)

    t0 = time.perf_counter()
    bk = bootstrap.generate_bootstrap_key(lwe_key, gsw_key)
    t_keygen = time.perf_counter() - t0
    print(f"Keygen : {t_keygen:.2f}s")

    ct_t = lwe.lwe_encrypt(lwe.lwe_encode_bool(True), lwe_key)
    ct_f = lwe.lwe_encrypt(lwe.lwe_encode_bool(False), lwe_key)
    noise_fresh = abs(measure_noise(ct_t, lwe_key, utils.encode_bool(True)))

    # --- XOR benchmark ---
    print(f"Lancement XOR x{n_gates}...")
    times_xor  = []
    noises_xor = []
    ct = ct_t

    for _ in range(n_gates):
        t0 = time.perf_counter()
        ct = xor.lwe_xor(ct, ct_f, bk)
        times_xor.append(time.perf_counter() - t0)
        noises_xor.append(abs(measure_noise(ct, lwe_key, utils.encode_bool(True))))

    # --- Addition LWE benchmark ---
    print(f"Lancement addition LWE x{n_add}...")
    times_add  = []
    noises_add = []
    ct_add = ct_t

    for _ in range(n_add):
        t0 = time.perf_counter()
        ct_add = lwe.lwe_add(ct_add, ct_f)
        times_add.append(time.perf_counter() - t0)
        noises_add.append(abs(measure_noise(ct_add, lwe_key, utils.encode_bool(True))))

    times_xor  = np.array(times_xor)
    noises_xor = np.array(noises_xor)
    times_add  = np.array(times_add)
    noises_add = np.array(noises_add)

    # --- Affichage texte ---
    print(f"\nBruit frais                : {noise_fresh}")
    print(f"\nXOR x{n_gates} (4 bootstrappings/porte)")
    print(f"  temps moyen/porte        : {times_xor.mean()*1000:.1f} ms")
    print(f"  temps total              : {times_xor.sum():.2f}s")
    print(f"  bruit moyen              : {noises_xor.mean():.0f}")
    print(f"  bruit max                : {noises_xor.max():.0f}")
    dec = lwe.lwe_decode_bool(lwe.lwe_decrypt(ct, lwe_key))
    print(f"  resultat correct         : {dec == True}")

    print(f"\nAddition LWE x{n_add} (0 bootstrapping)")
    print(f"  temps moyen/addition     : {times_add.mean()*1000:.4f} ms")
    print(f"  temps total              : {times_add.sum()*1000:.3f} ms")
    print(f"  bruit moyen              : {noises_add.mean():.0f}")
    print(f"  bruit max                : {noises_add.max():.0f}")
    dec_add = lwe.lwe_decode_bool(lwe.lwe_decrypt(ct_add, lwe_key))
    print(f"  resultat correct         : {dec_add == True}")

    # --- Plot ---
    gates_x = np.arange(1, n_gates + 1)
    add_x   = np.arange(1, n_add + 1)
    seuil   = 2**29

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"XOR homomorphe (x{n_gates}) vs Addition LWE (x{n_add})",
        fontsize=14
    )

    # 1. Temps XOR
    ax = axes[0, 0]
    ax.plot(gates_x, times_xor * 1000, marker='o', color='steelblue', linewidth=1.5)
    ax.axhline(times_xor.mean() * 1000, color='steelblue', linestyle='--',
               alpha=0.6, label=f"moyenne : {times_xor.mean()*1000:.1f} ms")
    ax.set_title(f"Temps par porte XOR x{n_gates} (avec bootstrapping)")
    ax.set_ylabel("Temps (ms)")
    ax.set_xlabel("Numero de porte")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Temps Addition
    ax = axes[0, 1]
    ax.plot(add_x, times_add * 1000, marker='o', color='steelblue',
            linewidth=1.5, markersize=3)
    ax.axhline(times_add.mean() * 1000, color='steelblue', linestyle='--',
               alpha=0.6, label=f"moyenne : {times_add.mean()*1000:.4f} ms")
    ax.set_title(f"Temps par addition LWE x{n_add} (sans bootstrapping)")
    ax.set_ylabel("Temps (ms)")
    ax.set_xlabel("Numero d'addition")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Bruit XOR
    ax = axes[1, 0]
    ax.semilogy(gates_x, np.maximum(noises_xor, 1), marker='o', color='darkorange',
                linewidth=1.5, label="bruit apres bootstrap")
    ax.axhline(max(noise_fresh, 1), color='green', linestyle='--', linewidth=1.5,
               label=f"bruit frais : {noise_fresh}")
    ax.axhline(seuil, color='red', linestyle='-', linewidth=1.5,
               label=f"seuil : $2^{{29}}$")
    ax.set_ylim(1, 2**33)
    ax.set_title("Bruit apres XOR (bootstrapping reinitialise le bruit)")
    ax.set_xlabel("Numero de porte")
    ax.set_ylabel("Bruit (log)")
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # 4. Bruit Addition
    ax = axes[1, 1]
    ax.semilogy(add_x, np.maximum(noises_add, 1), marker='o', color='darkorange',
                linewidth=1.5, markersize=3, label="bruit accumule")
    ax.axhline(max(noise_fresh, 1), color='green', linestyle='--', linewidth=1.5,
               label=f"bruit frais : {noise_fresh}")
    ax.axhline(seuil, color='red', linestyle='-', linewidth=1.5,
               label=f"seuil : $2^{{29}}$")
    ax.set_ylim(1, 2**33)
    ax.set_title("Bruit addition LWE (bruit qui s'accumule)")
    ax.set_xlabel("Numero d'addition")
    ax.set_ylabel("Bruit (log)")
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig("benchmark_xor_vs_add.png", dpi=150)
    plt.show()
    print("Figure sauvegardee : benchmark_xor_vs_add.png")


if __name__ == "__main__":
    benchmark_xor_vs_add(n_gates=5, n_add=10000)