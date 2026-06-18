import time
import numpy as np
import matplotlib.pyplot as plt


def measure_noise(ct, lwe_key, expected_encoded):
    import lwe
    phase = lwe.lwe_decrypt(ct, lwe_key).message  # = encode(m) + e
    noise = int(phase) - int(expected_encoded)     # = e en int Python
    # recentrage modulo 2^32
    if noise > 2**31:
        noise -= 2**32
    if noise < -2**31:
        noise += 2**32
    return noise


def benchmark_nand_vs_xor(n_gates: int = 10):
    import bootstrap, gsw, lwe, nand, xor, config, utils

    # Setup
    lwe_key = lwe.generate_lwe_key(config.LWE_CONFIG)
    gsw_key = gsw.convert_lwe_key_to_gsw(lwe_key, config.GSW_CONFIG)

    t0 = time.perf_counter()
    bk = bootstrap.generate_bootstrap_key(lwe_key, gsw_key)
    t_keygen = time.perf_counter() - t0
    print(f"Keygen : {t_keygen:.2f}s")

    # Table de verite
    inputs = [(True, True), (True, False), (False, True), (False, False)]
    nand_table = {(a, b): not (a and b) for a, b in inputs}
    xor_table  = {(a, b): a ^ b         for a, b in inputs}

    # Chiffres frais
    ct_t = lwe.lwe_encrypt(lwe.lwe_encode_bool(True),  lwe_key)
    ct_f = lwe.lwe_encrypt(lwe.lwe_encode_bool(False), lwe_key)
    noise_fresh = abs(measure_noise(ct_t, lwe_key, utils.encode_bool(True)))

    times_nand, noises_nand = [], []
    times_xor,  noises_xor  = [], []

 # Chiffres initiaux
    ct_left  = lwe.lwe_encrypt(lwe.lwe_encode_bool(True),  lwe_key)
    ct_right = lwe.lwe_encrypt(lwe.lwe_encode_bool(False), lwe_key)

    ct_cur_nand = ct_left
    ct_cur_xor  = ct_left

    for i in range(n_gates):
        # NAND enchaîné
        t0 = time.perf_counter()
        ct_cur_nand = nand.lwe_nand(ct_cur_nand, ct_right, bk)
        times_nand.append(time.perf_counter() - t0)
        # NAND(True, False) = True, NAND(True, False) = True, ...
        # le résultat alterne : True, True, True, ...
        expected_nand = True
        noises_nand.append(abs(measure_noise(ct_cur_nand, lwe_key,
                                             utils.encode_bool(expected_nand))))

        # XOR enchaîné
        t0 = time.perf_counter()
        ct_cur_xor = xor.lwe_xor(ct_cur_xor, ct_right, bk)
        times_xor.append(time.perf_counter() - t0)
        # XOR(True, False) = True, XOR(True, False) = True, ...
        expected_xor = True
        noises_xor.append(abs(measure_noise(ct_cur_xor, lwe_key,
                                            utils.encode_bool(expected_xor))))

    ct_nand = ct_cur_nand
    ct_xor  = ct_cur_xor

    last_expected_nand = expected_nand
    last_expected_xor  = expected_xor
    
    
    times_nand = np.array(times_nand)
    times_xor  = np.array(times_xor)
    noises_nand = np.array(noises_nand)
    noises_xor  = np.array(noises_xor)

    # --- Affichage texte ---
    print(f"\nBruit frais       : {noise_fresh}")
    print(f"\nNAND x{n_gates}")
    print(f"  temps moyen     : {times_nand.mean()*1000:.1f} ms")
    print(f"  temps total     : {times_nand.sum():.2f}s")
    print(f"  bruit moyen     : {noises_nand.mean():.0f}")
    print(f"  bruit max       : {noises_nand.max():.0f}")
    dec = lwe.lwe_decode_bool(lwe.lwe_decrypt(ct_nand, lwe_key))
    print(f"  resultat correct: {dec == True}")

    print(f"\nXOR x{n_gates} (= {4*n_gates} bootstrappings)")
    print(f"  temps moyen/XOR : {times_xor.mean()*1000:.1f} ms")
    print(f"  temps total     : {times_xor.sum():.2f}s")
    print(f"  bruit moyen     : {noises_xor.mean():.0f}")
    print(f"  bruit max       : {noises_xor.max():.0f}")
    dec = lwe.lwe_decode_bool(lwe.lwe_decrypt(ct_xor, lwe_key))
    print(f"  resultat correct: {dec == True}")

    # --- Plot ---
    gates = np.arange(1, n_gates + 1)
    seuil = 2**29

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Benchmark NAND vs XOR x{n_gates}", fontsize=14)

    # 1. Temps NAND
    ax = axes[0, 0]
    ax.plot(gates, times_nand * 1000, marker='o', color='steelblue', linewidth=1.5)
    ax.axhline(times_nand.mean() * 1000, color='steelblue', linestyle='--',
               alpha=0.6, label=f"moyenne : {times_nand.mean()*1000:.1f} ms")
    ax.set_title("Temps par porte NAND")
    ax.set_ylabel("Temps (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Bruit NAND
    ax = axes[1, 0]
    ax.plot(gates, noises_nand, marker='o', color='darkorange', linewidth=1.5,
            label="bruit apres bootstrap")
    ax.axhline(noise_fresh, color='green', linestyle='--', linewidth=1.5,
               label=f"bruit frais : {noise_fresh}")
    ax.axhline(seuil, color='red', linestyle='-', linewidth=1.5,
               label=f"seuil : $2^{{29}}$")
    ax.set_title("Bruit apres bootstrapping NAND")
    ax.set_xlabel("Numero de porte")
    ax.set_ylabel("Bruit (valeur absolue)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Temps XOR
    ax = axes[0, 1]
    ax.plot(gates, times_xor * 1000, marker='o', color='steelblue', linewidth=1.5)
    ax.axhline(times_xor.mean() * 1000, color='steelblue', linestyle='--',
               alpha=0.6, label=f"moyenne : {times_xor.mean()*1000:.1f} ms")
    ax.set_title("Temps par porte XOR (4 bootstrappings)")
    ax.set_ylabel("Temps (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Bruit XOR
    ax = axes[1, 1]
    ax.plot(gates, noises_xor, marker='o', color='darkorange', linewidth=1.5,
            label="bruit apres bootstrap")
    ax.axhline(noise_fresh, color='green', linestyle='--', linewidth=1.5,
               label=f"bruit frais : {noise_fresh}")
    ax.axhline(seuil, color='red', linestyle='-', linewidth=1.5,
               label=f"seuil : $2^{{29}}$")
    ax.set_title("Bruit apres bootstrapping XOR")
    ax.set_xlabel("Numero de porte")
    ax.set_ylabel("Bruit (valeur absolue)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("benchmark_nand_vs_xor.png", dpi=150)
    plt.show()
    print("Figure sauvegardee : benchmark_nand_vs_xor.png")


if __name__ == "__main__":
    benchmark_nand_vs_xor(n_gates=50)