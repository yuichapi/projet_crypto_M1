import bootstrap, lwe, utils


def lwe_xor(
    lwe_ciphertext_left: lwe.LweCiphertext,
    lwe_ciphertext_right: lwe.LweCiphertext,
    bootstrap_key: bootstrap.BootstrapKey,
) -> lwe.LweCiphertext:
    """Homomorphically evaluate the XOR function.

    Suppose that lwe_ciphertext_left is an LWE encryption of an encoding of the
    boolean b_left and lwe_ciphertext_right is an LWE encryption of an encoding
    of the boolean b_right. Then the output is an LWE encryption of an encoding
    of XOR(b_left, b_right).

    XOR is implemented using 4 NAND gates:
        XOR(a, b) = NAND(NAND(a, NAND(a,b)), NAND(b, NAND(a,b)))
    """
    from nand import lwe_nand

    # NAND(a, b)
    nand_ab = lwe_nand(lwe_ciphertext_left, lwe_ciphertext_right, bootstrap_key)

    # NAND(a, NAND(a, b))
    nand_left = lwe_nand(lwe_ciphertext_left, nand_ab, bootstrap_key)

    # NAND(b, NAND(a, b))
    nand_right = lwe_nand(lwe_ciphertext_right, nand_ab, bootstrap_key)

    # NAND(NAND(a, NAND(a,b)), NAND(b, NAND(a,b)))
    return lwe_nand(nand_left, nand_right, bootstrap_key)