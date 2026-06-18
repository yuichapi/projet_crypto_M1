import gsw
import lwe
import rlwe

LWE_CONFIG = lwe.LweConfig(dimension=1024, noise_std=2 ** (-24))


LWE_CONFIG_NOISY = lwe.LweConfig(dimension=1024, noise_std=2**(-15))


RLWE_CONFIG = rlwe.RlweConfig(degree=1024, noise_std=2 ** (-24))

GSW_CONFIG = gsw.GswConfig(rlwe_config=RLWE_CONFIG, log_p=8)
