import numpy as np
def compute_snr(clean, denoised):
    noise = clean - denoised
    return 10 * np.log10(np.sum(clean**2) / np.sum(noise**2))
