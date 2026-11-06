"""Vegetation index computation.

Contents
--------
calculate_vegetation_index  Compute NDVI, CARI, DD, REP_LE, Clred_edge or REIP from a (H, W, bands) cube and
                            its wavelength list.
"""

import numpy as np


# index_name: name vegetation index to calculate, hyperspectral_data: numpy array (height, width, bands),
# wavelengths: list or array of wavelengthes corresponding to spectral bands
def calculate_vegetation_index(index_name, hyperspectral_data, wavelengths):
    wavelengths = np.array(wavelengths)

    def find_band(wl):
        if wl < wavelengths.min() or wl > wavelengths.max():
            print(f"Warning: Wavelength {wl} nm is outside the available range.")
        return np.argmin(np.abs(wavelengths - wl))

    index = np.zeros(hyperspectral_data.shape[:2])

    if index_name == 'NDVI':
      red_band = find_band(670)
      nir_band = find_band(800)

      red = hyperspectral_data[:, :, red_band]
      nir = hyperspectral_data[:, :, nir_band]

      index = (nir - red) / (nir + red + 1e-10)

    elif index_name == 'CARI':
        b670 = find_band(670)
        b700 = find_band(700)
        b550 = find_band(550)

        R670 = hyperspectral_data[:, :, b670]
        R700 = hyperspectral_data[:, :, b700]
        R550 = hyperspectral_data[:, :, b550]
        index = (R700 - R670) - 0.2 * (R700 - R550)


    elif index_name == 'DD':
        b749 = find_band(749)
        b720 = find_band(720)
        b701 = find_band(701)
        b672 = find_band(672)

        R749 = hyperspectral_data[:, :, b749]
        R720 = hyperspectral_data[:, :, b720]
        R701 = hyperspectral_data[:, :, b701]
        R672 = hyperspectral_data[:, :, b672]

        index = (R749 - R720) - (R701 - R672)

    elif index_name == 'REP_LE':
        far_red_bands = [672, 703]
        nir_bands = [732, 778]

        idx_far_red = [find_band(b) for b in far_red_bands]
        idx_nir = [find_band(b) for b in nir_bands]

        R_fr1 = hyperspectral_data[:, :, idx_far_red[0]]
        R_fr2 = hyperspectral_data[:, :, idx_far_red[1]]
        R_nir1 = hyperspectral_data[:, :, idx_nir[0]]
        R_nir2 = hyperspectral_data[:, :, idx_nir[1]]

        lambda_fr1 = wavelengths[idx_far_red[0]]
        lambda_fr2 = wavelengths[idx_far_red[1]]
        lambda_nir1 = wavelengths[idx_nir[0]]
        lambda_nir2 = wavelengths[idx_nir[1]]

        D_fr = (R_fr2 - R_fr1) / (lambda_fr2 - lambda_fr1)
        D_nir = (R_nir2 - R_nir1) / (lambda_nir2 - lambda_nir1)

        lambda_fr_mid = (lambda_fr1 + lambda_fr2) / 2
        lambda_nir_mid = (lambda_nir1 + lambda_nir2) / 2

        idx_fr1_prev = max(0, idx_far_red[0] - 1)
        idx_fr2_next = min(len(wavelengths)-1, idx_far_red[1] + 1)
        idx_nir1_prev = max(0, idx_nir[0] - 1)
        idx_nir2_next = min(len(wavelengths)-1, idx_nir[1] + 1)

        D_fr1 = (hyperspectral_data[:, :, idx_far_red[0]] - hyperspectral_data[:, :, idx_fr1_prev]) / (wavelengths[idx_far_red[0]] - wavelengths[idx_fr1_prev])
        D_fr2 = (hyperspectral_data[:, :, idx_fr2_next] - hyperspectral_data[:, :, idx_far_red[1]]) / (wavelengths[idx_fr2_next] - wavelengths[idx_far_red[1]])

        D_nir1 = (hyperspectral_data[:, :, idx_nir[0]] - hyperspectral_data[:, :, idx_nir1_prev]) / (wavelengths[idx_nir[0]] - wavelengths[idx_nir1_prev])
        D_nir2 = (hyperspectral_data[:, :, idx_nir2_next] - hyperspectral_data[:, :, idx_nir[1]]) / (wavelengths[idx_nir2_next] - wavelengths[idx_nir[1]])


        lambda_D_fr1 = (wavelengths[idx_far_red[0]] + wavelengths[idx_fr1_prev]) / 2
        lambda_D_fr2 = (wavelengths[idx_fr2_next] + wavelengths[idx_far_red[1]]) / 2
        lambda_D_nir1 = (wavelengths[idx_nir[0]] + wavelengths[idx_nir1_prev]) / 2
        lambda_D_nir2 = (wavelengths[idx_nir2_next] + wavelengths[idx_nir[1]]) / 2

        m1 = (D_fr2 - D_fr1) / (lambda_D_fr2 - lambda_D_fr1)
        c1 = D_fr1 - m1 * lambda_D_fr1

        m2 = (D_nir2 - D_nir1) / (lambda_D_nir2 - lambda_D_nir1)
        c2 = D_nir1 - m2 * lambda_D_nir1

        with np.errstate(divide='ignore', invalid='ignore'):
            rep = -(c1 - c2) / (m1 - m2)

            rep = np.where(np.abs(m1 - m2) < 1e-10, np.nan, rep)
            rep = np.where(np.isinf(rep), np.nan, rep)

        index = rep

    elif index_name == 'Clred_edge':
        b730 = find_band(730)
        b850 = find_band(850)

        R730 = hyperspectral_data[:, :, b730]
        R850 = hyperspectral_data[:, :, b850]

        index = R850 / (R730 + 1e-10)

    elif index_name == 'REIP':
        b670 = find_band(670)
        b700 = find_band(700)
        b740 = find_band(740)
        b780 = find_band(780)

        R670 = hyperspectral_data[:, :, b670]
        R700 = hyperspectral_data[:, :, b700]
        R740 = hyperspectral_data[:, :, b740]
        R780 = hyperspectral_data[:, :, b780]

        index = 700 + 40 * (((R670 + R780)/2) - R700) / (R740 - R700 + 1e-10)


    else:
        raise ValueError(f"Unknown vegetation index: {index_name}")

    return index
