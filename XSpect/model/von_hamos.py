import numpy as np


class vonHamos:
    def __init__(self):
        pass

    def dspacing_cubic(self, a, h, k, l):
        d = a / (np.sqrt(h**2 + k**2 + l**2))
        return d

    def dspacing_hexagonal(self, a, c, h, k, l):
        d = np.sqrt(1 / ((4/3) * ((h**2 + h*k + k**2) / (a**2)) + (l**2) / (c**2)))
        return d

    def dspacing(self, crystal, h, k, l):
        if crystal == 'Si':
            a = 5.430986
            d = self.dspacing_cubic(a, h, k, l)
        elif crystal == 'Ge':
            a = 5.65774
            d = self.dspacing_cubic(a, h, k, l)
        elif crystal == 'LiNbO3':
            a = 5.148
            c = 13.863
            d = self.dspacing_hexagonal(a, c, h, k, l)
        return d

    def bragg2eV(self, bragg_angle, dspacing):
        conversion_factor = 12398.419
        energy = conversion_factor / (2 * dspacing * np.sin(np.deg2rad(bragg_angle)))
        return energy

    def eV2bragg(self, energy, dspacing):
        conversion_factor = 12398.419
        bragg_angle = np.rad2deg(np.arcsin(conversion_factor / (energy * 2 * dspacing)))
        return bragg_angle

    def vH_energy_axis(self, avg_detector_distance, spectrum_length, crystal, h, k, l, crystal_radius, pixel_width=0.05):
        conversion_factor = 12398.419
        n_pix = np.arange(spectrum_length)
        d_rel = n_pix * pixel_width - (np.max(n_pix * pixel_width) - np.min(n_pix * pixel_width)) / 2
        dspacing = self.dspacing(crystal, h, k, l)
        energy = conversion_factor / (2 * dspacing * np.sin(np.arctan((2 * crystal_radius) / (d_rel + avg_detector_distance))))
        return energy
