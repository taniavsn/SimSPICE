import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor

def moments(I, wvl, valid, centre, hw=0.7, dlam_nm=0.019):
    """I: (N, L) spectra of one window; valid: (L,) bool, False on padding."""
    bg_px = valid & (np.abs(wvl - centre) > 3 * hw)              # real, line-free pixels
    bg = np.median(I[:, bg_px], axis=1, keepdims=True)
    m = valid & (np.abs(wvl - centre) < hw)
    f = np.clip(I[:, m] - bg, 0, None); S = np.maximum(f.sum(1), 1e-12)
    cen = (f * wvl[m]).sum(1) / S
    sig = np.sqrt((f * (wvl[m] - cen[:, None]) ** 2).sum(1) / S)
    return S * dlam_nm, cen, sig                                 # radiance, centroid (Å), width (Å)

def probe(Zn, y, raster_id):
    cv = GroupKFold(n_splits=len(np.unique(raster_id)))          # leave-one-raster-out
    return {name: cross_val_score(mdl, Zn, y, groups=raster_id, cv=cv, scoring="r2").mean()
            for name, mdl in [("ridge", RidgeCV()), ("knn", KNeighborsRegressor(50))]}

def invariance_ratio(z, z_aug, rng):
    return (np.linalg.norm(z - z_aug, axis=1).mean()
            / np.linalg.norm(z - z[rng.permutation(len(z))], axis=1).mean())

def collapse(Z):
    zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False); p = s / s.sum()
    return {"std_x_sqrt_d": zn.std(0).mean() * np.sqrt(Z.shape[1]),  # about 1 when healthy
            "eff_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum()))}
