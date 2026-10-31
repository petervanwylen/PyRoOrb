"""healpix.py: pixel numbers and inclusive discs equal to the HEALPix library's.

The pinned values were produced by healpy 1.19 (HEALPix C++), including points
exactly on pixel edges and corners, where the tie-break decides the pixel. The
debias tile table and the ITF tracklet index were both built with those
numbers, so any drift here silently mis-indexes them. When healpy happens to be
installed, a randomised comparison runs as well.
"""
import hashlib

import numpy as np
import pytest

from pyroorb import healpix

POINTS = [(0.0, 0.0), (180.0, 0.0), (11.25, 0.0), (45.0, 41.8103149),
          (45.0, -41.8103149), (112.5, -30.0), (0.0, 90.0), (0.0, -90.0),
          (359.9999999, 89.99999), (12.3, -4.5), (200.0, 12.0), (-10.0, 5.0),
          (370.0, 5.0), (90.0, 66.4), (270.0, -89.5)]
PIXELS = {
    (64, False): [24192, 24576, 24456, 7843, 41119, 36816, 0, 49148, 3, 26504,
                  19470, 22393, 22151, 2145, 49151],
    (64, True): [19456, 26282, 18026, 3072, 33791, 39786, 4095, 32768, 16383,
                 17946, 26570, 18907, 18151, 7167, 45056],
    (1024, True): [4980736, 6728362, 4614826, 786432, 8650751, 10185386, 1048575,
                   8388608, 4194303, 4594303, 6802085, 4840225, 4646674, 1835007,
                   11534472],
}
# (lon, lat, radius deg) -> (count, sha256[:16] of the sorted int64 pixels)
DISCS = {
    (10.0, -3.0, 2.5): (41, "4ade0ecab6c11117"),
    (0.0, 90.0, 1.0): (12, "93c3cf85655c9d79"),
    (359.5, 0.2, 0.7): (7, "4bd26843e9127094"),
    (123.0, -88.0, 5.0): (125, "0f9fd8abfd6a0607"),
    (45.0, 41.81, 12.0): (601, "7dda6f276b54ffb0"),
    (200.0, 10.0, 0.05): (2, "8ef625aabf61d94a"),
}


def _vec(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


@pytest.mark.parametrize("nside,nest", list(PIXELS))
def test_ang2pix_pinned(nside, nest):
    ra = np.array([p[0] for p in POINTS])
    dec = np.array([p[1] for p in POINTS])
    assert healpix.ang2pix(nside, ra, dec, nest=nest).tolist() == PIXELS[nside, nest]
    # the scalar path is the one the per-observation debias lookup takes
    assert [healpix.ang2pix(nside, r, d, nest=nest) for r, d in POINTS] == PIXELS[nside, nest]


def test_ang2pix_scalar_returns_int():
    assert isinstance(healpix.ang2pix(64, 12.3, -4.5), int)


@pytest.mark.parametrize("disc", list(DISCS))
def test_query_disc_pinned(disc):
    lon, lat, r = disc
    pix = healpix.query_disc_nest(64, _vec(lon, lat), np.radians(r))
    n, digest = DISCS[disc]
    assert len(pix) == n
    assert hashlib.sha256(pix.astype("<i8").tobytes()).hexdigest()[:16] == digest


def test_query_discs_batch_equals_single():
    discs = list(DISCS)
    vecs = np.array([_vec(lon, lat) for lon, lat, _ in discs])
    radii = np.radians([r for _, _, r in discs] + [0.0])
    vecs = np.vstack([vecs, _vec(1.0, 2.0)])
    batch = healpix.query_discs_nest(64, vecs, radii)
    for v, r, b in zip(vecs, radii, batch):
        assert np.array_equal(healpix.query_disc_nest(64, v, r), b)
    assert len(batch[-1]) >= 1                     # a zero-radius disc keeps its own pixel


def test_whole_sky_disc():
    assert len(healpix.query_disc_nest(8, _vec(0, 0), np.pi)) == 12 * 64


def test_nside_must_be_power_of_two():
    with pytest.raises(ValueError):
        healpix.ang2pix(48, 0.0, 0.0)


def test_against_healpy_when_installed():
    hp = pytest.importorskip("healpy")
    rng = np.random.default_rng(7)
    ra = rng.uniform(-20, 380, 20000)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 20000)))
    g_ra, g_dec = np.meshgrid(np.arange(0, 360, 1.40625), np.arange(-90, 90.01, 0.5))
    ra, dec = np.r_[ra, g_ra.ravel()], np.r_[dec, g_dec.ravel()]
    for nside, nest in [(1, False), (64, False), (64, True), (4096, True)]:
        assert np.array_equal(healpix.ang2pix(nside, ra, dec, nest=nest),
                              hp.ang2pix(nside, ra, dec, nest=nest, lonlat=True))
    lon = rng.uniform(0, 360, 300)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 300)))
    lat[:4] = [90.0, -90.0, 89.9, -89.9]
    radii = np.radians(np.exp(rng.uniform(np.log(0.02), np.log(30), 300)))
    vecs = np.array([_vec(a, b) for a, b in zip(lon, lat)])
    ours = healpix.query_discs_nest(64, vecs, radii)
    for v, r, p in zip(vecs, radii, ours):
        assert np.array_equal(np.sort(hp.query_disc(64, v, r, inclusive=True, nest=True)), p)
