import numpy as np
import rembg
import torch
import trimesh
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, grey_erosion
from transformers import pipeline

from model_bosalt import _ilk_katman_ustu, temizle

_derinlik_modeli = None


def _model():
    global _derinlik_modeli
    if _derinlik_modeli is None:
        _derinlik_modeli = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Large-hf",
            device=0 if torch.cuda.is_available() else -1,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    return _derinlik_modeli


def bellekten_cikar():
    global _derinlik_modeli
    if _derinlik_modeli is not None:
        _derinlik_modeli = None
        torch.cuda.empty_cache()


def _sinir_yumusat(verts, sinir, N, tekrar=6):
    # Piksel izgarasindan gelen merdiven gorunumlu kenari, sinir cizgisi boyunca
    # komsu ortalamasiyla yumusat (ust ve alt kenar ayni XY'yi paylasir).
    kose = np.unique(sinir)
    xy = verts[:, :2].copy()
    for _ in range(tekrar):
        toplam = np.zeros_like(xy)
        sayi = np.zeros(len(xy))
        np.add.at(toplam, sinir[:, 0], xy[sinir[:, 1]])
        np.add.at(toplam, sinir[:, 1], xy[sinir[:, 0]])
        np.add.at(sayi, sinir[:, 0], 1)
        np.add.at(sayi, sinir[:, 1], 1)
        ort = toplam[kose] / sayi[kose, None]
        xy[kose] = 0.5 * xy[kose] + 0.5 * ort
    verts[kose, :2] = xy[kose]
    verts[kose + N, :2] = xy[kose]


def _ic_bosalt(Z, maske, et_mm, piksel_mm):
    # Arka yuzeyi on yuzeyin her yonde et_mm uzagina koy: kure yapi elemanli gri
    # erozyon. Duz asagi kaydirmak dik yamaclarda duvari inceltirdi. Maske disi 0
    # sayildigi icin kenarlarda kendiliginden en az et_mm kalinliginda duvar kalir.
    r = max(1, int(np.ceil(et_mm / piksel_mm)))
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1] * piksel_mm
    uzaklik2 = ox ** 2 + oy ** 2
    ayak = uzaklik2 <= et_mm ** 2
    kure = np.where(ayak, np.sqrt(np.clip(et_mm ** 2 - uzaklik2, 0, None)), 0.0)
    ust = np.where(maske, Z, 0.0)
    alt = grey_erosion(ust, footprint=ayak, structure=kure, mode="constant", cval=0.0)
    return np.clip(alt, 0.0, None)


def kabartma_uret(image_path, rembg_session, genislik_mm=30.0, kabartma_mm=3.0,
                  taban_mm=1.5, izgara=450, bosalt=False, et_mm=0.8):
    img = Image.open(image_path).convert("RGB")
    maske = np.array(rembg.remove(img, session=rembg_session))[:, :, 3] > 127
    if not maske.any():
        raise ValueError("Gorselde nesne bulunamadi (arka plan ayrilamadi)")

    with torch.no_grad():
        d = _model()(img)["predicted_depth"].squeeze().float().cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    d = np.array(Image.fromarray(d).resize(img.size, Image.BICUBIC))

    ys, xs = np.where(maske)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    d, maske = d[y0:y1, x0:x1], maske[y0:y1, x0:x1]
    h, w = maske.shape
    olcek = izgara / max(h, w)
    gh, gw = max(2, int(h * olcek)), max(2, int(w * olcek))
    d = np.array(Image.fromarray(d).resize((gw, gh), Image.BILINEAR))
    maske = np.array(
        Image.fromarray(maske.astype(np.uint8) * 255).resize((gw, gh), Image.BILINEAR)
    ) > 127

    icerde = d[maske]
    lo, hi = np.percentile(icerde, 1), np.percentile(icerde, 99)
    dn = gaussian_filter(np.clip((d - lo) / (hi - lo + 1e-8), 0, 1), 0.7)
    return kabartma_mesh(dn, maske, genislik_mm, kabartma_mm, taban_mm, bosalt, et_mm)


def _maskeli_bulanik(a, maske, sigma):
    agirlik = gaussian_filter(maske.astype(float), sigma)
    return gaussian_filter(np.where(maske, a, 0.0), sigma) / np.maximum(agirlik, 1e-6)


def _on_yuz(mesh, izgara=300):
    # Her sutunda +Z'den (gorselin bakis yonunden) gorulen ilk yuzeyin yuksekligi.
    (xmin, ymin, _), (xmax, ymax, _) = mesh.bounds
    adim = max(xmax - xmin, ymax - ymin) / izgara
    x0, y0 = xmin - 2 * adim, ymin - 2 * adim
    nx, ny = int((xmax - x0) / adim) + 3, int((ymax - y0) / adim) + 3
    _, _, k, z = _ilk_katman_ustu(mesh, x0, y0, adim, nx, ny)
    F = np.full(nx * ny, -np.inf)
    np.maximum.at(F, k, z)
    F = F.reshape(nx, ny)
    xs = x0 + (np.arange(nx) + 0.5) * adim
    ys = y0 + (np.arange(ny) + 0.5) * adim
    return F, np.isfinite(F), xs, ys, izgara


def model_kabartma(mesh, genislik_mm=30.0, etki=1.0, kalinlik_mm=None):
    # 3D modeli (on yuz +Z) arkasi duz bir kabartmaya cevirir. Yukseklik haritasina
    # sikistirmak yerine modelin kendi geometrisi kullanilir: gaga, alt oyuklar ve ust
    # uste binen tuyler aynen kalir; sadece genis olcekli kivrim duzeltilir ve uydurma
    # arka duz kesilir. etki=1 modelin kendi derinligi, kucukse genel kivrim bastirilir.
    m = mesh.copy()
    m.apply_scale(genislik_mm / m.extents[0])

    # Model genelde geriye yatik uretiliyor: on yuze oturan duzlem Z'den cikarilir (kayma).
    # Dondurmek yerine kaydirmak onden gorunusu gorselle birebir ayni birakir; dondurme yan
    # donuk (profil) kafalari kameraya ceviriyordu.
    F, maske, xs, ys, _ = _on_yuz(m)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    A = np.c_[X[maske], Y[maske], np.ones(maske.sum())]
    (a, b, _), *_ = np.linalg.lstsq(A, F[maske], rcond=None)
    m.vertices[:, 2] -= a * m.vertices[:, 0] + b * m.vertices[:, 1]

    if etki < 1.0:
        # Kayma alani yumusak oldugu icin yerel sekiller bozulmadan sadece kanatlarin
        # arkaya bukulmesi gibi buyuk kivrimlar duzlesir.
        F, maske, xs, ys, izgara = _on_yuz(m)
        genis = _maskeli_bulanik(np.where(maske, F, 0.0), maske, izgara * 0.08)
        kayma = (1 - etki) * (genis - np.median(genis[maske]))
        kayma = _maskeli_bulanik(np.where(maske, kayma, 0.0), maske, 3)
        f = RegularGridInterpolator((xs, ys), kayma, bounds_error=False, fill_value=None)
        m.vertices[:, 2] -= f(m.vertices[:, :2])

    # Uydurma arkayi duz kes: on yuzun %97'si kalacak yukseklikten.
    F, maske, *_ = _on_yuz(m)
    zkes = np.percentile(F[maske], 3)
    (bx0, by0, _), (bx1, by1, bz1) = m.bounds
    kutu = trimesh.creation.box(bounds=[[bx0 - 1, by0 - 1, zkes], [bx1 + 1, by1 + 1, bz1 + 1]])
    sonuc = temizle(m.intersection(kutu, engine="manifold"))
    sonuc.apply_translation(-sonuc.bounds[0])
    if kalinlik_mm:
        sonuc.apply_scale([1.0, 1.0, kalinlik_mm / sonuc.extents[2]])
    return sonuc


def kabartma_mesh(dn, maske, genislik_mm, kabartma_mm, taban_mm, bosalt, et_mm):
    # dn: 0-1 arasi on yuz yuksekligi, maske: kabartmanin disi; satir 0 = ust kenar.
    gh, gw = maske.shape
    piksel_mm = genislik_mm / gw
    yy, xx = np.mgrid[0:gh, 0:gw]
    X = xx * piksel_mm
    Y = (gh - 1 - yy) * piksel_mm
    Z = taban_mm + dn * kabartma_mm
    Z_alt = _ic_bosalt(Z, maske, et_mm, piksel_mm) if bosalt else np.zeros_like(Z)

    N = gh * gw
    idx = np.arange(N).reshape(gh, gw)
    verts = np.vstack([
        np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1),
        np.stack([X.ravel(), Y.ravel(), Z_alt.ravel()], 1),
    ])

    kare = maske[:-1, :-1] & maske[1:, :-1] & maske[:-1, 1:] & maske[1:, 1:]
    a, b = idx[:-1, :-1][kare], idx[:-1, 1:][kare]
    c, e = idx[1:, 1:][kare], idx[1:, :-1][kare]
    ust = np.concatenate([np.stack([a, e, b], 1), np.stack([b, e, c], 1)])
    alt = ust[:, ::-1] + N

    kenarlar = np.concatenate([ust[:, [0, 1]], ust[:, [1, 2]], ust[:, [2, 0]]])
    _, ters, sayi = np.unique(np.sort(kenarlar, 1), axis=0,
                              return_inverse=True, return_counts=True)
    sinir = kenarlar[sayi[ters.ravel()] == 1]
    p, q = sinir[:, 0], sinir[:, 1]
    duvar = np.concatenate([np.stack([q, p, p + N], 1), np.stack([q, p + N, q + N], 1)])

    _sinir_yumusat(verts, sinir, N)

    mesh = trimesh.Trimesh(verts, np.vstack([ust, alt, duvar]), process=True)
    mesh.remove_unreferenced_vertices()
    parcalar = mesh.split(only_watertight=False)
    if len(parcalar) > 1:
        mesh = max(parcalar, key=lambda m: len(m.faces))
    return mesh
