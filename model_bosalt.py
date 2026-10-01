import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter, grey_erosion
from scipy.spatial import cKDTree

GUMUS_G_CM3 = 10.36
MAKS_SUTUN = 4_000_000
ACILIR_NZ = 0.5

# oyulacak taraf -> o tarafi -Z'ye getiren donus
YONLER = {
    "-z": np.eye(4),
    "+z": trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]),
    "-y": trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]),
    "+y": trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]),
    "-x": trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]),
    "+x": trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]),
}


def _sifir_alanlari_sil(mesh):
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()


def _parcalari_birlestir(mesh):
    # Sadece gercek kirintilar atilir. Asma halkasi gibi govdeye kaynaklanmamis ama anlamli
    # parcalar govdeyle birlestirilir (eskiden en buyuk parca disindakiler siliniyordu ve
    # kolye ucunun halkasi kayboluyordu).
    esik = max(10, int(len(mesh.faces) * 0.001))
    parcalar = [p for p in mesh.split(only_watertight=False) if len(p.faces) >= esik]
    if not parcalar:
        return mesh
    if len(parcalar) == 1:
        return parcalar[0]
    if all(p.is_watertight for p in parcalar):
        try:
            birlesik = trimesh.boolean.union(parcalar, engine="manifold")
            _sifir_alanlari_sil(birlesik)
            return birlesik
        except Exception:
            pass
    return trimesh.util.concatenate(parcalar)


def temizle(mesh):
    # Boolean islemler alani sifir minik ucgenler birakabiliyor; bunlar STL'e yazilinca
    # model "acik" okunuyor. Kaydetmeden once temizlenir.
    _sifir_alanlari_sil(mesh)
    mesh = _parcalari_birlestir(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _meshlab_onar(mesh):
    # Meshy gibi programlarin ciktilarinda ikiden fazla yuzeyin paylastigi kenarlar,
    # kendine degen delikler ve kopuk kirintilar olur. Standart delik kapatma kendine degen
    # delikleri kapatamiyor; bu yuzden her sorunlu noktanin cevresindeki birkac sira ucgen
    # silinip sade kenarli bir delik birakilir ve o kapatilir. Silinen alan cok kucuk
    # oldugu icin sekil degismez (Poisson ile yeniden kurma denendi, kalinligi 2 kat sisirdi).
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    kirinti = max(10, int(len(mesh.faces) * 0.001))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=kirinti)
    for _ in range(6):
        for secim in ("compute_selection_by_non_manifold_edges_per_face",
                      "compute_selection_by_non_manifold_per_vertex",
                      "compute_selection_from_mesh_border"):
            getattr(ms, secim)()
            if ms.current_mesh().selected_face_number() == 0:
                continue
            for _ in range(2):
                ms.apply_selection_dilatation()
            ms.meshing_remove_selected_vertices_and_faces()
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=kirinti)
        try:
            ms.meshing_close_holes(maxholesize=200000)
        except pymeshlab.PyMeshLabException:
            pass  # bir sonraki turda sorunlu bolge tekrar silinip denenir
        s = ms.current_mesh()
        onarilmis = trimesh.Trimesh(s.vertex_matrix(), s.face_matrix(), process=True)
        onarilmis.merge_vertices()
        if _kullanilabilir(onarilmis):
            break
    return onarilmis


def _kullanilabilir(mesh):
    return mesh.is_watertight and mesh.is_winding_consistent


def model_yukle(yol):
    # Donus: (mesh, onarim notu ya da None). Bozuk model reddedilmez; kademeli onarilir.
    mesh = trimesh.load(yol, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("Dosyada 3D model bulunamadi")
    mesh.merge_vertices()
    not_ = None
    if not _kullanilabilir(mesh):
        # Basit yama tam kapatmazsa sonucu atilir: yuzlerce acik kenari gelisiguzel yamayip
        # modeli karistiriyor ve arkasindan gelen guclu onarim da basarisiz oluyordu.
        deneme = mesh.copy()
        trimesh.repair.fill_holes(deneme)
        if _kullanilabilir(deneme):
            mesh = deneme
    if not _kullanilabilir(mesh):
        mesh = _meshlab_onar(mesh)
        not_ = "Modeldeki delikler, hatali kenarlar ve kopuk kirintilar otomatik onarildi."
    if not _kullanilabilir(mesh):
        raise ValueError("Model otomatik onarilamadi; dosyadaki hatalar cok fazla")
    mesh = temizle(mesh)
    return mesh, not_


def _ilk_katman_ustu(mesh, x0, y0, adim, nx, ny):
    # Her sutunda alttan (acilacak arka taraftan) yukari cikarken rastlanan ilk malzeme
    # katmaninin ust yuzeyi. En ust yuzey kullanilsaydi, alttan gorunmeyen girinti ve
    # tasmalarin altinda duvar incelirdi. Yukseklikler dogrudan ucgenlerden gelir.
    anahtarlar, zler, nzler = [], [], []
    ucgen = mesh.triangles
    nz = mesh.face_normals[:, 2]
    xy = ucgen[:, :, :2]
    a = xy[:, 1] - xy[:, 0]
    b = xy[:, 2] - xy[:, 0]
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    tut = np.abs(det) > 1e-14
    ucgen, xy, a, b, det, nz = ucgen[tut], xy[tut], a[tut], b[tut], det[tut], nz[tut]

    kay = 1e-7 * adim
    imin = np.ceil((xy[:, :, 0].min(1) - x0 - kay) / adim - 0.5).astype(np.int64)
    imax = np.floor((xy[:, :, 0].max(1) - x0 - kay) / adim - 0.5).astype(np.int64)
    jmin = np.ceil((xy[:, :, 1].min(1) - y0 - kay) / adim - 0.5).astype(np.int64)
    jmax = np.floor((xy[:, :, 1].max(1) - y0 - kay) / adim - 0.5).astype(np.int64)
    ni = np.clip(imax - imin + 1, 0, None)
    nj = np.clip(jmax - jmin + 1, 0, None)
    sayi = ni * nj

    sinirlar = np.searchsorted(np.cumsum(sayi), np.arange(0, sayi.sum(), 20_000_000), "right")
    sinirlar = list(sinirlar) + [len(sayi)]
    for s, e in zip(sinirlar[:-1], sinirlar[1:]):
        c = sayi[s:e]
        if c.sum() == 0:
            continue
        t = np.repeat(np.arange(s, e), c)
        yerel = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        ii = imin[t] + yerel // nj[t]
        jj = jmin[t] + yerel % nj[t]
        px = x0 + (ii + 0.5) * adim + kay
        py = y0 + (jj + 0.5) * adim + kay
        dx, dy = px - xy[t, 0, 0], py - xy[t, 0, 1]
        u = (dx * b[t, 1] - dy * b[t, 0]) / det[t]
        v = (a[t, 0] * dy - a[t, 1] * dx) / det[t]
        icinde = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        z = ucgen[t, 0, 2] + u * (ucgen[t, 1, 2] - ucgen[t, 0, 2]) + v * (ucgen[t, 2, 2] - ucgen[t, 0, 2])
        anahtarlar.append(ii[icinde] * ny + jj[icinde])
        zler.append(z[icinde])
        nzler.append(nz[t][icinde])

    ust = np.full(nx * ny, -np.inf)
    arka_nz = np.zeros(nx * ny)
    if not anahtarlar:
        return ust.reshape(nx, ny), arka_nz.reshape(nx, ny) > 0, np.array([], np.int64), np.array([])
    k = np.concatenate(anahtarlar)
    z = np.concatenate(zler)
    n = np.concatenate(nzler)
    sira = np.lexsort((z, k))
    k, z, n = k[sira], z[sira], n[sira]
    # iki komsu ucgenin ortak kenarina denk gelen ayni kesisimleri tekille
    tekil = np.ones(len(k), bool)
    tekil[1:] = (k[1:] != k[:-1]) | (z[1:] - z[:-1] > 1e-7)
    k, z, n = k[tekil], z[tekil], n[tekil]
    bas = np.r_[True, k[1:] != k[:-1]]
    ilk = np.flatnonzero(bas)
    adet = np.diff(np.r_[ilk, len(k)])
    ikinci = np.where(adet >= 2, ilk + 1, ilk)
    ust[k[ilk]] = z[ikinci]
    # Arka yuzey bu sutunda yeterince asagi bakiyorsa agiz acilabilir; dik/yan
    # yuzeyler acilirsa agiz kenari bicak gibi incelir. Tek ucgen normali gurultulu
    # oldugu icin egim once yumusatilir, yoksa agiz kenari disli cikar.
    arka_nz[k[ilk]] = n[ilk]
    malzeme = np.zeros(nx * ny)
    malzeme[k[ilk]] = 1.0
    arka_nz, malzeme = arka_nz.reshape(nx, ny), malzeme.reshape(nx, ny)
    yumusak = gaussian_filter(arka_nz, 1.5) / np.maximum(gaussian_filter(malzeme, 1.5), 1e-9)
    acilabilir = (malzeme > 0) & (yumusak < -ACILIR_NZ)
    return ust.reshape(nx, ny), acilabilir, k, z


def _metal_icinde(k, z, sutun, yukseklik):
    # Nokta, kendi sutununda altinda tek sayida yuzey varsa metalin icindedir.
    if len(k) == 0:
        return np.zeros(len(sutun), bool)
    zlo = z.min() - 1.0
    olcek = (z.max() - zlo + 1.0) * 1.0001
    birlesik = k + (z - zlo) / olcek
    bas = np.searchsorted(birlesik, sutun.astype(float))
    son = np.searchsorted(birlesik, sutun + np.clip((yukseklik - zlo) / olcek, 0, 0.9999))
    return (son - bas) % 2 == 1


def _izgara_kutu(X, Y, H, z_alt):
    # Ustu H yuksekliginde, alti z_alt'ta duz, kapali bir "izgara kutusu" mesh'i.
    nx, ny = H.shape
    N = nx * ny
    idx = np.arange(N).reshape(nx, ny)
    verts = np.vstack([
        np.stack([X.ravel(), Y.ravel(), H.ravel()], 1),
        np.stack([X.ravel(), Y.ravel(), np.full(N, z_alt)], 1),
    ])
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    ust = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    alt = ust[:, ::-1] + N
    kenar = np.concatenate([ust[:, [0, 1]], ust[:, [1, 2]], ust[:, [2, 0]]])
    _, ters, say = np.unique(np.sort(kenar, 1), axis=0, return_inverse=True, return_counts=True)
    sinir = kenar[say[ters.ravel()] == 1]
    p, q = sinir[:, 0], sinir[:, 1]
    duvar = np.concatenate([np.stack([q, p, p + N], 1), np.stack([q, p + N, q + N], 1)])
    m = trimesh.Trimesh(verts, np.vstack([ust, alt, duvar]), process=True)
    trimesh.repair.fix_normals(m)
    return m


def _oyuk(mesh, et_mm):
    (xmin, ymin, zmin), (xmax, ymax, _) = mesh.bounds
    alan = (xmax - xmin) * (ymax - ymin)
    adim = max(et_mm / 10, 0.02, np.sqrt(alan / MAKS_SUTUN))
    r = int(np.ceil(et_mm / adim))
    x0, y0 = xmin - (r + 2) * adim, ymin - (r + 2) * adim
    nx = int(np.ceil((xmax - x0) / adim)) + r + 2
    ny = int(np.ceil((ymax - y0) / adim)) + r + 2

    ust, acilabilir, kesit_k, kesit_z = _ilk_katman_ustu(mesh, x0, y0, adim, nx, ny)
    taban = zmin - 0.5
    # Agiz acilamayan sutunlar "disarisi" sayilir: erozyon onlarin cevresinde de
    # en az et_mm'lik bir kenar birakir.
    ust = np.where(np.isfinite(ust) & acilabilir, ust, taban - et_mm)

    # Kure yapi elemanli gri erozyon: ust yuzeyden her yonde et_mm iceride kalan yuzey.
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1] * adim
    uz2 = ox ** 2 + oy ** 2
    ayak = uz2 <= et_mm ** 2
    kure = np.where(ayak, np.sqrt(np.clip(et_mm ** 2 - uz2, 0, None)), 0.0)
    H = grey_erosion(ust, footprint=ayak, structure=kure, mode="constant", cval=taban - et_mm)
    # Kenar dislerini yumusat; sadece asagi cekildigi icin kalinlik hic azalmaz.
    H = np.minimum(H, gaussian_filter(H, 1.0))
    # Oyuk modelin altindan disari tasar (agiz acilir); model disinda kalan kismi zararsiz.
    H = np.maximum(H, taban)
    if (H > zmin).sum() == 0:
        raise ValueError("Model bu kalinlikta bosaltilamayacak kadar ince")

    ii, jj = np.mgrid[0:nx, 0:ny]
    X = x0 + (ii + 0.5) * adim
    Y = y0 + (jj + 0.5) * adim

    # Olcum icin: oyuk tavaninin gercekten metalin icinde kalan noktalari. Havada kalan
    # kisim parcaya dokunmaz, olcume katilmamali.
    aday = np.flatnonzero((H > zmin).ravel())
    icinde = _metal_icinde(kesit_k, kesit_z, aday, H.ravel()[aday])
    secili = aday[icinde]
    tavan = np.stack([X.ravel()[secili], Y.ravel()[secili], H.ravel()[secili]], 1)
    return _izgara_kutu(X, Y, H, zmin - 1.0), tavan


def model_bosalt(mesh, et_mm, yon="-z"):
    donus = YONLER[yon]
    calisma = mesh.copy()
    calisma.apply_transform(donus)

    oyuk, tavan = _oyuk(calisma, et_mm)
    sonuc = calisma.difference(oyuk, engine="manifold")
    if not isinstance(sonuc, trimesh.Trimesh) or len(sonuc.faces) == 0:
        raise ValueError("Bosaltma basarisiz oldu")
    sonuc = temizle(sonuc)

    # Olculen kalinlik: oyuk tavaninin orijinal yuzeye uzakligi. En yakin nokta asagi
    # bakan bir yuzeydeyse (arka taraf) orasi zaten acilan agizdir, olcume katilmaz.
    dis_nokta, yuz_id = trimesh.sample.sample_surface(calisma, 2_000_000, seed=0)
    olcum = None
    if len(tavan):
        uz, en_yakin = cKDTree(dis_nokta).query(tavan)
        arka = calisma.face_normals[yuz_id[en_yakin], 2] < -ACILIR_NZ
        uz = uz[~arka]
        if len(uz):
            olcum = {
                "medyan_mikron": round(float(np.median(uz)) * 1000),
                "en_ince_yuzde1_mikron": round(float(np.percentile(uz, 1)) * 1000),
                "en_ince_mikron": round(float(uz.min()) * 1000),
            }

    sonuc.apply_transform(np.linalg.inv(donus))
    bilgi = {
        "once_hacim_mm3": round(mesh.volume, 1),
        "sonra_hacim_mm3": round(sonuc.volume, 1),
        "once_gumus_gr": round(mesh.volume / 1000 * GUMUS_G_CM3, 2),
        "sonra_gumus_gr": round(sonuc.volume / 1000 * GUMUS_G_CM3, 2),
        "olcum": olcum,
        "watertight": bool(sonuc.is_watertight),
        "uyari": None,
    }
    if olcum and olcum["en_ince_yuzde1_mikron"] < 0.9 * et_mm * 1000:
        bilgi["uyari"] = (
            f"Bazi yerlerde kalinlik {olcum['en_ince_yuzde1_mikron']} mikrona dustu "
            f"(hedef {round(et_mm * 1000)}). Oyulan yon dogru mu kontrol et."
        )
    return sonuc, bilgi
