import os
import random
import sys

import numpy as np
import rembg
import torch
import trimesh
from PIL import Image

from kabartma import _on_yuz

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hunyuan3D-2"))
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402

# 6GB VRAM'de octree 380 ile tepe kullanim 4.6 GB olculdu; daha yukseği denenmedi.
OCTREE = 380

_boru = None


def _pipeline():
    global _boru
    if _boru is None:
        _boru = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini", variant="fp16"
        )
    return _boru


def bellekten_cikar():
    global _boru
    if _boru is not None:
        _boru = None
        torch.cuda.empty_cache()


def _siluet(maske, yukseklik=128):
    # Siluet kutusunu yukseklige gore olcekleyip sabit tuvale ortalar; en-boy orani korunur,
    # boylece yana donuk (daha dar) bir poz dusuk puan alir.
    r, c = np.nonzero(maske)
    kirp = maske[r.min():r.max() + 1, c.min():c.max() + 1]
    en = max(1, round(kirp.shape[1] * yukseklik / kirp.shape[0]))
    kucuk = np.array(Image.fromarray(kirp.astype(np.uint8) * 255).resize((en, yukseklik))) > 127
    tuval = np.zeros((yukseklik, yukseklik * 2), bool)
    bas = max(0, (tuval.shape[1] - en) // 2)
    tuval[:, bas:bas + min(en, tuval.shape[1])] = kucuk[:, :tuval.shape[1]]
    return tuval


def _poz_puani(kaba_mesh, hedef_siluet):
    F, maske, *_ = _on_yuz(kaba_mesh, izgara=150)
    s = _siluet(maske.T[::-1, :])
    return (s & hedef_siluet).sum() / max((s | hedef_siluet).sum(), 1)


def _donus(aci, merkez):
    return trimesh.transformations.rotation_matrix(np.radians(aci), [0, 1, 0], point=merkez)


def _en_iyi_aci(kaba_mesh, hedef_siluet):
    # Hunyuan profil (yan donuk) gorsellerde bile kafayi kameraya donuk "standart" pozda
    # uretebiliyor. Dikey eksende dondurup siluetin gorselle en iyi ortustugu aci aranir.
    # Onden bakan modellerin gereksiz donmemesi icin aci basina kucuk bir ceza var.
    en_iyi = (-1.0, 0, 0.0)
    for aci in range(-90, 91, 15):
        d = kaba_mesh.copy()
        d.apply_transform(_donus(aci, d.centroid))
        puan = _poz_puani(d, hedef_siluet)
        duzeltilmis = puan - 0.0005 * abs(aci)
        if duzeltilmis > en_iyi[0]:
            en_iyi = (duzeltilmis, aci, puan)
    return en_iyi[1], en_iyi[2]


def hunyuan_mesh(image_path, rembg_session, adim=30, octree=OCTREE, aday=4):
    # Ham 3D model, birim olcekte; on yuz +Z'ye (gorselin bakis yonune) bakar.
    # Her uretimde rastgele baslangic bazen gorsele uymayan (yana donuk, asimetrik) poz
    # veriyor. Poz kisa difuzyon adiminda belirlenir, pahali olan yuzey cikarmadir: birkac
    # baslangici kaba cozunurlukte deneyip siluet uyumu en iyi olani yuksek cozunurlukte cikar.
    img = rembg.remove(Image.open(image_path).convert("RGB"), session=rembg_session)
    hedef = _siluet(np.array(img)[:, :, 3] > 127)
    boru = _pipeline()
    en_iyi, en_iyi_puan, en_iyi_aci = None, -1.0, 0
    with torch.no_grad():
        for _ in range(aday):
            latent = boru(
                image=img,
                num_inference_steps=adim,
                generator=torch.manual_seed(random.randint(0, 2**31 - 1)),
                output_type="latent",
                enable_pbar=False,
            )
            kaba = boru._export(latent, "trimesh", octree_resolution=128, num_chunks=20000,
                                enable_pbar=False)[0]
            aci, puan = _en_iyi_aci(kaba, hedef)
            if puan > en_iyi_puan:
                en_iyi, en_iyi_puan, en_iyi_aci = latent, puan, aci
            # 0.95 puanli bir adayda bile basin yana donuk oldugu goruldu; ancak cok yuksek
            # uyumda erken dur.
            if puan > 0.97:
                break
        mesh = boru._export(en_iyi, "trimesh", octree_resolution=octree, num_chunks=20000,
                            enable_pbar=False)[0]
    torch.cuda.empty_cache()
    if en_iyi_aci:
        mesh.apply_transform(_donus(en_iyi_aci, mesh.centroid))

    mesh.merge_vertices()
    # Uretimde bazen birkac ucgenlik kopuk kirinti cikiyor ve modeli "acik" gosteriyor.
    mesh = max(mesh.split(only_watertight=False), key=lambda p: len(p.faces))
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh


def tam3d_uret(image_path, rembg_session, genislik_mm=30.0, kalinlik_mm=None, adim=30):
    mesh = hunyuan_mesh(image_path, rembg_session, adim)

    # On yuz +Z'ye bakiyor; kalinlik Z ekseni, arka yuz -Z (bosaltmanin varsayilan yonu).
    olcek = genislik_mm / mesh.extents[0]
    z_olcek = kalinlik_mm / mesh.extents[2] if kalinlik_mm else olcek
    mesh.apply_scale([olcek, olcek, z_olcek])
    mesh.apply_translation(-mesh.bounds[0])
    return mesh
