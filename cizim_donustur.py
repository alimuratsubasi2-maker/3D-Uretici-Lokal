import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage

# Karakalem/murekkep cizimlerinde golge yerine ince cizgiler var; 3D modeller derinligi
# golgeden okudugu icin her cizgiyi ayri bir iplik sandi (yele spagetti gibi cikti). Once
# cizim, cizgilerini takip eden (ControlNet Lineart) bir kabartma heykel gorseline cevrilir.
# Konu istemde yazmazsa model cizgilerden ne oldugunu anlayamiyor: profil aslan insan yuzune
# donustu. Konu BLIP ile cizimden otomatik okunur ("a drawing of a lion" -> "a lion").
ISTEM = ("bas-relief sculpture of {konu}, silver jewelry pendant, carved in smooth rounded "
         "volumes, hair and fur as thick flowing sculpted locks merged into large solid masses, "
         "compact silhouette, deep relief, uniform light gray clay material, soft studio "
         "lighting, isolated on plain empty neutral gray background, highly detailed 3d render, "
         "zbrush sculpt")
NEGATIF = ("pencil, sketch, drawing, lineart, hatching, thin lines, text, watermark, color, flat, "
           "2d, photo, fur strands, separated strands, dangling hair, dripping, whiskers, thin "
           "wires, body, torso, neck stand, wall, pedestal, plinth, base, architecture, "
           "cropped, cut off, noisy, blurry")
CIZGI_BAGLILIGI = 0.8

_boru = None
_tanima = None


def konuyu_tani(yol):
    # Islemcide calisir (ekran kartini 3D modellere birakir), ~1-3 sn.
    global _tanima
    from transformers import BlipForConditionalGeneration, BlipProcessor

    if _tanima is None:
        ad = "Salesforce/blip-image-captioning-base"
        _tanima = (BlipProcessor.from_pretrained(ad),
                   BlipForConditionalGeneration.from_pretrained(ad))
    islemci, model = _tanima
    onek = "a drawing of"
    with torch.no_grad():
        cikti = model.generate(**islemci(Image.open(yol).convert("RGB"), onek, return_tensors="pt"),
                               max_new_tokens=25)
    metin = islemci.decode(cikti[0], skip_special_tokens=True).replace(" ' s", "'s").strip()
    konu = metin[len(onek):].strip() if metin.startswith(onek) else metin
    return konu or "an animal"


def _pipeline():
    global _boru
    if _boru is None:
        from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                               UniPCMultistepScheduler)

        cn = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_lineart", dtype=torch.float16, variant="fp16"
        )
        _boru = StableDiffusionControlNetPipeline.from_pretrained(
            "Lykon/dreamshaper-8", controlnet=cn, dtype=torch.float16, variant="fp16",
            safety_checker=None,
        )
        _boru.scheduler = UniPCMultistepScheduler.from_config(_boru.scheduler.config)
        _boru.to("cuda")
    return _boru


def bellekten_cikar():
    global _boru
    if _boru is not None:
        _boru = None
        torch.cuda.empty_cache()


def _cizgileri_hazirla(yol):
    # Cizgileri bul, en buyuk cizim kumesine kirp (alttaki yazi bandi gibi ayri parcalar
    # disarida kalir), ControlNet icin siyah zemin + beyaz cizgi.
    g = np.array(Image.open(yol).convert("L"), dtype=float)
    kume = ndimage.binary_closing(g < 200, iterations=12)
    etiket, n = ndimage.label(kume)
    if n == 0:
        raise ValueError("Gorselde cizim bulunamadi")
    boyut = ndimage.sum(kume, etiket, range(1, n + 1))
    ana = etiket == (np.argmax(boyut) + 1)
    ys, xs = np.nonzero(ana)
    kesit = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    kirp = g[kesit]
    siluet = ndimage.binary_fill_holes(ndimage.binary_closing(ana, iterations=20))[kesit]
    # Kenarlara bosluk birakilir; yoksa model konuyu kadraji dolduracak kadar buyutup burnu
    # kenara yapistiriyordu.
    pay = int(max(kirp.shape) * 0.12)
    kirp = Image.fromarray(np.pad(kirp, pay, constant_values=255).astype(np.uint8))
    siluet = Image.fromarray(np.pad(siluet, pay).astype(np.uint8) * 255)
    olcek = 768 / max(kirp.size)
    w, h = int(kirp.width * olcek) // 8 * 8, int(kirp.height * olcek) // 8 * 8
    kontrol = ImageOps.invert(kirp.resize((w, h), Image.LANCZOS)).convert("RGB")
    return kontrol, np.array(siluet.resize((w, h), Image.NEAREST)) > 127


def _siluet_disini_sil(img, siluet):
    # Model bos alani gorunce oraya govde, pence, kaide uyduruyordu (istemde yasaklansa
    # bile). Cizimin kendi siluetinin disi duz arka plana cevrilir; kenar yumusak gecisli.
    yumusak = ndimage.gaussian_filter(ndimage.binary_dilation(siluet, iterations=6)
                                      .astype(float), 3)[..., None]
    arr = np.array(img).astype(float)
    arka = np.array([128.0, 128.0, 128.0])
    return Image.fromarray((arr * yumusak + arka * (1 - yumusak)).astype(np.uint8))


def cizimi_heykele_cevir(girdi_yolu, cikti_yolu, tohum=7):
    # Donus: cizimde taninan konu (ornek "a lion's head").
    konu = konuyu_tani(girdi_yolu)
    kontrol, siluet = _cizgileri_hazirla(girdi_yolu)
    with torch.no_grad():
        img = _pipeline()(
            ISTEM.format(konu=konu), negative_prompt=NEGATIF, image=kontrol, num_inference_steps=30,
            guidance_scale=7.0, controlnet_conditioning_scale=CIZGI_BAGLILIGI,
            generator=torch.manual_seed(tohum), width=kontrol.width, height=kontrol.height,
        ).images[0]
    _siluet_disini_sil(img, siluet).save(cikti_yolu)
    return konu
