import glob
import json
import os
import re
import subprocess
import threading
import time
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import rembg
import trimesh
from flask import Flask, request, jsonify, send_from_directory, render_template

import cizim_donustur
import hunyuan_uret
import kabartma
from model_bosalt import YONLER, model_bosalt, model_yukle

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

rembg_session = rembg.new_session()
# Ekran karti 6GB: derinlik modeli ile Hunyuan ayni anda sigmaz, ayni anda tek uretim.
gpu_kilidi = threading.Lock()
print("[3D-Uretici] Hazir, http://127.0.0.1:5050 adresinden acabilirsin.")


def kabartma_3d(image_path, job_dir, genislik_mm, kabartma_mm, taban_mm, izgara, bosalt, et_mm,
                yuksek_kalite, etki, kalinlik_mm):
    bilgi = None
    if yuksek_kalite:
        kabartma.bellekten_cikar()
        # 512'de tepe bellek 5.55 GB olculdu (6 GB kart); daha yukarisi sigmaz.
        octree = 512 if izgara >= 650 else 380
        model = hunyuan_uret.hunyuan_mesh(image_path, rembg_session, octree=octree)
        mesh = kabartma.model_kabartma(model, genislik_mm, etki, kalinlik_mm)
        if bosalt:
            mesh, bilgi = model_bosalt(mesh, et_mm, "-z")
    else:
        hunyuan_uret.bellekten_cikar()
        mesh = kabartma.kabartma_uret(
            image_path, rembg_session, genislik_mm, kabartma_mm, taban_mm, izgara, bosalt, et_mm
        )
    _gri_disa_aktar(mesh, job_dir)
    return bilgi


def tam3d(image_path, job_dir, genislik_mm, kalinlik_mm, adim):
    # Bosaltma burada yok: uretilen modelin arkasi yuvarlak ve uydurma oldugu icin tek
    # taraftan oyma kutu gibi levhalar ve kirik parcalar birakiyor.
    kabartma.bellekten_cikar()
    mesh = hunyuan_uret.tam3d_uret(image_path, rembg_session, genislik_mm, kalinlik_mm, adim)
    _gri_disa_aktar(mesh, job_dir)


def hazir_model_bosalt(model_path, job_dir, et_mm, yon, uzun_kenar_mm, bosalt):
    mesh, onarim = model_yukle(model_path)
    if uzun_kenar_mm:
        mesh.apply_scale(uzun_kenar_mm / mesh.extents.max())
    elif mesh.extents.max() < 3:
        raise ValueError(
            f"Model cok kucuk ({mesh.extents.max():.2f} mm). Dosya mm biriminde degil gibi; "
            "'Uzun kenar (mm)' alanina parcanin gercek boyutunu yaz."
        )
    if not bosalt:
        _gri_disa_aktar(mesh, job_dir)
        return {"onarim": onarim} if onarim else None
    sonuc, bilgi = model_bosalt(mesh, et_mm, yon)
    bilgi["onarim"] = onarim
    _gri_disa_aktar(sonuc, job_dir)
    return bilgi


def _gri_disa_aktar(mesh, job_dir):
    mesh.export(os.path.join(job_dir, "model.stl"))
    mesh.export(os.path.join(job_dir, "model.obj"))
    onizleme = mesh.copy()
    onizleme.visual = trimesh.visual.texture.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[140, 140, 140, 255], metallicFactor=0.0, roughnessFactor=0.7
        )
    )
    onizleme.export(os.path.join(job_dir, "model.glb"), include_normals=True)


def _sayi(ad, varsayilan, en_az, en_cok):
    try:
        deger = float(request.form.get(ad, varsayilan))
    except ValueError:
        deger = varsayilan
    return max(en_az, min(deger, en_cok))


@app.route("/")
def index():
    yanit = app.make_response(render_template("index.html"))
    # Tarayici eski sayfayi gostermesin; arayuz degisiklikleri hemen gelsin.
    yanit.headers["Cache-Control"] = "no-store"
    return yanit


@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return jsonify({"error": "Gorsel yuklenmedi"}), 400
    f = request.files["image"]

    job_id = uuid.uuid4().hex[:10]
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    ext = os.path.splitext(f.filename or "")[1].lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,5}", ext):
        ext = ".png"
    in_path = os.path.join(job_dir, "girdi" + ext)
    f.save(in_path)

    mod = request.form.get("mod", "kabartma")
    bilgi = None
    kalinlik_mm = (
        _sayi("kalinlik_mm", 6, 0.5, 200) if request.form.get("kalinlik_mm", "").strip() else None
    )
    if mod != "model" and ext in (".stl", ".obj"):
        return jsonify({"error": "Bu mod icin gorsel yukle"}), 400
    t0 = time.time()
    heykel_url = None
    konu = None
    try:
        gpu_kilidi.acquire()
        if mod != "model" and request.form.get("cizim") == "1":
            # Cizim once kabartma heykel gorseline cevrilir; SD modeli 3D modellerle ayni
            # anda ekran kartina sigmadigi icin is bitince hemen bellekten cikarilir.
            hunyuan_uret.bellekten_cikar()
            kabartma.bellekten_cikar()
            heykel = os.path.join(job_dir, "heykel.png")
            konu = cizim_donustur.cizimi_heykele_cevir(in_path, heykel)
            cizim_donustur.bellekten_cikar()
            in_path = heykel
            heykel_url = f"/outputs/{job_id}/heykel.png"
        if mod == "model":
            if ext not in (".stl", ".obj"):
                return jsonify({"error": "Bu mod icin STL ya da OBJ dosyasi yukle"}), 400
            yon = request.form.get("yon", "-z")
            bilgi = hazir_model_bosalt(
                in_path,
                job_dir,
                et_mm=_sayi("et_mikron", 800, 10, 10000) / 1000,
                yon=yon if yon in YONLER else "-z",
                uzun_kenar_mm=(
                    _sayi("uzun_kenar_mm", 30, 1, 500)
                    if request.form.get("uzun_kenar_mm", "").strip()
                    else None
                ),
                bosalt=request.form.get("bosalt") == "1",
            )
        elif mod == "tam3d":
            tam3d(
                in_path,
                job_dir,
                genislik_mm=_sayi("genislik_mm", 30, 5, 200),
                kalinlik_mm=kalinlik_mm,
                adim=30,
            )
        else:
            bilgi = kabartma_3d(
                in_path,
                job_dir,
                genislik_mm=_sayi("genislik_mm", 30, 5, 200),
                kabartma_mm=_sayi("kabartma_mm", 3, 0.3, 30),
                taban_mm=_sayi("taban_mm", 1.5, 0.3, 20),
                izgara=int(_sayi("izgara", 650, 150, 900)),
                bosalt=request.form.get("bosalt") == "1",
                et_mm=_sayi("et_mikron", 800, 10, 10000) / 1000,
                yuksek_kalite=request.form.get("kalite", "yuksek") == "yuksek",
                etki=_sayi("etki", 1.0, 0.2, 1.0),
                kalinlik_mm=kalinlik_mm,
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        gpu_kilidi.release()
    sure = round(time.time() - t0, 1)

    return jsonify(
        {
            "job_id": job_id,
            "sure_sn": sure,
            "obj_url": f"/outputs/{job_id}/model.obj",
            "stl_url": f"/outputs/{job_id}/model.stl",
            "glb_url": f"/outputs/{job_id}/model.glb",
            "heykel_url": heykel_url,
            "konu": konu,
            "bilgi": bilgi,
        }
    )


@app.route("/outputs/<job_id>/<filename>")
def outputs(job_id, filename):
    return send_from_directory(os.path.join(OUTPUT_DIR, job_id), filename)


def blender_yolu():
    adaylar = glob.glob(r"C:\Program Files\Blender Foundation\*\blender.exe")
    return sorted(adaylar)[-1] if adaylar else None


@app.route("/blender/<job_id>", methods=["POST"])
def blender_ac(job_id):
    if not re.fullmatch(r"[0-9a-f]{10}", job_id):
        return jsonify({"error": "Gecersiz model"}), 400
    stl_yolu = os.path.join(OUTPUT_DIR, job_id, "model.stl")
    if not os.path.exists(stl_yolu):
        return jsonify({"error": "Model dosyasi bulunamadi"}), 404
    blender = blender_yolu()
    if blender is None:
        return jsonify({"error": "Blender bulunamadi"}), 500
    subprocess.Popen(
        [blender, "--python", os.path.join(BASE_DIR, "blender_ac.py"), "--", stl_yolu]
    )
    return jsonify({"ok": True})


PROGRAM_AYAR = os.path.join(BASE_DIR, "programlar.json")


def _elle_yol(ad):
    # Otomatik bulunamazsa kullanici programlar.json'a exe (Matrix icin kisayol da olur) yolu yazar.
    try:
        with open(PROGRAM_AYAR, encoding="utf-8") as f:
            yol = (json.load(f).get(ad) or "").strip().strip('"')
        return yol if yol and os.path.exists(yol) else None
    except (OSError, ValueError):
        return None


def _baslat_menusu_kisayolu(desen):
    # Masaustu alt klasorleriyle taranmaz: binlerce dosya var, her istek dakikalarca takiliyordu.
    klasorler = [
        (os.path.join(os.environ.get("PROGRAMDATA", r"C:\ProgramData"), r"Microsoft\Windows\Start Menu\Programs"), True),
        (os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs"), True),
        (os.path.join(os.environ.get("USERPROFILE", ""), "Desktop"), False),
        (r"C:\Users\Public\Desktop", False),
    ]
    for k, alt_klasor in klasorler:
        desen_yol = os.path.join(k, "**", "*.lnk") if alt_klasor else os.path.join(k, "*.lnk")
        for lnk in glob.glob(desen_yol, recursive=alt_klasor):
            if re.search(desen, os.path.basename(lnk), re.I):
                return lnk
    return None


def zbrush_yolu():
    elle = _elle_yol("zbrush")
    if elle:
        return elle
    adaylar = []
    for kok in (r"C:\Program Files", r"C:\Program Files (x86)", r"D:\Program Files"):
        adaylar += glob.glob(os.path.join(kok, "Maxon ZBrush*", "ZBrush.exe"))
        adaylar += glob.glob(os.path.join(kok, "Pixologic", "ZBrush*", "ZBrush.exe"))
        adaylar += glob.glob(os.path.join(kok, "Maxon", "ZBrush*", "ZBrush.exe"))
    return sorted(adaylar)[-1] if adaylar else None


def matrix_yolu():
    # Matrix/MatrixGold Rhino uzerinde calisir; en guvenlisi kendi kisayolunu (dogru /scheme ile) kullanmak.
    elle = _elle_yol("matrix")
    if elle:
        return elle
    lnk = _baslat_menusu_kisayolu(r"matrix")
    if lnk:
        return lnk
    for rhino in sorted(glob.glob(r"C:\Program Files\Rhino*\System\Rhino.exe"), reverse=True):
        return rhino
    return None


def _model_yolu(job_id, dosya):
    if not re.fullmatch(r"[0-9a-f]{10}", job_id):
        return None
    yol = os.path.join(OUTPUT_DIR, job_id, dosya)
    return yol if os.path.exists(yol) else None


@app.route("/programlar")
def programlar():
    return jsonify({
        "blender": blender_yolu() is not None,
        "matrix": matrix_yolu() is not None,
        "zbrush": zbrush_yolu() is not None,
    })


@app.route("/matrix/<job_id>", methods=["POST"])
def matrix_ac(job_id):
    stl_yolu = _model_yolu(job_id, "model.stl")
    if stl_yolu is None:
        return jsonify({"error": "Model dosyasi bulunamadi"}), 404
    program = matrix_yolu()
    if program is None:
        return jsonify({"error": "Matrix bulunamadi. programlar.json dosyasina Matrix kisayolunun ya da Rhino.exe'nin yolunu yaz."}), 500
    if program.lower().endswith(".lnk"):
        # Kisayolun hedefi + argumanlari (ör. /scheme="MatrixGold") korunur, dosya sona eklenir.
        ps = ("$k=(New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK);"
              "Start-Process -FilePath $k.TargetPath -WorkingDirectory $k.WorkingDirectory "
              "-ArgumentList ($k.Arguments + ' /nosplash \"' + $env:MODEL + '\"')")
        subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                         env={**os.environ, "LNK": program, "MODEL": stl_yolu},
                         creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        subprocess.Popen([program, "/nosplash", stl_yolu])
    return jsonify({"ok": True})


@app.route("/zbrush/<job_id>", methods=["POST"])
def zbrush_ac(job_id):
    obj_yolu = _model_yolu(job_id, "model.obj")
    if obj_yolu is None:
        return jsonify({"error": "Model dosyasi bulunamadi"}), 404
    program = zbrush_yolu()
    if program is None:
        return jsonify({"error": "ZBrush bulunamadi. programlar.json dosyasina ZBrush.exe yolunu yaz."}), 500
    # ZBrush OBJ'yi komut satirindan dogrudan acmaz; acilista calisan kucuk bir ZScript ile
    # Tool > Import yapilir, model tuvale cizilip Edit moduna alinir.
    betik = os.path.join(OUTPUT_DIR, job_id, "zbrush_ac.txt")
    with open(betik, "w", encoding="utf-8") as f:
        f.write(
            '[FileNameSetNext,"' + obj_yolu.replace("\\", "/") + '"]\n'
            "[IPress,Tool:Import]\n"
            "[CanvasClick,400,400,600,600]\n"
            "[IPress,Transform:Edit]\n"
        )
    subprocess.Popen([program, betik])
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
