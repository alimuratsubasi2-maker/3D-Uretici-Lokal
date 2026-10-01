# Yerel 3D Model Üretici

Görselden 3D model üreten, tamamen kendi bilgisayarında çalışan bir web uygulaması. İnternet bağlantısı, API anahtarı ya da abonelik gerekmez. Kuyumculuk için tasarlandı: gümüş kolye ucu, amblem ve yüzük tablası gibi işler, gerçek milimetre ölçüsünde.

## Neler yapıyor?

| Mod | Girdi | Çıktı |
|---|---|---|
| **Kabartma – Yüksek kalite 3D** | Görsel | Önü gerçek 3D, arkası düz kolye ucu (~6 dk) |
| **Kabartma – Hızlı** | Görsel | Derinlik haritasından kabartma (~20 sn) |
| **Tam 3D heykel** | Görsel | Her yönü şekilli model (~3 dk) |
| **Hazır model** | STL / OBJ | Bozuk dosyayı onarır, istenirse içini boşaltır |

- **İçi boşaltma:** Model arkadan oyulur, istenen et kalınlığı (mikron) bırakılır. Ölçülen gerçek kalınlık ile önceki ve sonraki gümüş gramajı (925 gümüş, 10,36 g/cm³) gösterilir.
- **Karakalem çizimler:** Çizim önce heykel görseline çevrilir, sonra 3D'ye dönüştürülür (Stable Diffusion 1.5 + ControlNet Lineart + BLIP).
- **Çıktılar:** STL, OBJ ve GLB dosyaları; tarayıcıda 3D önizleme; Blender, Matrix ve ZBrush'ta tek tıkla açma.

## Kullanılan modeller

- [Hunyuan3D-2mini](https://github.com/Tencent/Hunyuan3D-2): görselden 3D şekil
- [Depth Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf): hızlı kabartma için derinlik
- DreamShaper 8 + [ControlNet Lineart](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart): çizimden heykel görseli
- [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base): çizimdeki konuyu tanıma
- [rembg](https://github.com/danielgatis/rembg): arka plan silme

Modeller ilk kullanımda Hugging Face'ten otomatik indirilir.

## Gereksinimler

- Windows, Python 3.11
- NVIDIA ekran kartı (6 GB VRAM ile test edildi: RTX 4050 Laptop)
- İsteğe bağlı: Blender, Matrix (Rhino), ZBrush

## Kurulum

```bash
git clone https://github.com/<kullanici-adi>/3D-Uretici-Lokal.git
cd 3D-Uretici-Lokal

# PyTorch (CUDA 12.8)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# Diğer paketler
pip install -r requirements.txt

# Hunyuan3D-2 kodu bu klasörün içine klonlanır
git clone https://github.com/Tencent/Hunyuan3D-2.git
```

## Çalıştırma

`Baslat.bat` dosyasına çift tıkla. Tarayıcıda `http://127.0.0.1:5050` açılır. Doğrudan hazır model (STL/OBJ) moduyla açmak için: `http://127.0.0.1:5050/?mod=model`

Matrix veya ZBrush otomatik bulunamazsa, klasöre bir `programlar.json` dosyası koyup yollarını yaz:

```json
{ "zbrush": "C:\\Program Files\\Maxon ZBrush 2025\\ZBrush.exe", "matrix": "C:\\Users\\Public\\Desktop\\MatrixGold.lnk" }
```

## Dosyalar

| Dosya | Görevi |
|---|---|
| `app.py` | Flask sunucusu, istek akışı, ekran kartı kilidi, dışa aktarma |
| `hunyuan_uret.py` | Hunyuan3D-2mini ile şekil üretimi ve görsele en uygun poz seçimi |
| `kabartma.py` | Derinlik kabartması ve 3D modelden arkası düz kolye ucu |
| `model_bosalt.py` | STL/OBJ yükleme, onarım, mikron hassasiyetinde içini boşaltma |
| `cizim_donustur.py` | Karakalem çizimi heykel görseline çevirme |
| `blender_ac.py` | Modeli Blender'a aktarma |
| `templates/index.html` | Arayüz |

## Lisans notu

Hunyuan3D-2 modeli [Tencent Hunyuan 3D 2.0 Community License](https://github.com/Tencent/Hunyuan3D-2/blob/main/LICENSE) kapsamındadır. Bu lisans Avrupa Birliği, Birleşik Krallık ve Güney Kore'de geçerli değildir. Kullanmadan önce lisans koşullarını oku.
