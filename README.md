# İşaret Dilinden Türkçe Cümleye

Bilgisayar Mühendisliği bitirme projesi — Bandırma Onyedi Eylül Üniversitesi, 2026–2027

Kameraya yapılan Türk İşaret Dili (TİD) işaretlerini kelime kelime tanıyıp, tanınan
kelime dizisini doğal dil işleme ile dilbilgisi açısından doğru Türkçe cümlelere çeviren
bir sistem.

**Projenin durumu:** Bu depoda şu an **1. aşama** (işaret tanıma) bulunuyor.
Dil işleme aşaması (kelime dizisi → Türkçe cümle) daha sonra `nlp/` klasörü olarak eklenecek.

## Sistem mimarisi

```
kamera → MediaPipe noktaları → kelime tanıma modeli → kelime dizisi → dil modeli → Türkçe cümle
         (görüntü işleme)      (makine öğrenmesi)                      (dil işleme)
         |________________ 1. AŞAMA: bu depoda ______________|        |__ 2. AŞAMA: sonra __|
```

Her kare 67 noktaya indirgenir (25 vücut + 21 sol el + 21 sağ el), her nokta (x, y, z)
koordinatıyla temsil edilir. Böylece kare başına **201 sayı** elde edilir. Ham görüntü
yerine iskelet verisi kullanıldığı için sistem arka plan, kıyafet ve ışık değişimlerine
dayanıklıdır ve dizüstü bilgisayarda gerçek zamanlı çalışır.

## Veri seti

[AUTSL](https://chalearnlap.cvc.uab.cat/dataset/40/description/) (Ankara Üniversitesi
Türk İşaret Dili Veri Seti): 226 kelime, 43 işaretçi, 36.302 video.

Veri seti kullanım koşulları nedeniyle **videolar bu depoda bulunmaz.**

## Klasör yapısı

| Klasör | İçerik |
| --- | --- |
| `src/` | Ana kod: nokta çıkarma, ön işleme, model, canlı demo |
| `notebooks/` | Kaggle'da çalıştırılan eğitim notebook'u |
| `demo/` | Kamera demosunu kurup çalıştıran dosyalar |
| `tools/` | Notebook'u `src/` içindeki koddan üreten betik |
| `models/` | Eğitilmiş modeller (depoya yüklenmez) |
| `sonuclar/` | Grafikler ve deney sonuçları |

### src/ içindeki dosyalar

- **`landmarks.py`** — Videodan/kameradan MediaPipe ile el ve vücut noktalarını çıkarır.
- **`preprocess.py`** — Kırpma, sabit uzunluğa getirme (48 kare), normalizasyon, veri artırma.
- **`models.py`** — Kelime tanıma modeli (çift yönlü GRU).
- **`app.py`** — Canlı kamera demosu.

`landmarks.py` ve `preprocess.py` hem eğitimde hem canlı demoda **aynen** kullanılır.
"Eğitimde iyi, kamerada kötü" sorununun en sık sebebi iki tarafta verinin farklı
işlenmesidir; bu tasarım onu baştan engeller.

## Çalışma düzeni

| Yer | Ne için |
| --- | --- |
| GitHub | Kodun asıl kopyası, sürüm geçmişi, yedek |
| Kaggle | Nokta çıkarma ve model eğitimi (ağır işler, bir kerelik) |
| Kendi bilgisayarı | Kod yazma, canlı demo, rapor |

### Notebook nasıl güncellenir

`notebooks/isaret_mini_egitim.ipynb` elle düzenlenmez; `src/` içindeki koddan üretilir:

```bash
python tools/build_nb.py src notebooks/isaret_mini_egitim.ipynb
```

Böylece notebook ile `src/` içindeki kod asla birbirinden ayrışmaz.

### Demoyu çalıştırma

`demo/BENIOKU.txt` dosyasına bakın. Kısaca: `demo/kurulum.bat` (bir kez),
sonra `demo/baslat.bat`.

## Şimdiye kadarki sonuçlar

| Deney | Kelime sayısı | Test doğruluğu | Not |
| --- | --- | --- | --- |
| Mini eğitim (GRU) | 3 + "diğer" | **%97,3** | 75 video, eğitimde olmayan 7 kişi |

Test bölmesi **kişi bazında** yapılır: test setindeki işaretçiler eğitimde hiç bulunmaz.
Bu, gerçek kullanımı (sistemi yeni biri kullanır) yansıtan katı bir değerlendirmedir.
Karşılaştırma için AUTSL'nin kendi temel modeli, 226 kelimede kişiden bağımsız testte
%62,02 almıştır.

## Bulgular

- **AUTSL videoları aynalanmış kaydedilmiş.** Bu nedenle veri setindeki aktif el,
  MediaPipe tarafından "sol el" olarak etiketleniyor. Canlı kamera girdisiyle
  eşleşmesi için aynalama gerekti.
- **Veri setindeki 43 işaretçiden yalnızca 2'si solak.** Bu yanlılık nedeniyle model
  tek bir eli tanıyordu. Eğitimde aynalama ile veri artırma uygulanarak sistem el
  tercihinden bağımsız hale getirildi ve etkisi ölçüldü.
- **AUTSL'de "gitmek" ve "gelmek" fiilleri yok.** Cümle şablonları veri setinde
  gerçekten bulunan fiillere göre kurulmalı.

## Kaynaklar

- Sincan, O. M. & Keleş, H. Y. (2020). [AUTSL: A Large Scale Multi-modal Turkish Sign Language Dataset and Baseline Methods](https://arxiv.org/abs/2008.00932). IEEE Access.
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
