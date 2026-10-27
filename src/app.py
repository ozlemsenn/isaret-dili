"""Canlı işaret tanıma demosu (Kaggle'da eğitilen modelle).

Çalıştırma:   python app.py          (kamera açılmazsa: python app.py 1)
Kısayollar:   SPACE      elle kayıt başlat / bitir
              a          otomatik kaydı aç / kapat
              d          "diğer" sınıfını kapat / aç
              m          aynayı aç / kapat (varsayılan açık; tanımıyorsa kapatmayı dene)
              Backspace  son kelimeyi sil
              c          tüm kelimeleri temizle
              q          çıkış
"""
import os
import sys
import json
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from landmarks import make_landmarkers, to_vector, to_mp_image
from preprocess import prepare
from models import KucukGRU

KLASOR = os.path.dirname(os.path.abspath(__file__))
MODELLER = os.path.join(KLASOR, "models")

with open(os.path.join(MODELLER, "config.json"), encoding="utf-8") as f:
    CFG = json.load(f)
with open(os.path.join(MODELLER, "labels.json"), encoding="utf-8") as f:
    ETIKETLER = json.load(f)

ESIK = CFG["esik"]            # bu güvenin altındaki tahminler kabul edilmez
DIGER = CFG["diger_index"]    # "diğer" sınıfının numarası (yoksa None)
BITIS_BOSLUK = 12             # el bu kadar kare görünmezse kelime bitti sayılır (~0,4 sn)
MIN_KARE = 10                 # bundan kısa kayıtlar gürültü sayılır
MAX_KARE = 150                # bir kelime en fazla ~5 sn
KAMERA = int(sys.argv[1]) if len(sys.argv) > 1 else 0

model = KucukGRU(CFG["n_feat"], CFG["n_class"], CFG["hidden"])
model.load_state_dict(torch.load(os.path.join(MODELLER, "model.pt"), map_location="cpu"))
model.eval()
hand_lm, pose_lm = make_landmarkers(os.path.join(MODELLER, "hand_landmarker.task"),
                                    os.path.join(MODELLER, "pose_landmarker_full.task"))


def yazi_tipi(boyut):
    """Türkçe karakterleri (ğ, ş, ı, İ) çizebilen bir yazı tipi bulur."""
    adaylar = ["arial.ttf", "C:/Windows/Fonts/arial.ttf",
               "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for yol in adaylar:
        try:
            return ImageFont.truetype(yol, boyut)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=boyut)
    except TypeError:
        return ImageFont.load_default()


BUYUK, KUCUK = yazi_tipi(24), yazi_tipi(17)


DIGER_KAPALI = False          # d tuşu: "diğer" sınıfını yok sayıp en olası kelimeyi seçer
# AUTSL videolarındaki aktif el "sol el" kutusuna düşüyor (videolar aynalanmış kaydedilmiş).
# Bu yüzden kamera görüntüsünü de aynalayıp veriyoruz; varsayılan AÇIK. m tuşu kapatır.
# Model aynalama ile veri artırma kullanılarak yeniden eğitilirse buna gerek kalmayacak.
AYNA = bool(CFG.get("ayna", True))
son_tahminler = ""            # ekranda gösterilen ilk 3 tahmin


def tahmin_et(kareler):
    """Kaydedilen kareleri modele verir; her sınıfın olasılığını döndürür."""
    x = torch.from_numpy(prepare(np.array(kareler, np.float32), train=False, t=CFG["T"]))[None]
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0].numpy()
    if DIGER_KAPALI and DIGER is not None:
        p[DIGER] = 0.0
        p = p / p.sum()
    return p


def kelimeyi_bitir(kareler):
    """Kaydı değerlendirir; (eklenecek kelime ya da None, ekrana yazılacak bilgi) döndürür."""
    global son_tahminler
    if len(kareler) < MIN_KARE:
        return None, "Kayıt çok kısaydı, tekrar dene"
    p = tahmin_et(kareler)
    sira = np.argsort(p)[::-1][:3]
    son_tahminler = "Model: " + "  ".join(f"{ETIKETLER[str(i)]} %{p[i] * 100:.0f}" for i in sira)
    print(son_tahminler)
    idx, guven = int(sira[0]), float(p[sira[0]])
    ad = ETIKETLER[str(idx)]
    if idx == DIGER:
        return None, f"Tanınmadı (diğer, {guven:.2f}) - tekrar dene"
    if guven < ESIK:
        return None, f"Emin olamadım ({ad}? {guven:.2f}) - tekrar dene"
    return ad, f"Tanındı: {ad} ({guven:.2f})"


def kare_kirp(kare):
    """Eğitim videoları kare biçiminde (512x512) olduğu için kamera görüntüsü de ortadan kare kırpılır.
    Aksi halde 4:3 görüntüde yatay ve dikey koordinatlar farklı ölçeklenir ve model şaşırır."""
    h, w = kare.shape[:2]
    m = min(h, w)
    y0, x0 = (h - m) // 2, (w - m) // 2
    return kare[y0:y0 + m, x0:x0 + m].copy()


def kadraj_uyarisi(vek):
    """Omuzlar kadrajda değilse uyarı döndürür (eğitim videolarında omuzlar hep görünür)."""
    v = vek.reshape(-1, 3)
    if not v[:25].any():
        return "Vücudun görünmüyor: kameranın karşısına geç"
    omuzlar = v[[11, 12], :2]
    if (omuzlar < 0.03).any() or (omuzlar > 0.97).any():
        return "Omuzların kadrajda değil: biraz geri çekil"
    if np.linalg.norm(omuzlar[0] - omuzlar[1]) > 0.6:
        return "Kameraya çok yakınsın: biraz geri çekil"
    return ""


# MediaPipe'ın resmi bağlantı listeleri: hangi nokta hangisine bağlanır (iskeleti çizmek için).
EL_BAGLANTI = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
VUCUT_BAGLANTI = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),          # omuz - dirsek - bilek
                  (11, 23), (12, 24), (23, 24),                              # gövde
                  (15, 17), (15, 19), (15, 21), (17, 19),                    # sol el (vücut modelinin gördüğü)
                  (16, 18), (16, 20), (16, 22), (18, 20),                    # sağ el
                  (0, 2), (2, 7), (0, 5), (5, 8), (9, 10)]                   # yüz


def cizgileri_ciz(kare, vek):
    """Noktalar arasına iskelet çizgilerini çizer. Model bu çizgileri kullanmaz, sadece gösterim içindir."""
    h, w = kare.shape[:2]
    v = vek.reshape(-1, 3)

    def ciz(baslangic, baglantilar, renk):
        for a, b in baglantilar:
            pa, pb = v[baslangic + a], v[baslangic + b]
            if (pa[0] or pa[1]) and (pb[0] or pb[1]):        # iki ucu da bulunduysa çiz
                cv2.line(kare, (int(pa[0] * w), int(pa[1] * h)),
                         (int(pb[0] * w), int(pb[1] * h)), renk, 2, cv2.LINE_AA)

    ciz(0, VUCUT_BAGLANTI, (0, 170, 0))      # vücut: yeşil
    ciz(25, EL_BAGLANTI, (200, 60, 60))      # sol el: mavi
    ciz(46, EL_BAGLANTI, (60, 60, 200))      # sağ el: kırmızı


def noktalari_ciz(kare, vek):
    h, w = kare.shape[:2]
    for j, (x, y, _) in enumerate(vek.reshape(-1, 3)):
        if x or y:
            renk = (0, 255, 0) if j < 25 else (255, 0, 0) if j < 46 else (0, 0, 255)   # BGR
            cv2.circle(kare, (int(x * w), int(y * h)), 3, renk, -1)


def yazilari_ciz(kare, satirlar, alt_satir):
    """cv2.putText Türkçe harfleri çizemediği için yazılar PIL ile çizilir."""
    img = Image.fromarray(cv2.cvtColor(kare, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for k, (metin, renk) in enumerate(satirlar):
        d.text((12, 10 + 32 * k), metin, font=BUYUK, fill=renk, stroke_width=3, stroke_fill=(0, 0, 0))
    d.text((12, img.height - 28), alt_satir, font=KUCUK, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


cap = cv2.VideoCapture(KAMERA)
if not cap.isOpened():
    sys.exit(f"Kamera {KAMERA} açılamadı. Kamerayı kullanan başka bir uygulamayı kapat veya 'python app.py 1' dene.")

baslangic, son_ts = time.monotonic(), -1
durum, tampon, bosluk, elle = "BEKLE", [], 0, False
otomatik = True
kelimeler, bilgi = [], "Elini kaldır ve bir işaret yap"
fps, onceki = 0.0, time.monotonic()
print("Demo başladı. Çıkmak için pencere seçiliyken q tuşuna bas.")

while True:
    ok, kare = cap.read()
    if not ok:
        print("Kameradan görüntü alınamadı.")
        break
    kare = kare_kirp(kare)
    if AYNA:
        kare = cv2.flip(kare, 1)      # modele giden kareyi ters cevir
    ts = max(int((time.monotonic() - baslangic) * 1000), son_ts + 1)   # zaman damgası hep artmalı
    son_ts = ts
    img = to_mp_image(kare)                               # modele AYNALANMAMIŞ kare gider
    vek = to_vector(pose_lm.detect_for_video(img, ts), hand_lm.detect_for_video(img, ts))
    el_var = bool((vek[75:] != 0).any())
    sol_var, sag_var = bool((vek[75:138] != 0).any()), bool((vek[138:] != 0).any())
    aktif_el = "iki el" if sol_var and sag_var else "SOL" if sol_var else "SAĞ" if sag_var else "-"

    # Otomatik mod: el görününce kayıt başlar, el ~0,4 sn kaybolunca kelime biter.
    if durum == "BEKLE" and otomatik and el_var:
        durum, tampon, bosluk, elle = "KAYIT", [], 0, False
    if durum == "KAYIT":
        tampon.append(vek)
        bosluk = 0 if el_var else bosluk + 1
        if (not elle and bosluk >= BITIS_BOSLUK) or len(tampon) >= MAX_KARE:
            kelime, bilgi = kelimeyi_bitir(tampon[:len(tampon) - bosluk])
            if kelime:
                kelimeler.append(kelime)
            durum, elle = "BEKLE", False

    simdi = time.monotonic()
    fps = 0.9 * fps + 0.1 / max(simdi - onceki, 1e-6)
    onceki = simdi

    cizgileri_ciz(kare, vek)
    noktalari_ciz(kare, vek)
    ekran = cv2.resize(cv2.flip(kare, 1), (640, 640))     # sadece ekranda ayna görüntüsü
    if durum == "KAYIT":
        cv2.circle(ekran, (ekran.shape[1] - 28, 28), 12, (0, 0, 255), -1)
    satirlar = [
        (f"{durum}{' (elle)' if elle else ''} | Aktif el kutusu: {aktif_el}"
         f" | Ayna: {'AÇIK' if AYNA else 'kapalı'}", (255, 255, 255)),
        (f"Oto: {'açık' if otomatik else 'kapalı'} | Diğer: {'kapalı' if DIGER_KAPALI else 'açık'}"
         f" | FPS {fps:.0f}", (200, 200, 200)),
        (bilgi, (255, 230, 0)),
        (son_tahminler, (150, 220, 255)),
        ("Kelimeler: " + " ".join(kelimeler), (120, 255, 120)),
        (kadraj_uyarisi(vek), (255, 90, 90)),
    ]
    ekran = yazilari_ciz(ekran, satirlar, "SPACE: kayıt  a: oto  m: ayna  d: diğer  Backspace: sil  c: temizle  q: çıkış")
    cv2.imshow("Isaret Tanima", ekran)

    tus = cv2.waitKey(1) & 0xFF
    if tus == ord("q"):
        break
    elif tus == 32:                                       # SPACE: elle kayıt
        if durum == "KAYIT":
            kelime, bilgi = kelimeyi_bitir(tampon)
            if kelime:
                kelimeler.append(kelime)
            durum, elle = "BEKLE", False
        else:
            durum, tampon, bosluk, elle = "KAYIT", [], 0, True
            bilgi = "Elle kayıt: işareti yap, bitince SPACE"
    elif tus == ord("a"):
        otomatik = not otomatik
    elif tus == ord("m"):
        AYNA = not AYNA
        bilgi = "Ayna " + ("açıldı" if AYNA else "kapandı") + ": sol/sağ el ataması değişti"
    elif tus == ord("d") and DIGER is not None:
        DIGER_KAPALI = not DIGER_KAPALI
        bilgi = "Diğer sınıfı " + ("kapatıldı: hep en olası kelime seçilir" if DIGER_KAPALI else "açıldı")
    elif tus in (8, 127) and kelimeler:                   # Backspace (Windows: 8, macOS: 127)
        kelimeler.pop()
    elif tus == ord("c"):
        kelimeler, bilgi = [], "Temizlendi"

cap.release()
hand_lm.close()
pose_lm.close()
cv2.destroyAllWindows()
