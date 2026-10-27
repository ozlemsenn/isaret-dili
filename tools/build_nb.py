"""Kaggle notebook'unu (ipynb) üretir. Kullanım: python build_nb.py <src_klasoru> <cikti.ipynb>"""
import json
import os
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


def writefile(name):
    with open(os.path.join(SRC, name), encoding="utf-8") as f:
        code(f"%%writefile {name}\n" + f.read())


md("""
# İşaret Dili — Mini Kelime Tanıma (AUTSL, 3 kelime + "diğer")

Bu notebook baştan sona şunları yapar:

1. AUTSL videolarını ve etiketlerini bulur, seçtiğin kelimelerin videolarını ayırır.
2. **Görüntü işleme:** MediaPipe ile her videonun her karesinden el ve vücut noktalarını çıkarır.
3. **Makine öğrenmesi:** Küçük bir GRU modelini eğitir ve modelin hiç görmediği kişilerle test eder.
4. Kendi bilgisayarında kamerayla çalışacak demoyu `demo_paketi.zip` olarak hazırlar.

**Çalıştırmadan önce (sağ panel):**
- *Settings → Internet:* **On** (MediaPipe'ı ve model dosyalarını indirmek için; telefon doğrulaması isteyebilir)
- *Settings → Accelerator:* None yeterli, GPU gerekmez
- *Input:* AUTSL veri setin ekli olmalı

Sonra **Run All**. Bitince sağdaki *Output* bölümünden `demo_paketi.zip` dosyasını indir.

> Kaggle senin kameranı açamaz; notebook uzaktaki bir sunucuda çalışır. Kamera demosu zip'in içindeki `app.py` ile kendi bilgisayarında çalışır.
""")

md("## 1. Ayarlar\nSadece bu hücreyi değiştirmen yeterli.")
code('''
SECILEN_KELIMELER = ["anne", "doktor", "evet"]   # 2-3 kelime; listede yoksa en yakın ad kullanılır
DIGER_SINIFI = True      # True: başka kelimelerden "diğer" sınıfı eklenir; demoda yanlış alarmı azaltır
GORUNEN_AD = {}          # ekranda farklı yazılsın istersen, örnek: {"tesekkur": "teşekkür"}
ESIK = 0.6               # canlı demoda bu güvenin altındaki tahminler kabul edilmez
T = 48                   # her örnek bu kadar kareye getirilir
EPOCH = 80               # en fazla bu kadar tur eğitilir (iyileşme durursa erken biter)
VERI_KLASORU = "/kaggle/input"
CIKTI = "/kaggle/working"
''')

md("## 2. Kurulum ve MediaPipe model dosyaları")
code('''
!pip install -q mediapipe==0.10.32
import os, re, sys, glob, json, time, random, difflib, shutil, urllib.request, multiprocessing
import numpy as np, pandas as pd, cv2, torch
import matplotlib.pyplot as plt
import mediapipe as mp

os.chdir(CIKTI)
sys.path.insert(0, CIKTI)          # birazdan yazacağımız .py dosyaları buradan içe aktarılacak
print("mediapipe", mp.__version__, "| opencv", cv2.__version__, "| torch", torch.__version__)
''')
code('''
MODEL_URL = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}
for ad, url in MODEL_URL.items():
    if not os.path.exists(ad):
        try:
            urllib.request.urlretrieve(url, ad)
        except Exception as e:
            raise RuntimeError("Model indirilemedi. Settings -> Internet açık mı?") from e
    print(ad, round(os.path.getsize(ad) / 1e6, 1), "MB")
''')

md("## 3. Veri setini bul\nVideoları, etiket dosyalarını ve kelime listesini klasör adlarına bakmadan otomatik bulur.")
code(r'''
# 3a) Renkli videolar (derinlik videoları kullanılmaz)
def ornek_adi(yol):
    return re.sub(r"(_color)?\.mp4$", "", os.path.basename(yol), flags=re.I)

tum_mp4 = glob.glob(f"{VERI_KLASORU}/**/*.mp4", recursive=True)
VIDEO = {}
for p in sorted(tum_mp4):
    if not p.lower().endswith("_depth.mp4"):
        VIDEO.setdefault(ornek_adi(p), p)
print(len(tum_mp4), "mp4 dosyası,", len(VIDEO), "renkli video")
assert VIDEO, "Hiç video bulunamadı. Sağ panelden AUTSL'yi 'Add Input' ile eklediğinden emin ol."

# 3b) Etiket dosyaları: "signerX_sampleY,etiket" satırları içeren tüm CSV'ler
csvler = sorted(glob.glob(f"{VERI_KLASORU}/**/*.csv", recursive=True))
ETIKET = {}
for f in csvler:
    try:
        d = pd.read_csv(f, header=None, dtype=str, on_bad_lines="skip")
    except Exception:
        continue
    if d.shape[1] < 2:
        continue
    ad = d.iloc[:, 0].astype(str).map(lambda s: re.sub(r"(_color)?(\.mp4)?$", "", os.path.basename(s.strip())))
    no = pd.to_numeric(d.iloc[:, 1], errors="coerce")
    uygun = ad.str.match(r"^signer\d+_sample\d+$") & no.notna()
    if uygun.sum() > 0:
        ETIKET.update(dict(zip(ad[uygun], no[uygun].astype(int))))
        print(f"etiket dosyası: {f} -> {uygun.sum()} satır")
print("Toplam etiketli örnek:", len(ETIKET))
if not ETIKET:
    for f in csvler[:5]:
        print("-----", f); print(open(f, encoding="utf-8", errors="ignore").read(300))
    raise RuntimeError("Etiket dosyası bulunamadı; yukarıdaki CSV'lerin ilk satırlarını kontrol et.")

# 3c) Kelime listesi (etiket no -> kelime): ~226 satırlı, ilk sütunu 0,1,2,... olan CSV
SINIF = {}
for f in csvler:
    for baslik in (0, None):
        try:
            d = pd.read_csv(f, header=baslik)
        except Exception:
            continue
        if not (100 <= len(d) <= 400) or d.shape[1] < 2:
            continue
        no = pd.to_numeric(d.iloc[:, 0], errors="coerce")
        if no.notna().all() and set(no.astype(int)) == set(range(len(d))):
            kolon = next((c for c in d.columns if str(c).strip().upper() == "TR"), d.columns[1])
            SINIF = {int(n): str(w).strip() for n, w in zip(no, d[kolon])}
            print("kelime listesi:", f, "| kelime sütunu:", kolon)
            break
    if SINIF:
        break
if not SINIF and len(set(ETIKET.values())) == 226:
    # Bazı Kaggle kopyalarında kelime listesi dosyası yok; AUTSL'nin resmi listesi (SignList_ClassId_TR_EN) kullanılır.
    RESMI_LISTE = """abla acele acikmak afiyet_olsun agabey agac agir aglamak aile akilli akilsiz akraba alisveris anahtar anne arkadas ataturk ayakkabi ayna ayni baba bahce bakmak bal bardak bayrak bayram bebek bekar beklemek ben benzin beraber bilgi_vermek biz calismak carsamba catal cay caydanlik cekic cirkin cocuk corba cuma cumartesi cuzdan dakika dede degistirmek devirmek devlet doktor dolu dugun dun dusman duvar eczane eldiven emek emekli erkek et ev evet evli ezberlemek fil fotograf futbol gecmis gecmis_olsun getirmek gol gomlek gormek gostermek gulmek hafif hakli hali hasta hastane hata havlu hayir hayirli_olsun hayvan hediye helal hep hic hoscakal icmek igne ilac ilgilenmemek isik itmek iyi kacmak kahvalti kalem kalorifer kapi kardes kavsak kaza kemer keske kim kimlik kira kitap kiyma kiz koku kolonya komur kopek kopru kotu kucak leke maas makas masa masallah melek memnun_olmak mendil merdiven misafir mudur musluk nasil neden nerede nine ocak oda odun ogretmen okul olimpiyat olmaz olur onlar orman oruc ozur_dilemek pamuk pantolon para pastirma patates pazar pazartesi pencere persembe piknik polis psikoloji rica_etmek saat sabun salca sali sampiyon sapka savas seker selam semsiye sen senet serbest ses sevmek seytan sinir siz soylemek soz sut tamam tarak tarih tatil tatli tavan tehlike telefon terazi terzi tesekkur tornavida turkiye turuncu tuvalet un uzak uzgun var vergi yakin yalniz yanlis yapmak yarabandi yardim yarin yasak yastik yatak yavas yemek yemek_pisirmek yildiz yok yol yorgun yumurta zaman zor""".split()
    SINIF = dict(enumerate(RESMI_LISTE))
    print("Kelime listesi dosyası yok; AUTSL'nin resmi kelime listesi kullanılıyor.")
if not SINIF:
    print("UYARI: kelime listesi bulunamadı; kelimeleri numarayla seç (örn. SECILEN_KELIMELER = ['14', '52']).")
    SINIF = {i: str(i) for i in sorted(set(ETIKET.values()))}
print(len(SINIF), "kelime:", ", ".join(SINIF[i] for i in sorted(SINIF)))
''')

md("## 4. Kelimeleri seç ve kişi bazında böl\nTest setindeki kişiler eğitimde hiç bulunmaz; sonuç bu yüzden gerçek kullanımı yansıtır.")
code(r'''
def sade(s):   # karşılaştırma için: küçük harf, Türkçe harfler sade
    s = str(s).replace("İ", "i").replace("I", "ı").lower().strip()
    return s.translate(str.maketrans("çğıöşü", "cgiosu")).replace(" ", "_")

ad_no = {sade(w): i for i, w in SINIF.items()}
secilen = []
for k in SECILEN_KELIMELER:
    if sade(k) in ad_no:
        secilen.append(ad_no[sade(k)])
        continue
    yakin = difflib.get_close_matches(sade(k), list(ad_no), n=3, cutoff=0.6)
    if not yakin:
        raise ValueError(f"'{k}' kelime listesinde yok. Yukarıdaki listeden bir kelime seç.")
    print(f"'{k}' listede yok, en yakın kelime kullanılıyor: {yakin[0]}  (diğer öneriler: {yakin[1:]})")
    secilen.append(ad_no[yakin[0]])
secilen = list(dict.fromkeys(secilen))
assert len(secilen) >= 2, "En az 2 farklı kelime seçmelisin."

tum = pd.DataFrame([(s, n, VIDEO[s]) for s, n in ETIKET.items() if s in VIDEO], columns=["sample", "label", "path"])
tum["signer"] = tum["sample"].str.extract(r"signer(\d+)_")[0].astype(int)
print("Etiketi ve videosu olan örnek:", len(tum), "| işaretçi sayısı:", tum.signer.nunique())

SINIF_ADLARI = [GORUNEN_AD.get(SINIF[i], SINIF[i]) for i in secilen]
veri = tum[tum.label.isin(secilen)].copy()
veri["y"] = veri.label.map({n: k for k, n in enumerate(secilen)})
if DIGER_SINIFI:
    kalan = tum[~tum.label.isin(secilen)]
    adet = min(len(kalan), int(veri.groupby("y").size().mean()))
    diger = kalan.sample(n=adet, random_state=42).copy()
    diger["y"] = len(secilen)
    veri = pd.concat([veri, diger])
    SINIF_ADLARI.append("(diğer)")
veri = veri.reset_index(drop=True)

from sklearn.model_selection import GroupShuffleSplit
a, test_i = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(veri, groups=veri.signer))
egitdog = veri.iloc[a]
b, dog_i = next(GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=42).split(egitdog, groups=egitdog.signer))
veri["split"] = "train"
veri.loc[egitdog.index[dog_i], "split"] = "val"
veri.loc[veri.index[test_i], "split"] = "test"
kisiler = {s: set(veri[veri.split == s].signer) for s in ["train", "val", "test"]}
assert kisiler["train"].isdisjoint(kisiler["val"]) and kisiler["train"].isdisjoint(kisiler["test"]) \
    and kisiler["val"].isdisjoint(kisiler["test"]), "Aynı kişi iki bölmede olamaz!"

tablo = pd.crosstab(veri.y.map(dict(enumerate(SINIF_ADLARI))), veri.split).reindex(
    columns=["train", "val", "test"], fill_value=0)
print({s: len(k) for s, k in kisiler.items()}, "kişi (eğitim / doğrulama / test)")
print(tablo)
''')

md("""
### Seçilen kelimelerin örnek videoları
Kamera demosunda bu işaretleri sen yapacaksın; buradan izleyip çalış. Her kelimeden iki farklı kişi gösterilir.
Videodaki kişi **sağ elini** kullanıyorsa sen de **sağ elini** kullan (karşındaki birini taklit eder gibi, ayna gibi değil).
Videolar `ornek_videolar/` klasörüne de kaydedilir ve demo paketine eklenir.
""")
code(r'''
import io, base64
from PIL import Image
from IPython.display import HTML, display

def video_gif(yol, boyut=256, adim=2):
    """Videoyu, notebook'ta ve her bilgisayarda oynayan hareketli GIF'e çevirir."""
    cap = cv2.VideoCapture(yol)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    kareler, i = [], 0
    while True:
        ok, kare = cap.read()
        if not ok:
            break
        if i % adim == 0:
            kareler.append(Image.fromarray(cv2.cvtColor(cv2.resize(kare, (boyut, boyut)), cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    buf = io.BytesIO()
    kareler[0].save(buf, format="GIF", save_all=True, append_images=kareler[1:],
                    duration=int(1000 * adim / fps), loop=0)
    return buf.getvalue()

os.makedirs("ornek_videolar", exist_ok=True)
html = ""
for k in range(len(secilen)):
    ad = SINIF_ADLARI[k]
    html += f"<h4 style='margin:12px 0 4px'>{ad}</h4>"
    for _, r in veri[veri.y == k].drop_duplicates("signer").head(2).iterrows():
        gif = video_gif(r.path)
        with open(f"ornek_videolar/{sade(ad)}_{r['sample']}.gif", "wb") as f:
            f.write(gif)
        html += (f"<div style='display:inline-block;margin:4px;text-align:center'>"
                 f"<img src='data:image/gif;base64,{base64.b64encode(gif).decode()}' width='256'>"
                 f"<br><small>{r['sample']}</small></div>")
display(HTML(html))
''')

md("## 5. Görüntü işleme: nokta çıkarma kodu\nBu dosyalar hem burada hem kamera demosunda **aynen** kullanılır.")
writefile("landmarks.py")
writefile("preprocess.py")
writefile("models.py")

md("### Tek videoda dene ve gözle kontrol et\nYeşil: vücut, mavi: kişinin sol eli, kırmızı: sağ eli. Video aynalı olmadığı için kişinin sol eli görüntünün sağında durur.")
code(r'''
from landmarks import extract_video

ornek = veri[veri.y == 0].iloc[0]
dizi = extract_video(ornek.path)
el_orani = (dizi[:, 75:] != 0).any(axis=1).mean() if len(dizi) else 0.0
print(f"{ornek['sample']} ({SINIF_ADLARI[0]}): {dizi.shape[0]} kare, her kare {dizi.shape[1]} sayı, "
      f"en az bir elin bulunduğu kareler: %{el_orani * 100:.0f}")

k = int((dizi != 0).sum(axis=1).argmax())       # en çok noktanın bulunduğu kare
cap = cv2.VideoCapture(ornek.path)
for _ in range(k + 1):
    ok, kare = cap.read()
cap.release()
h, w = kare.shape[:2]
for j, (x, y, _) in enumerate(dizi[k].reshape(-1, 3)):
    if x or y:
        renk = (0, 255, 0) if j < 25 else (255, 0, 0) if j < 46 else (0, 0, 255)   # BGR
        cv2.circle(kare, (int(x * w), int(y * h)), 4, renk, -1)
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title(f"'{SINIF_ADLARI[0]}' işareti, kare {k}")
plt.savefig("ornek_kare.png", dpi=120, bbox_inches="tight"); plt.show()
''')

md("### Tüm videoları işle\nBirkaç yüz video için genelde 5-15 dakika sürer. Yarıda kalırsa tekrar çalıştır; biten videolar atlanır.")
code(r'''
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from landmarks import process_one

NOKTA = os.path.join(CIKTI, "noktalar")
os.makedirs(NOKTA, exist_ok=True)
t0 = time.time()
try:
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 2,
                             mp_context=multiprocessing.get_context("spawn")) as ex:
        sonuc = list(tqdm(ex.map(process_one, veri["sample"], veri["path"], [NOKTA] * len(veri), chunksize=2),
                          total=len(veri)))
except Exception as e:
    print("Paralel işlem çalışmadı, videolar tek tek işleniyor:", e)
    sonuc = [process_one(s, p, NOKTA) for s, p in tqdm(list(zip(veri["sample"], veri["path"])))]
hatali = [r for r in sonuc if str(r[1]).startswith("HATA")]
print(f"{len(sonuc)} video {(time.time() - t0) / 60:.1f} dakikada işlendi | hatalı: {len(hatali)}", hatali[:3])
''')
code(r'''
# Kalite kontrol: her videoda ellerin bulunduğu karelerin oranı
DIZI = {}
for s in veri["sample"]:
    f = os.path.join(NOKTA, s + ".npy")
    if os.path.exists(f):
        d = np.load(f).astype(np.float32)
        if len(d):
            DIZI[s] = d
veri = veri[veri["sample"].isin(DIZI)].copy()
veri["el_orani"] = veri["sample"].map(lambda s: (DIZI[s][:, 75:] != 0).any(axis=1).mean())
print("Ortalama el bulunma oranı (kelimeye göre):")
print(veri.groupby("y").el_orani.mean().rename(index=dict(enumerate(SINIF_ADLARI))).round(2))
dusuk = (veri.split == "train") & (veri.el_orani < 0.1)
print(f"Ellerin neredeyse hiç bulunmadığı {dusuk.sum()} eğitim videosu çıkarıldı (test setine dokunulmaz).")
veri = veri[~dusuk]
''')

md("""
## 6. Makine öğrenmesi: modeli eğit

En iyi model, doğrulama setindeki (eğitimde olmayan kişiler) başarıya göre seçilir.

**Aynalama ile veri artırma:** Eğitim örneklerinin yarısı aynalanarak gösterilir, yani sağ elle yapılan
bir işaret sol elle yapılmış gibi de öğretilir. AUTSL'deki 43 işaretçiden yalnızca 2'si solak olduğu için
bu olmadan model yalnızca tek bir eli tanıyor. Bölüm 7'de etkisi ölçülüyor.
""")
code(r'''
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from preprocess import prepare, mirror
from models import KucukGRU

random.seed(42); np.random.seed(42); torch.manual_seed(42)
cihaz = "cuda" if torch.cuda.is_available() else "cpu"

class IsaretVeri(Dataset):
    def __init__(self, d, egitim, ayna=False):
        self.ornek, self.y, self.egitim, self.ayna = list(d["sample"]), list(d["y"]), egitim, ayna
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        x = prepare(DIZI[self.ornek[i]], train=self.egitim, t=T)
        if self.ayna:                      # testte: her örneği diğer elle yapılmış gibi göster
            x = mirror(x)
        return torch.from_numpy(x), self.y[i]

egit_d, dog_d, test_d = (veri[veri.split == s] for s in ["train", "val", "test"])
egit_yk = DataLoader(IsaretVeri(egit_d, True), batch_size=32, shuffle=True)
dog_yk = DataLoader(IsaretVeri(dog_d, False), batch_size=64)
test_yk = DataLoader(IsaretVeri(test_d, False), batch_size=64)
test_ayna_yk = DataLoader(IsaretVeri(test_d, False, ayna=True), batch_size=64)   # aynalanmış test

@torch.no_grad()
def degerlendir(model, yk):
    model.eval()
    dogru = n = 0
    kayip = 0.0
    for x, y in yk:
        x, y = x.to(cihaz), y.to(cihaz)
        cikti = model(x)
        kayip += nn.functional.cross_entropy(cikti, y, reduction="sum").item()
        dogru += (cikti.argmax(1) == y).sum().item()
        n += len(y)
    return dogru / max(n, 1), kayip / max(n, 1)

C = len(SINIF_ADLARI)
model = KucukGRU(201, C, 128).to(cihaz)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
kayip_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
en_iyi, sabir, gecmis = (-1.0, 0.0), 0, []
for epoch in range(1, EPOCH + 1):
    model.train()
    toplam = 0.0
    for x, y in egit_yk:
        x, y = x.to(cihaz), y.to(cihaz)
        opt.zero_grad()
        kayip = kayip_fn(model(x), y)
        kayip.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        toplam += kayip.item() * len(y)
    d_basari, d_kayip = degerlendir(model, dog_yk)
    gecmis.append((epoch, toplam / len(egit_d), d_kayip, d_basari))
    if (d_basari, -d_kayip) > en_iyi:
        en_iyi, sabir = (d_basari, -d_kayip), 0
        torch.save(model.state_dict(), "en_iyi_model.pt")
    else:
        sabir += 1
    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:3d} | eğitim kaybı {toplam / len(egit_d):.3f} | doğrulama başarısı %{d_basari * 100:.1f}")
    if sabir >= 25:
        print(f"25 epoch boyunca iyileşme olmadı, epoch {epoch}'te durduruldu.")
        break
print(f"En iyi doğrulama başarısı: %{en_iyi[0] * 100:.1f}")

g = pd.DataFrame(gecmis, columns=["epoch", "egitim_kaybi", "dogrulama_kaybi", "dogrulama_basarisi"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(g.epoch, g.egitim_kaybi, label="eğitim"); ax[0].plot(g.epoch, g.dogrulama_kaybi, label="doğrulama")
ax[0].set_title("Kayıp"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(g.epoch, g.dogrulama_basarisi * 100); ax[1].set_title("Doğrulama başarısı (%)"); ax[1].set_xlabel("epoch")
plt.tight_layout(); plt.savefig("egitim_egrisi.png", dpi=120); plt.show()
''')

md("## 7. Test: modelin hiç görmediği kişiler")
code(r'''
model.load_state_dict(torch.load("en_iyi_model.pt", map_location=cihaz))
model.eval()
tahmin, gercek, guven = [], [], []
with torch.no_grad():
    for x, y in test_yk:
        p = torch.softmax(model(x.to(cihaz)), dim=1).cpu()
        g_, t_ = p.max(dim=1)
        tahmin += t_.tolist(); gercek += y.tolist(); guven += g_.tolist()
tahmin, gercek, guven = np.array(tahmin), np.array(gercek), np.array(guven)
TEST_BASARISI = float((tahmin == gercek).mean())
print(f"TEST BAŞARISI: %{TEST_BASARISI * 100:.1f}  ({len(gercek)} video, {test_d.signer.nunique()} kişi, "
      f"şans eseri tahmin %{100 / C:.0f})")

km = pd.crosstab(pd.Series([SINIF_ADLARI[i] for i in gercek], name="gerçek"),
                 pd.Series([SINIF_ADLARI[i] for i in tahmin], name="tahmin")).reindex(
                 index=SINIF_ADLARI, columns=SINIF_ADLARI, fill_value=0)
print(km)
plt.figure(figsize=(4.5, 4))
plt.imshow(km.values, cmap="Blues")
plt.xticks(range(C), SINIF_ADLARI, rotation=45); plt.yticks(range(C), SINIF_ADLARI)
for i in range(C):
    for j in range(C):
        plt.text(j, i, km.values[i, j], ha="center", va="center")
plt.xlabel("tahmin"); plt.ylabel("gerçek"); plt.title("Karışıklık matrisi (test)")
plt.tight_layout(); plt.savefig("karisiklik_matrisi.png", dpi=120); plt.show()

# --- El tercihinden bağımsızlık ölçümü (aynalama ile veri artırmanın etkisi) ---
ayna_basarisi, _ = degerlendir(model, test_ayna_yk)
print()
print("EL TERCİHİNDEN BAĞIMSIZLIK:")
print(f"  Normal test       : %{TEST_BASARISI * 100:.1f}")
print(f"  Aynalanmış test   : %{ayna_basarisi * 100:.1f}   (diğer elle yapılmış gibi)")
fark = abs(TEST_BASARISI - ayna_basarisi) * 100
print(f"  Fark              : {fark:.1f} puan ->", "iyi, model iki eli de tanıyor" if fark < 10
      else "yüksek, aynalama yeterince öğrenilmemiş")
print()

print("Güven eşiği seçimi (kamera demosu için):")
for esik in [0.4, 0.5, 0.6, 0.7, 0.8]:
    m = guven >= esik
    if m.any():
        print(f"  eşik {esik}: tahminlerin %{m.mean() * 100:.0f}'i kabul edilir, "
              f"kabul edilenlerin başarısı %{(tahmin[m] == gercek[m]).mean() * 100:.1f}")
''')

md("## 8. Kamera demosunu paketle\nAşağıdaki dosyalar senin bilgisayarında çalışacak demoyu oluşturur.")
writefile("app.py")
writefile("requirements.txt")
writefile("BENIOKU.txt")
code(r'''
PAKET = os.path.join(CIKTI, "isaret_demo")
shutil.rmtree(PAKET, ignore_errors=True)
os.makedirs(os.path.join(PAKET, "models"))
os.makedirs(os.path.join(PAKET, "sonuclar"))

torch.save(model.cpu().state_dict(), os.path.join(PAKET, "models", "model.pt"))
with open(os.path.join(PAKET, "models", "labels.json"), "w", encoding="utf-8") as f:
    json.dump({str(i): ad for i, ad in enumerate(SINIF_ADLARI)}, f, ensure_ascii=False, indent=2)
with open(os.path.join(PAKET, "models", "config.json"), "w", encoding="utf-8") as f:
    json.dump({"T": T, "n_feat": 201, "n_class": C, "hidden": 128, "esik": ESIK,
               "diger_index": C - 1 if DIGER_SINIFI else None,
               "ayna": False,   # model aynalama ile eğitildi; demo artık görüntüyü çevirmiyor
               "ayna_test_basarisi": round(ayna_basarisi, 4),
               "test_basarisi": round(TEST_BASARISI, 4)}, f, indent=2)
for ad in MODEL_URL:
    shutil.copy(ad, os.path.join(PAKET, "models"))
for ad in ["app.py", "landmarks.py", "preprocess.py", "models.py", "requirements.txt", "BENIOKU.txt"]:
    shutil.copy(ad, PAKET)
for ad in ["ornek_kare.png", "egitim_egrisi.png", "karisiklik_matrisi.png"]:
    if os.path.exists(ad):
        shutil.copy(ad, os.path.join(PAKET, "sonuclar"))
if os.path.isdir("ornek_videolar"):
    shutil.copytree("ornek_videolar", os.path.join(PAKET, "ornek_videolar"))

zip_yolu = shutil.make_archive(os.path.join(CIKTI, "demo_paketi"), "zip", CIKTI, "isaret_demo")
print("Hazır:", zip_yolu, round(os.path.getsize(zip_yolu) / 1e6, 1), "MB")
print("Kelimeler:", SINIF_ADLARI, "| test başarısı: %" + f"{TEST_BASARISI * 100:.1f}")
''')

md("""
## 9. Sonraki adım: kendi bilgisayarında kamera demosu

1. Sağdaki *Output* bölümünden `demo_paketi.zip` dosyasını indir ve bir klasöre çıkar.
2. `isaret_demo/BENIOKU.txt` dosyasındaki adımları izle: sanal ortam kur, `pip install -r requirements.txt`, sonra `python app.py`.
3. Elini kaldır, işareti yap, ellerini indir; tanınan kelime ekranda listeye eklenir.

Kelime değiştirmek istersen 1. hücredeki `SECILEN_KELIMELER` listesini değiştirip tekrar **Run All** de.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
    c["source"] = [line + "\n" for line in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("yazıldı:", OUT, len(cells), "hücre")
