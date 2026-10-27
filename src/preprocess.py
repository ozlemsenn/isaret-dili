"""Nokta dizilerini modele hazırlama. Eğitimde ve canlı demoda AYNI dosya kullanılır."""
import numpy as np

N_PTS = 67    # 25 vücut + 21 sol el + 21 sağ el


def trim(seq):
    """Başta ve sonda hiç el görünmeyen kareleri atar."""
    has_hand = (seq[:, 75:] != 0).any(axis=1)
    idx = np.where(has_hand)[0]
    return seq if len(idx) == 0 else seq[idx[0]: idx[-1] + 1]


def random_time_crop(seq, rng=np.random):
    """Sadece eğitimde: dizinin rastgele %80-100'lük bir parçasını alır."""
    n = len(seq)
    if n < 10:
        return seq
    keep = int(n * rng.uniform(0.8, 1.0))
    start = rng.randint(0, n - keep + 1)
    return seq[start: start + keep]


def resample(seq, t):
    """Diziyi tam t kareye getirir: kısaysa kareleri tekrarlar, uzunsa seyreltir."""
    if len(seq) == 0:
        return np.zeros((t, N_PTS * 3), np.float32)
    idx = np.linspace(0, len(seq) - 1, t).round().astype(int)
    return seq[idx]


def normalize(seq):
    """Omuz ortasını merkeze alır, omuz genişliğini 1 birim yapar. Eksik noktalar 0 kalır."""
    s = seq.reshape(len(seq), N_PTS, 3).astype(np.float32).copy()
    missing = (s == 0).all(axis=2)
    ok = ~missing[:, 11] & ~missing[:, 12]          # iki omzun da bulunduğu kareler
    if ok.any():
        center = ((s[ok, 11, :2] + s[ok, 12, :2]) / 2).mean(axis=0)
        width = np.linalg.norm(s[ok, 11, :2] - s[ok, 12, :2], axis=1).mean()
    else:
        center, width = np.array([0.5, 0.5], np.float32), 0.25
    width = max(float(width), 1e-3)
    s[:, :, :2] = (s[:, :, :2] - center) / width
    s[:, :, 2] = s[:, :, 2] / width
    s[missing] = 0.0
    return s.reshape(len(s), -1)


def augment(seq, rng=np.random):
    """Sadece eğitimde: küçük döndürme, ölçek, kaydırma ve gürültü."""
    s = seq.reshape(len(seq), N_PTS, 3).copy()
    missing = (s == 0).all(axis=2)
    a = np.deg2rad(rng.uniform(-10, 10))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    s[:, :, :2] = (s[:, :, :2] @ rot.T) * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1, 2)
    s += rng.normal(0, 0.01, s.shape).astype(np.float32)
    s[missing] = 0.0
    return s.reshape(len(s), -1)


# Vücuttaki sol/sağ eş noktalar: aynalayınca bunlar yer değiştirmeli.
# (1,4) göz içi, (2,5) göz, (3,6) göz dışı, (7,8) kulak, (9,10) ağız köşesi,
# (11,12) omuz, (13,14) dirsek, (15,16) bilek, (17,18) serçe, (19,20) işaret,
# (21,22) başparmak, (23,24) kalça. 0 (burun) ortada olduğu için yerinde kalır.
ESLER = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12),
         (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]


def mirror(seq):
    """Örneği aynalar: sağ elle yapılan işaret, sol elle yapılmış gibi olur.
    normalize() sonrası çağrılmalı (koordinatlar 0 etrafında ortalanmış olmalı)."""
    s = seq.reshape(len(seq), N_PTS, 3).copy()
    s[:, :, 0] = -s[:, :, 0]                       # yatay eksende çevir
    for a, b in ESLER:                             # vücut eşlerini takas et
        s[:, [a, b]] = s[:, [b, a]]
    sol = s[:, 25:46].copy()                       # sol el <-> sağ el
    s[:, 25:46] = s[:, 46:67]
    s[:, 46:67] = sol
    return s.reshape(len(s), -1)


def prepare(seq, train=False, t=48):
    """Ham dizi -> modele girecek (t, 201) dizi. Canlı demoda train=False ile çağrılır."""
    seq = trim(np.asarray(seq, np.float32))
    if train:
        seq = random_time_crop(seq)
    seq = resample(seq, t)
    seq = normalize(seq)
    if train:
        if np.random.random() < 0.5:   # örneklerin yarısı aynalanır -> model el tercihinden bağımsız olur
            seq = mirror(seq)
        seq = augment(seq)
    return seq.astype(np.float32)
