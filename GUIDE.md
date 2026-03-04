# DentalAI - Kapsamli Ogrenci Rehberi

> Bu rehber, 3B dental AI projesi hakkinda **hic bilgisi olmayan** universite ogrencileri icin hazirlanmistir.
> Temel kavramlardan baslayarak projenin her bileseni adim adim anlatilmaktadir.

---

# ICERIK

- [Bolum 1: Temel Kavramlar](#bolum-1-temel-kavramlar)
- [Bolum 2: Dis Hekimligi ve Yapay Zeka](#bolum-2-dis-hekimligi-ve-yapay-zeka)
- [Bolum 3: Teeth3DS+ Veri Seti](#bolum-3-teeth3ds-veri-seti)
- [Bolum 4: Proje Mimarisi](#bolum-4-proje-mimarisi)
- [Bolum 5: Kod Rehberi - Modul Modul](#bolum-5-kod-rehberi---modul-modul)
- [Bolum 6: PointNet Mimarisi Detayli](#bolum-6-pointnet-mimarisi-detayli)
- [Bolum 7: Kurulum ve Calistirma](#bolum-7-kurulum-ve-calistirma)
- [Bolum 8: Pratik Uygulamalar ve Odevler](#bolum-8-pratik-uygulamalar-ve-odevler)
- [Bolum 9: Sikca Sorulan Sorular](#bolum-9-sikca-sorulan-sorular)
- [Bolum 10: Terimler Sozlugu](#bolum-10-terimler-sozlugu)

---

# Bolum 1: Temel Kavramlar

Bu bolumde, projeyi anlamak icin bilmeniz gereken temel bilgisayar bilimi kavramlarini sifirdan acikliyoruz.

## 1.1 Yapay Zeka (AI) Nedir?

Yapay zeka, bilgisayarlarin insanlar gibi "dusunmesini" saglayan teknolojilerin genel adidir.
Ancak gercekte bilgisayar dusunmez - **veriden oruntuler bulur**.

Gunluk hayattan bir ornek dusunun: Bir cocuk yuzlerce kedi ve kopek fotografina bakarak kedileri kopeklerden ayirt etmeyi ogrenir. Yapay zeka da ayni mantikla calisir: binlerce ornege bakarak "kurallari" kendi ogrenir.

```
                        YAPAY ZEKA (AI)
                             |
                    ┌────────┴────────┐
                    |                 |
              Kural Tabanli     Makine Ogrenmesi (ML)
              (If-else ile      (Veriden ogrenen)
               programlama)          |
                              ┌──────┴──────┐
                              |             |
                        Klasik ML     Derin Ogrenme (DL)
                        (Karar agaci,  (Sinir aglari)
                         SVM, vb.)         |
                                     ┌─────┴─────┐
                                     |           |
                                   CNN        PointNet
                                (Resimler)   (3B Noktalar)
                                              ↑
                                          BU PROJEDE
                                       KULLANILAN YONTEM
```

### Makine Ogrenmesi (ML) vs Derin Ogrenme (DL)

| Ozellik | Makine Ogrenmesi | Derin Ogrenme |
|---------|-----------------|---------------|
| Veri ihtiyaci | Az-orta | Cok (binlerce ornek) |
| Ozellik cikarimi | Insan tanimlar | Model kendi ogrenir |
| Hesaplama gucu | CPU yeterli | GPU gerekebilir |
| Ornek yontem | Karar agaci, SVM | PointNet, CNN |
| Bu projede | Registration icin | Dis segmentasyonu icin |

## 1.2 3B Mesh Nedir?

Gunluk hayatta gordugumuz nesneler **3 boyutludur** (genislik, yukseklik, derinlik). Bilgisayarda bu 3B nesneleri temsil etmenin bir yolu **mesh** (orfemek: aga) yapisidir.

Mesh, uc temel bilesenden olusur:

### Vertex (Kose Noktasi)

Uzayda bir noktadir. X, Y, Z koordinatlari ile tanimlanir.

```
Ornek: Bir vertex (kose noktasi)

         Y (yukarisagi)
         |
         |     * (3.0, 4.0, 2.0)  <-- Bu bir vertex
         |    /
         |   /
         |  /
         | /
         +───────────── X (saga)
        /
       /
      Z (one dogru)

Koordinat anlamlari:
  X = 3.0 mm  (saga dogru)
  Y = 4.0 mm  (yukariya dogru)
  Z = 2.0 mm  (one dogru)
```

### Face (Yuz / Ucgen)

3 vertex'i birlestiren ucgen yuzeydir. Bir mesh, binlerce ucgenden olusur.

```
Ornek: Bir face (ucgen yuz)

        V1 (1,4,0)
        /\
       /  \
      /    \         Bu ucgen bir "face" dir.
     /      \        3 vertex'i birlestirir:
    /________\       V1, V2, V3
  V2 (0,0,0)  V3 (3,0,0)
```

### Normal (Yuzey Normali)

Her ucgenin "hangi yone baktigini" gosteren ok (vektor)dur. Isik hesaplamalari ve mesafe olcumlerinde kullanilir.

```
Ornek: Bir ucgen ve normali

           ↑ Normal (yukariya bakiyor)
           |
     ______↑______
    |      |      |     Yuzey saga-sola uzaniyor
    |______|______|     Normal ise dik yukariya bakiyor
```

### Gercek Bir Dis Mesh'i

Bu projede, her agiz ici tarama yaklasik olarak:
- **~93.000 vertex** (kose noktasi)
- **~186.000 face** (ucgen yuz)

icermektedir. Boyutlari yaklasik 80 x 58 x 40 mm'dir (gercek bir cene boyutu).

```
Basitlestirilmis dis mesh gosterimi:

  ~93.000 nokta birlestirildiginde bir cene modeli olusur:

         _______________
        /  __|__|__|__  \        <- Disler (ucgenlerden olusan yuzeylier)
       /  |  |  |  |  |  \
      |   |  |  |  |  |   |
       \  |__|__|__|__|  /       <- Diseti (gingiva)
        \_____________/

  Her kucuk ucgen bir "face",
  her kose noktasi bir "vertex" dir.
```

## 1.3 Nokta Bulutu (Point Cloud) Nedir?

Nokta bulutu, uzayda dagilmis 3B noktalarin kumesidir. Mesh'ten farki: **noktalar arasinda baglanti (face) yoktur**.

```
Mesh (baglantilar var):          Nokta Bulutu (sadece noktalar):

    *---*---*                        *   *   *
    |\ /|\ /|                        *  * *  *
    | * | * |                        * *   * *
    |/ \|/ \|                        *  * *  *
    *---*---*                        *   *   *
```

Bu projede mesh yuzeyinden noktalar ornekleyerek nokta bulutu olusturuyoruz. Neden?
- PointNet modeli **nokta bulutu** uzerinde calisir
- Mesh dogrudan islenemez (degisen face sayilari nedeniyle)
- Her taramadan **24.000 nokta** ornekliyoruz

## 1.4 Sinir Agi (Neural Network) Temelleri

Sinir agi, insan beyninden esinlenen bir hesaplama modelidir. Temel yapisi:

```
NÖRON (en kucuk birim):

  Girisler        Agirliklar         Cikis
  ─────────       ──────────       ────────

  x1 ─── w1 ───┐
                │
  x2 ─── w2 ───┼──→ [ toplam + bias ] ──→ [ aktivasyon ] ──→ y
                │
  x3 ─── w3 ───┘

  y = aktivasyon(w1*x1 + w2*x2 + w3*x3 + bias)

  Basit aciklama:
  - Girisler (x): Veriler (ornegin bir noktanin x,y,z koordinatlari)
  - Agirliklar (w): Modelin "ogrendigi" degerler
  - Bias: Esik degeri
  - Aktivasyon: Dogrusal olmayan donusum (ornegin ReLU)
```

Birden fazla noron **katman** (layer) olusturur:

```
SINIR AGI KATMANLARI:

  Giris Katmani      Gizli Katmanlar        Cikis Katmani
  (Verilerimiz)      (Oruntuleri ogrenir)    (Tahminler)

    x1  ──────────→  O ──→ O ──→ O  ──────→  y1 (dis 11?)
    x2  ──────────→  O ──→ O ──→ O  ──────→  y2 (dis 12?)
    x3  ──────────→  O ──→ O ──→ O  ──────→  y3 (dis 13?)
                     O ──→ O ──→ O           ...
                     O ──→ O ──→ O  ──────→  y33 (diseti?)

  Bu projede:
  - Giris: 3 deger (x, y, z koordinati)
  - Cikis: 33 deger (her sinif icin olasilik)
  - Gizli katmanlar: Yuzlerce noron
```

### Egitim Nasil Calisir?

```
EGITIM DONGUSU:

  ┌──────────────────────────────────────────────────┐
  |  1. Veriyi modele gonder                          |
  |     Giris: 24.000 nokta (x,y,z)                  |
  |                                                   |
  |  2. Model tahmin yapar                            |
  |     Cikis: 24.000 nokta icin sinif tahmini        |
  |                                                   |
  |  3. Kayip (Loss) hesapla                          |
  |     Tahmin vs Gercek etiket farki ne kadar?       |
  |                                                   |
  |  4. Geri yayilim (Backpropagation)                |
  |     Hangi agirliklar hataya neden oldu?            |
  |                                                   |
  |  5. Agirlik guncelleme (Optimizer)                |
  |     Hatayi azaltacak sekilde agirliklari degistir  |
  |                                                   |
  └──────────────┬───────────────────────────────────┘
                 |
                 ↓ Bu dongu 50-100 kez tekrarlanir (epoch)
                 |
  Sonuc: Model disleri tanimlamayi ogrenir!
```

## 1.5 Segmentasyon Nedir?

Segmentasyon, bir verinin (resim, 3B model vb.) anlamli parcalara bolunmesidir.

```
2B RESIM SEGMENTASYONU (kolay anlasilir ornek):

  Orijinal Resim:          Segmentasyon Sonucu:
  ┌──────────────┐         ┌──────────────┐
  |  ###  @@@@   |         |  AAA  BBBB   |    A = Kedi
  |  ###  @@@@   |   -->   |  AAA  BBBB   |    B = Kopek
  |  ###  @@@@   |         |  AAA  BBBB   |    C = Arka plan
  |              |         |  CCCCCCCCCC   |
  └──────────────┘         └──────────────┘

3B DIS SEGMENTASYONU (bu projede yapilan):

  Orijinal Mesh:            Segmentasyon Sonucu:
  ┌──────────────┐          ┌──────────────┐
  | .............|          | GGGGGGGGGGGGG|    G = Diseti (Gingiva)
  | .||.||.||.||.|   -->    | G11G12G13G14G|    11 = Ust sag santral
  | .||.||.||.||.|          | G11G12G13G14G|    12 = Ust sag lateral
  | .............|          | GGGGGGGGGGGGG|    13 = Ust sag kanin
  └──────────────┘          └──────────────┘    14 = Ust sag 1.premolar

  Her vertex'e (93.000 nokta) bir etiket atanir:
  0 = diseti, 11 = ust sag santral kesici, ... 48 = alt sag 3. molar
```

## 1.6 Registration (Cakistirma) Nedir?

Registration, iki farkli taramayi **ayni koordinat sistemine** getirme islemidir.

Neden gerekli? Hasta farkli zamanlarda tarandiginda, agiz tarayicisinin konumu degisir. Bu yuzden iki tarama farkli pozisyonda olur.

```
REGISTRATION ONCESI:                 REGISTRATION SONRASI:

  Tarama 1 (T0):     Tarama 2 (T1):        Her iki tarama ustuste:

    ____                    ____                 ____
   /    \              ____/    \            ____/    \
  |      |            /    |    |           /    |    |
  |      |           |     |    |          | T0  | T1 |
   \____/            |      \___/          |  +  |  + |
                      \____/                \____\___/

  --> Iki tarama farkli        --> Taramalar ayni konuma
      yerde duruyor                getirildi, artik
                                   degisimler olculebilir!
```

**Bu projede iki registration yontemi kullanilir:**

1. **RANSAC** (kaba hizalama): Yaklasik olarak ustuste getir
2. **ICP** (ince hizalama): Hassas ayarlama yap

---

# Bolum 2: Dis Hekimligi ve Yapay Zeka

## 2.1 Intraoral Tarama Nedir?

Intraoral tarama, agiz ici 3B tarayici (ornegin 3Shape TRIOS) ile hastanin dislerinin dijital modelinin olusturulmasi islemidir.

```
TARAMA SURECI:

  1. Tarayici agiza sokulur
     ┌────────────┐
     | ░░ TARAYICI |=====> agiz icinde gezdirme
     └────────────┘

  2. Binlerce goruntu birlestirilir
     [ foto 1 ] + [ foto 2 ] + [ foto 3 ] + ...

  3. 3B mesh modeli olusur
     ┌─────────────────────┐
     |    _______________   |
     |   /  __|__|__|__  \  |   OBJ dosyasi
     |  /  |  |  |  |  |  \ |   (~93.000 vertex)
     |  \  |__|__|__|__|  / |   (~186.000 face)
     |   \_____________/   |
     └─────────────────────┘

  4. Dosya olarak kaydedilir
     hasta_alt_cene.obj  (3B model)
     hasta_alt_cene.json (dis etiketleri)
```

## 2.2 FDI Dis Numaralama Sistemi

FDI (Federation Dentaire Internationale) sistemi, her disi 2 basamakli bir numara ile tanimlar.

**Birinci basamak: Kadran (ceyrek)**
```
         UST CENE (hastanin bakis acisi)
    ┌──────────────────────────────────┐
    |  Sag taraf (1)  | Sol taraf (2)  |    1. kadran: Ust sag
    |    18 17 16 15  |  25 26 27 28   |    2. kadran: Ust sol
    |    14 13 12 11  |  21 22 23 24   |
    ├─────────────────┼────────────────┤
    |    44 43 42 41  |  31 32 33 34   |    3. kadran: Alt sol
    |    48 47 46 45  |  35 36 37 38   |    4. kadran: Alt sag
    |  Sag taraf (4)  | Sol taraf (3)  |
    └──────────────────────────────────┘
         ALT CENE (hastanin bakis acisi)
```

**Ikinci basamak: Dis sirasi (1-8)**
```
  1 = Santral kesici    (on dis)
  2 = Lateral kesici    (on dis yani)
  3 = Kanin             (kopek disi)
  4 = 1. premolar       (kucuk azisi)
  5 = 2. premolar       (kucuk azisi)
  6 = 1. molar          (buyuk azisi)
  7 = 2. molar          (buyuk azisi)
  8 = 3. molar          (akil disi)
```

**Ornek:**
- `11` = Ust sag santral kesici (1. kadran, 1. dis)
- `36` = Alt sol 1. molar (3. kadran, 6. dis)
- `0` = Diseti (gingiva - dis degil)

**Bu projede 33 sinif var:** 0 (diseti) + 32 dis (11-18, 21-28, 31-38, 41-48)

## 2.3 Bu Projede AI Ne Icin Kullaniliyor?

```
PROJE HEDEFLERI:

  ┌─────────────────────────────────────────────────────────┐
  |                                                         |
  |  1. OTOMATIK DIS SEGMENTASYONU                          |
  |     "Bu nokta hangi dise ait?"                          |
  |     93.000 vertex -> her birine bir dis numarasi ata    |
  |                                                         |
  |  2. TEMPORAL DEGISIM ANALIZI                            |
  |     "3 ayda ne degisti?"                                |
  |     T0 tarama vs T1 tarama -> fark haritasi             |
  |                                                         |
  |  3. HASTALIK TESPITI                                    |
  |     "Curuk var mi? Asinma var mi?"                      |
  |     Yuzey degisimlerinden patoloji tahmini              |
  |                                                         |
  |  4. RISK SKORLAMA                                       |
  |     "En riskli dis hangisi?"                            |
  |     0-100 arasi dis bazinda risk degerlendirmesi        |
  |                                                         |
  └─────────────────────────────────────────────────────────┘
```

---

# Bolum 3: Teeth3DS+ Veri Seti

## 3.1 Veri Seti Nedir?

Bir AI modelini egitmek icin **etiketli veriye** ihtiyac vardir. Bu projedeki veri seti **Teeth3DS+** olarak adlandirilir ve MICCAI (Medical Image Computing and Computer Assisted Intervention) konferansi kapsaminda uzman dis hekimleri tarafindan olusturulmustur.

```
VERI SETI OZETI:

  ┌──────────────────────────────────────────┐
  |  Teeth3DS+ Veri Seti                     |
  |                                          |
  |  Toplam mesh:   1.900 OBJ dosyasi        |
  |  Etiketli:      1.800 (JSON etiketli)    |
  |  Etiketsiz:       100 (test seti)        |
  |  Hasta sayisi:   ~950                    |
  |  Cene turu:      Alt + Ust (ayri ayri)   |
  |  Sinif sayisi:   33                      |
  |                                          |
  |  Her mesh:                               |
  |    ~93.000 vertex (kose noktasi)         |
  |   ~186.000 face (ucgen yuz)             |
  |    ~80 x 58 x 40 mm boyut              |
  └──────────────────────────────────────────┘
```

## 3.2 Veri Dosya Yapisi

Veriler 7 parcaya bolunmustur. Her parca alt ve ust cene klasorleri icerir:

```
inovens-dis/
├── data_part_1/                  # Part 1: 150 hasta
│   ├── lower/                    # Alt cene
│   │   ├── O52P1SZT/            # Hasta klasoru (ID = O52P1SZT)
│   │   │   ├── O52P1SZT_lower.obj    # 3B mesh dosyasi
│   │   │   └── O52P1SZT_lower.json   # Segmentasyon etiketleri
│   │   ├── 01KKD7YK/            # Baska bir hasta
│   │   │   ├── 01KKD7YK_lower.obj
│   │   │   └── 01KKD7YK_lower.json
│   │   └── ... (148 hasta daha)
│   └── upper/                    # Ust cene (ayni yapi)
│       └── ...
│
├── data_part_2/                  # Part 2: 150 hasta
├── data_part_3/                  # Part 3: 150 hasta
├── data_part_4/                  # Part 4: 150 hasta
├── data_part_5/                  # Part 5: 150 hasta
├── data_part_6/                  # Part 6: 150 hasta
│
└── data_part_7/                  # Part 7: 50 hasta (ETIKETSIZ - test seti)
    ├── lower/
    │   └── XYZABC12/
    │       └── XYZABC12_lower.obj    # Sadece OBJ (JSON yok!)
    └── upper/
```

**Ozet tablo:**

| Part | Hasta sayisi | Cene | OBJ | JSON | Not |
|------|-------------|------|-----|------|-----|
| data_part_1 | 150 | Alt + Ust | 300 | 300 | Egitim |
| data_part_2 | 150 | Alt + Ust | 300 | 300 | Egitim |
| data_part_3 | 150 | Alt + Ust | 300 | 300 | Egitim |
| data_part_4 | 150 | Alt + Ust | 300 | 300 | Egitim |
| data_part_5 | 150 | Alt + Ust | 300 | 300 | Egitim |
| data_part_6 | 150 | Alt + Ust | 300 | 300 | Egitim |
| data_part_7 | 50  | Alt + Ust | 100 | **0** | **Test (etiketsiz)** |
| **Toplam** | **~950** | | **1900** | **1800** | |

## 3.3 OBJ Dosyasi Nedir? (Satir Satir Aciklama)

OBJ (Wavefront OBJ), 3B modelleri saklamak icin kullanilan basit bir metin dosyasidir.

```
# Bu bir OBJ dosyasinin icindeki satirlar:

v 0.846 21.803 -115.433 0.502 0.502 0.502     # 1. satir
v 1.045 23.123 -107.028 0.502 0.502 0.502     # 2. satir
v -2.123 19.456 -110.789 0.502 0.502 0.502    # 3. satir
...                                            # (~93.000 vertex satiri)
f 1 2 3                                        # face satiri
f 2 3 4                                        # face satiri
...                                            # (~186.000 face satiri)
```

**Satir satir aciklama:**

```
v 0.846 21.803 -115.433 0.502 0.502 0.502
^   ^      ^       ^      ^     ^     ^
|   |      |       |      |     |     |
|   X      Y       Z     Red  Green  Blue
|
"v" = vertex (kose noktasi) anlamina gelir

Anlami:
  X =   0.846 mm  (saga dogru konum)
  Y =  21.803 mm  (yukariya dogru konum)
  Z = -115.433 mm (arkaya dogru konum)
  R = 0.502       (kirmizi renk degeri, 0-1 arasi)
  G = 0.502       (yesil renk degeri)
  B = 0.502       (mavi renk degeri)

Renk degerleri = (0.502, 0.502, 0.502) → gri renk
```

```
f 1 2 3
^  ^ ^ ^
|  | | |
|  1. vertex, 2. vertex, 3. vertex
|
"f" = face (ucgen yuz) anlamina gelir

Anlami:
  1. vertex + 2. vertex + 3. vertex birlestirilerek
  bir ucgen yuzey olusturulur.
```

## 3.4 JSON Etiket Dosyasi (Satir Satir Aciklama)

Her OBJ dosyasina karsilik gelen bir JSON etiketi dosyasi vardir. Bu dosya, **her vertex'in hangi dise ait oldugunu** belirtir.

```json
{
  "id_patient": "O52P1SZT",
  "jaw": "lower",
  "labels": [0, 0, 0, 31, 31, 31, 32, 32, 0, 0, ...],
  "instances": [0, 0, 0, 1, 1, 1, 2, 2, 0, 0, ...]
}
```

**Satir satir aciklama:**

```
"id_patient": "O52P1SZT"
  → Hastanin kimlik numarasi (anonim, gercek isim degil)

"jaw": "lower"
  → Cene turu: "lower" = alt cene, "upper" = ust cene

"labels": [0, 0, 0, 31, 31, 31, 32, 32, 0, 0, ...]
  → HER VERTEX icin FDI dis numarasi
  → Toplam 93.288 sayi (mesh'teki vertex sayisi kadar)
  → 0 = diseti, 31 = alt sol santral kesici, 32 = alt sol lateral kesici, ...

  Gorsel aciklama:

  OBJ'deki vertex'ler:     JSON'daki etiketler:
  v  0.846  21.803 -115.4  → labels[0] = 0    (diseti)
  v  1.045  23.123 -107.0  → labels[1] = 0    (diseti)
  v -2.123  19.456 -110.7  → labels[2] = 0    (diseti)
  v  5.678  12.345  -98.0  → labels[3] = 31   (alt sol santral kesici)
  v  5.890  12.567  -97.5  → labels[4] = 31   (alt sol santral kesici)
  v  6.123  12.789  -97.0  → labels[5] = 31   (alt sol santral kesici)
  v  8.456  11.234  -96.0  → labels[6] = 32   (alt sol lateral kesici)
  ...

"instances": [0, 0, 0, 1, 1, 1, 2, 2, 0, 0, ...]
  → Her dis icin benzersiz instance (ornek) numarasi
  → 0 = diseti, 1 = 1. dis, 2 = 2. dis, ...
  → Ayni dis numarasina sahip olsa bile farkli disleri ayirt eder
```

## 3.5 Vertex-Seviye Segmentasyon Ne Demek?

Her vertex'e bir etiket atanir. Bu "vertex-seviye segmentasyon" olarak adlandirilir.

```
VERTEX SEVIYE SEGMENTASYON GORSELI:

Mesh'teki her ucgenin koselerinin etiketi var:

       V1 (etiket=31)              V4 (etiket=32)
       /\                          /\
      /  \    31 numarali dis     /  \    32 numarali dis
     / 31 \                      / 32 \
    /______\                    /______\
   V2 (31) V3 (31)            V5 (32)  V6 (32)


Tum mesh icin:

  Diseti       Dis 31      Dis 32      Dis 33      Diseti
  (etiket=0)   (etiket=31) (etiket=32) (etiket=33) (etiket=0)

  0 0 0 0 0 | 31 31 31 | 32 32 32 | 33 33 33 | 0 0 0 0 0
  0 0 0 0 0 | 31 31 31 | 32 32 32 | 33 33 33 | 0 0 0 0 0
  0 0 0 0 0 | 31 31 31 | 32 32 32 | 33 33 33 | 0 0 0 0 0

  93.288 vertex = 93.288 etiket (bire bir eslesme)
```

---

# Bolum 4: Proje Mimarisi

## 4.1 Buyuk Resim: Tum Pipeline

Bu proje, birden fazla modulu bir araya getiren kapsamli bir pipeline'dir:

```
┌─────────────────────────────────────────────────────────────────────┐
|                         DENTAL AI PIPELINE                          |
|                                                                     |
|  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               |
|  | OBJ+JSON |    | Veri Yukleme |    | Mesh Isleme  |               |
|  | Dosyalar |───→| data_loader  |───→| mesh_proc    |               |
|  └──────────┘    └──────┬───────┘    └──────┬───────┘               |
|                         |                    |                       |
|                         v                    v                       |
|              ┌──────────────────────────────────────┐               |
|              |        ESAS AI ISLEMLERI              |               |
|              |                                       |               |
|              |  ┌──────────────┐ ┌───────────────┐  |               |
|              |  | Registration | | Segmentasyon  |  |               |
|              |  | (Cakistirma) | | (PointNet)    |  |               |
|              |  | RANSAC + ICP | | 33 sinif      |  |               |
|              |  └──────┬───────┘ └──────┬────────┘  |               |
|              |         |                |            |               |
|              |         v                v            |               |
|              |  ┌──────────────────────────────┐    |               |
|              |  |      Degisim Analizi          |    |               |
|              |  |   Mesafe + Risk Skorlama      |    |               |
|              |  └──────────────┬────────────────┘    |               |
|              └────────────────┼───────────────────┘               |
|                               |                                     |
|                               v                                     |
|              ┌────────────────────────────────┐                     |
|              |      Gorsellestirme             |                     |
|              |  3B Plotly + Streamlit Dashboard |                     |
|              └────────────────────────────────┘                     |
└─────────────────────────────────────────────────────────────────────┘
```

## 4.2 Her Modulun Rolu

```
┌─────────────────────────────────────────────────────────────┐
|  DOSYA                   | NE YAPAR?                        |
├─────────────────────────────────────────────────────────────┤
|  config.py               | Tum ayarlar ve sabitler          |
|                          | FDI numaralari, egitim parametr. |
├─────────────────────────────────────────────────────────────┤
|  src/data_loader.py      | OBJ + JSON dosyalarini yukler    |
|                          | DentalMesh ve Dataset siniflari  |
├─────────────────────────────────────────────────────────────┤
|  src/mesh_processing.py  | Mesh temizleme, normalizasyon    |
|                          | Open3D ile 3B isleme             |
├─────────────────────────────────────────────────────────────┤
|  src/registration.py     | Iki taramayi hizalama            |
|                          | RANSAC (kaba) + ICP (ince)       |
├─────────────────────────────────────────────────────────────┤
|  src/segmentation.py     | PointNet sinir agi modeli        |
|                          | Her nokta icin dis tahmini       |
├─────────────────────────────────────────────────────────────┤
|  src/change_analysis.py  | Yuzey degisimi hesaplama         |
|                          | Mesafe + risk skoru uretme       |
├─────────────────────────────────────────────────────────────┤
|  src/visualization.py    | 3B grafikler (Plotly)            |
|                          | Mesafe haritalari, risk grafigi  |
├─────────────────────────────────────────────────────────────┤
|  train.py                | Model egitim scripti             |
|                          | Komut satirindan calistirma      |
├─────────────────────────────────────────────────────────────┤
|  app.py                  | Streamlit web dashboard          |
|                          | 5 sayfali interaktif arayuz      |
├─────────────────────────────────────────────────────────────┤
|  run_demo.py             | Tam pipeline demo                |
|                          | Tum modulleri test eder          |
├─────────────────────────────────────────────────────────────┤
|  tests/test_pipeline.py  | Otomatik testler                 |
|                          | 7 test modulu                    |
└─────────────────────────────────────────────────────────────┘
```

## 4.3 Modullerin Birbirine Baglantisi

```
Hangi modul hangisini kullaniyor?

  config.py  <───────────────────── (herkes config'i kullanir)
      |
      v
  data_loader.py ──────────┐
      |                    |
      |    mesh_processing.py ───→ registration.py
      |         |                       |
      v         v                       v
  segmentation.py              change_analysis.py
      |                              |
      v                              v
  train.py                   visualization.py
      |                              |
      └──────────┐    ┌──────────────┘
                 v    v
               app.py (Dashboard)
               run_demo.py (Demo)
```

---

# Bolum 5: Kod Rehberi - Modul Modul

Bu bolumde her modulu detayli inceliyoruz. Kodlari satir satir Turkce yorumlarla acikliyoruz.

## 5.1 config.py - Yapilandirma Dosyasi

**Ne yapar?** Tum proje icin ortak ayarlar ve sabitler burada tanimlanir.

```python
# ── Proje dizini ──
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
# Path(__file__) = bu dosyanin yolu (ornegin /Users/.../inovens-dis/config.py)
# .parent = bir ust dizin (ornegin /Users/.../inovens-dis/)

# ── Veri klasorleri ──
DATA_PARTS = [PROJECT_ROOT / f"data_part_{i}" for i in range(1, 8)]
# Sonuc: [Path(".../data_part_1"), Path(".../data_part_2"), ..., Path(".../data_part_7")]

# ── FDI dis isimleri ──
FDI_TOOTH_NAMES = {
    0:  "Diseti (Gingiva)",        # Dis degil, diseti
    11: "Ust sag santral kesici",  # 1. kadran, 1. dis
    12: "Ust sag lateral kesici",  # 1. kadran, 2. dis
    # ... (32 dis daha, toplam 33 sinif)
    48: "Alt sag 3. molar",       # 4. kadran, 8. dis
}

# ── FDI -> Ardisik Indeks Donusumu ──
ALL_FDI_LABELS = sorted(FDI_TOOTH_NAMES.keys())
# Sonuc: [0, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, ..., 48]
# Toplam 33 eleman

FDI_TO_INDEX = {fdi: idx for idx, fdi in enumerate(ALL_FDI_LABELS)}
# Sonuc: {0: 0, 11: 1, 12: 2, 13: 3, ..., 48: 32}
#
# NEDEN GEREKLI?
# FDI numaralari ardisik degil: 0, 11, 12, ..., 18, 21, 22, ..., 48
# Ama sinir agi 0'dan baslayan ardisik sayilar ister: 0, 1, 2, ..., 32
# Bu sozluk ceviriyi yapar:
#   FDI 0  → Indeks 0
#   FDI 11 → Indeks 1
#   FDI 12 → Indeks 2
#   ...
#   FDI 48 → Indeks 32

INDEX_TO_FDI = {idx: fdi for fdi, idx in FDI_TO_INDEX.items()}
# Sonuc: {0: 0, 1: 11, 2: 12, ..., 32: 48}  (tersi)

NUM_CLASSES = len(ALL_FDI_LABELS)  # 33
```

**Neden bu donusum gerekli?**
```
PROBLEM:  FDI numaralari ardisik degil

  FDI:     0  11  12  13  14 ... 18  21  22 ... 48
  Arada bosluklar var:  ↑   ↑        ↑
                       0→11 (10 bosluk)
                           18→21 (2 bosluk)

COZUM: Ardisik indekse cevirme

  FDI:     0   11   12   13  ...  48
  Indeks:  0    1    2    3  ...  32

  Sinir agi 0-32 arasi 33 sinif ile calisir
```

## 5.2 data_loader.py - Veri Yukleme

**Ne yapar?** OBJ mesh ve JSON etiket dosyalarini yukler, yonetir.

### DentalMesh Sinifi

Tek bir hastanin tek bir cene taramasini temsil eder.

```python
# DENTALMEESH: Tek bir tarama (ornegin O52P1SZT hastasinin alt cenesi)
#
# LAZY LOADING KAVRAMI:
# "Lazy" = tembel. Veriyi hemen yuklemez, sadece ihtiyac olunca yukler.
# Neden? 1900 mesh'in hepsini bellege yuklersek ~30 GB RAM gerekir!
#
# Calisma sekli:
#   scan = DentalMesh(...)     # Sadece dosya yollarini kaydeder (hizli)
#   scan.vertices              # ILK ERISIMDE OBJ dosyasini yukler
#   scan.labels                # ILK ERISIMDE JSON dosyasini yukler
#   scan.unload()              # Bellegi serbest birakir

class DentalMesh:
    patient_id: str            # Hasta kimlik numarasi (ornegin "O52P1SZT")
    jaw_type: str              # "upper" veya "lower"
    obj_path: Path             # OBJ dosya yolu
    json_path: Optional[Path]  # JSON yolu (test setinde None olabilir)

    # ── OZELLIKLER (Properties) ──
    #
    # @property: Python'da bir metodu ozellik gibi kullanmayi saglar
    # scan.vertices yazdiqimizda asagidaki fonksiyon calisir

    @property
    def vertices(self):
        if self._vertices is None:   # Daha once yuklenmemisse
            self.load_mesh()         # OBJ'yi yukle
        return self._vertices        # (93288, 3) numpy array dondur

    @property
    def has_labels(self):
        # JSON dosyasi var mi kontrol et
        return self.json_path is not None and self.json_path.exists()

    # ── MESH YUKLEME ──
    def load_mesh(self):
        # trimesh kutuphanesi ile OBJ dosyasini oku
        mesh = trimesh.load(str(self.obj_path), process=False)
        self._vertices = np.array(mesh.vertices, dtype=np.float32)  # (93288, 3)
        self._faces = np.array(mesh.faces, dtype=np.int64)          # (186499, 3)
        return mesh

    # ── ETIKET YUKLEME ──
    def load_labels(self):
        with open(self.json_path, "r") as f:
            data = json.load(f)
        self._labels = np.array(data["labels"], dtype=np.int32)     # (93288,)
        self._instances = np.array(data["instances"], dtype=np.int32)

    # ── TEKIL DIS VERTEX'LERINI AL ──
    def get_tooth_vertices(self, fdi_number):
        mask = self.labels == fdi_number  # Boolean maske: [False, False, True, True, ...]
        return self.vertices[mask]        # Sadece o dise ait vertex'ler
        #
        # Ornek: fdi_number = 31 (alt sol santral kesici)
        # labels = [0, 0, 31, 31, 31, 32, 32, 0, ...]
        # mask   = [F, F,  T,  T,  T,  F,  F, F, ...]
        # Sonuc: 3 vertex'in (x,y,z) koordinatlari

    # ── NOKTA ORNEKLEME ──
    def sample_points(self, num_points=24000):
        # Mesh yuzeyinden rastgele 24.000 nokta ornekle
        # 1. Ucgen yuzeylerde rastgele nokta sec
        # 2. O noktadaki normali hesapla
        # 3. En yakin vertex'in etiketini aktar

        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

        # Etiket aktarimi:
        # Orneklenen nokta bir ucgenin icinde.
        # O ucgenin kose vertex'lerinin etiketini kullan.
        face_verts = self.faces[face_indices]   # Hangi ucgenlerden orneklendi
        labels = self.labels[face_verts[:, 0]]  # Ilk vertex'in etiketini al

        return {"points": points, "normals": normals, "labels": labels}
```

**Gorsel aciklama - sample_points nasil calisir:**
```
MESH YUZEYINDEN NOKTA ORNEKLEME:

  Mesh (ucgenlerden olusur):

      V1──────V2──────V3
      |\ Face1 |\ Face2 |
      | \      | \      |
      |  \     |  \     |
      |   \    |   \    |
      | F3 \   | F4 \   |
      V4──────V5──────V6

  Ornekleme:

      V1──*───V2──*───V3     * = Yuzeyden orneklenen noktalar
      |\ * *  |\ *  * |     (rastgele pozisyonlarda)
      | \  *  | \  *   |
      | *\    |  \*    |
      |   \*  |   \ *  |
      V4──────V5──────V6

  Her orneklenen nokta icin:
  - Hangi ucgenden geldi? → face_indices
  - O ucgenin etiketini ver → labels
```

### Teeth3DSDataset Sinifi

Tum data_part klasorlerini yonetir.

```python
# TEETH3DSDATASET: Tum veri setini yonetir

class Teeth3DSDataset:
    def __init__(self, data_root, parts=[1,2,3,4,5,6,7]):
        self.scans = []      # Tum taramalarin listesi
        self._build_index()  # Dizinleri tara, taramalari bul

    def _build_index(self):
        # Her data_part klasorunu tara
        for part_num in self.parts:
            part_dir = self.data_root / f"data_part_{part_num}"
            for jaw_type in ["lower", "upper"]:
                jaw_dir = part_dir / jaw_type
                for patient_dir in jaw_dir.iterdir():
                    # OBJ dosyasini bul
                    obj_files = list(patient_dir.glob("*.obj"))
                    json_files = list(patient_dir.glob("*.json"))

                    # DentalMesh olustur (ama mesh YUKLEMEZ - sadece yol kaydeder)
                    self.scans.append(DentalMesh(
                        patient_id=patient_dir.name,
                        jaw_type=jaw_type,
                        obj_path=obj_files[0],
                        json_path=json_files[0] if json_files else None,
                    ))
        # Sonuc: self.scans listesinde ~1900 DentalMesh nesnesi
        # Ama hicbirinin mesh'i yuklenmedi (lazy loading!)

    def split_train_val(self, val_ratio=0.15):
        # HASTA BAZLI ayirma (onemli!)
        # Ayni hastanin alt ve ust cenesi ayni sete gitmeli
        # Aksi halde "veri sizintisi" (data leakage) olur
        #
        # Yanlis: Rastgele ayirmak
        #   Egitim: O52P1SZT_lower, O52P1SZT_upper  ← Ayni hasta!
        #   Test:   O52P1SZT_lower                    ← Ayni hasta!
        #   Problem: Model hastayi "tanir", gercek performans olculemez
        #
        # Dogru: Hasta bazli ayirmak
        #   Egitim: O52P1SZT_lower, O52P1SZT_upper  ← Hasta A
        #   Test:   XYZABC12_lower, XYZABC12_upper   ← Hasta B (farkli!)

        patient_ids = sorted(set(s.patient_id for s in labeled_scans))
        # %85 egitim, %15 validasyon
        n_val = int(len(patient_ids) * val_ratio)
        val_ids = set(patient_ids[:n_val])    # Ilk %15 hasta validasyon
        train_ids = set(patient_ids[n_val:])  # Kalan %85 hasta egitim
```

## 5.3 registration.py - 3B Cakistirma

**Ne yapar?** Iki farkli zamanda alinan taramalari ayni koordinat sistemine getirir.

### Registration Akis Semasi

```
REGISTRATION PIPELINE (Adim Adim):

  GIRIS: Iki mesh (farkli zamanlarda alinmis)
  ────────────────────────────────────────────

  Mesh T0 (Onceki)              Mesh T1 (Sonraki)
  ┌─────────────┐               ┌─────────────┐
  |  3B model   |               |  3B model   |
  └──────┬──────┘               └──────┬──────┘
         |                              |
         v                              v
  ┌─────────────┐               ┌─────────────┐
  | 50.000 nokta|               | 50.000 nokta|   ← Mesh'ten nokta ornekle
  | ornekle     |               | ornekle     |
  └──────┬──────┘               └──────┬──────┘
         |                              |
         v                              v
  ┌─────────────┐               ┌─────────────┐
  | Normal      |               | Normal      |   ← Her noktanin yuzey
  | hesapla     |               | hesapla     |      yonunu hesapla
  └──────┬──────┘               └──────┬──────┘
         |                              |
         v                              v
  ┌─────────────┐               ┌─────────────┐
  | FPFH        |               | FPFH        |   ← Her noktanin
  | ozellik     |               | ozellik     |      "parmak izi"ni
  | cikar       |               | cikar       |      hesapla
  └──────┬──────┘               └──────┬──────┘
         |                              |
         └───────────┬──────────────────┘
                     |
                     v
  ┌───────────────────────────────────┐
  |    ADIM 1: RANSAC KABA HIZALAMA   |
  |                                    |
  |  1. Rastgele 3 nokta cifti sec     |
  |  2. FPFH ile eslestir              |
  |  3. Donusum matrisi hesapla        |
  |  4. Ne kadar nokta uyusuyor say    |
  |  5. 100.000 kez tekrarla           |
  |  6. En iyi sonucu sec              |
  |                                    |
  |  Cikti: Yaklasik 4x4 matris       |
  └────────────────┬──────────────────┘
                   |
                   v
  ┌───────────────────────────────────┐
  |    ADIM 2: ICP INCE HIZALAMA      |
  |                                    |
  |  1. RANSAC sonucunu baslangic al   |
  |  2. Her noktanin en yakinin bul    |
  |  3. Mesafeyi minimize eden         |
  |     donusumu hesapla               |
  |  4. 200 kez tekrarla               |
  |  5. Yakinsamayi kontrol et         |
  |                                    |
  |  Cikti: Hassas 4x4 matris         |
  └────────────────┬──────────────────┘
                   |
                   v
  ┌───────────────────────────────────┐
  |         SONUC                      |
  |                                    |
  |  transformation: 4x4 matris       |
  |  fitness: 0.95 (uyum orani)        |
  |  inlier_rmse: 0.12 mm (hata)      |
  └───────────────────────────────────┘
```

### 4x4 Donusum Matrisi Nedir?

```
Donusum matrisi: 3B uzayda bir nesneyi dondurme + kaydirma

  ┌                    ┐
  | R11  R12  R13  Tx  |     R = 3x3 Rotasyon matrisi (dondurme)
  | R21  R22  R23  Ty  |     T = 3x1 Oteleme vektoru (kaydirma)
  | R31  R32  R33  Tz  |
  |  0    0    0    1  |     Son satir her zaman [0, 0, 0, 1]
  └                    ┘

  Kullanim:
  yeni_nokta = Matris x eski_nokta

  ┌    ┐     ┌                ┐   ┌    ┐
  | x' |     | R11 R12 R13 Tx |   | x  |
  | y' |  =  | R21 R22 R23 Ty | x | y  |
  | z' |     | R31 R32 R33 Tz |   | z  |
  | 1  |     |  0   0   0  1  |   | 1  |
  └    ┘     └                ┘   └    ┘
```

### FPFH Nedir?

FPFH (Fast Point Feature Histogram), her noktanin "kimligini" (parmak izini) tanimlayan bir ozelliktir.

```
FPFH OZELLIK HESABI:

  1. Bir nokta sec (P)
  2. Yakinindaki noktalari bul (daire icindekiler)
  3. P ile komsulari arasindaki aci farklarini hesapla
  4. Bu farklardan histogram (ozellik vektoru) olustur

       ○  ○
     ○  P  ○        P'nin etrafindaki noktalarla
       ○  ○         acilari hesapla → histogram

  Sonuc: 33 boyutlu bir vektor
  [0.2, 0.1, 0.05, 0.3, ...]

  Bu vektor P noktasinin "parmak izi" gibidir.
  Ayni sekildeki yuzeyler benzer FPFH'ye sahip olur.

  RANSAC bu FPFH'leri eslestirerek iki mesh'teki
  ayni noktalari bulur.
```

## 5.4 change_analysis.py - Degisim Analizi

**Ne yapar?** Iki cakistirilmis tarama arasindaki yuzey degisimlerini olcer.

### Mesafe Hesaplama Yontemleri

```
1. NOKTADAN NOKTAYA MESAFE (Point-to-Point):

   T0 Tarama:    T1 Tarama:

   *  *  *       *  *  *
   *  *  *   →   *  *   *     ← Bu nokta hareket etmis!
   *  *  *       *  *  *

   Her T0 noktasi icin, T1'deki EN YAKIN noktayi bul.
   Aralarindaki mesafeyi hesapla.

   KDTree algoritmasini kullanir (hizli en yakin komsu arama).


2. ISARETLI MESAFE (Signed Distance):

   Negatif = Madde KAYBI (curuk, asinma)
   Pozitif = Madde BIRIKIMI (distarisi, kalkulus)

                Normal yonu
                    ↑
         T0 ████████|████████ T0 yuzeyi
                    |
         T1 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ T1 yuzeyi
                    |
              ←─── mesafe ──→

   Eger T1 normal yonunde ise → pozitif (madde birikti)
   Eger T1 normal tersinde ise → negatif (madde kaybi)


3. HAUSDORFF MESAFESI:

   "En kotu durum" mesafesidir.
   Iki yuzey arasindaki maksimum sapmay olcer.

   Formula:
   H(A,B) = max( max_a(d(a,B)), max_b(d(b,A)) )

   Yani: "Herhangi bir noktanin diger yuzeyden ne kadar
          uzak olabileceginin en kotu senaryosu"
```

### Risk Skorlama Sistemi

```
RISK SKORU HESAPLAMA (0-100):

  Her dis icin 4 faktor degerlendirilir:

  ┌────────────────────────────────────────────────┐
  |  Faktor              | Agirlik | Aralik        |
  ├────────────────────────────────────────────────┤
  |  Ortalama mesafe     |   %30   | 0-2mm → 0-100|
  |  Maksimum mesafe     |   %20   | 0-5mm → 0-100|
  |  Madde kaybi orani   |   %30   | 0-1   → 0-100|
  |  Mesafe varyans      |   %20   | 0-1mm → 0-100|
  └────────────────────────────────────────────────┘

  Toplam Skor = (0.3 x ort_mesafe_norm + 0.2 x maks_mesafe_norm
                + 0.3 x madde_kaybi + 0.2 x varyans_norm) x 100

  RISK SEVIYELERI:
  ┌─────────────────────────────────────┐
  |  Skor     | Seviye  | Renk         |
  ├─────────────────────────────────────┤
  |   0 - 20  | Dusuk   | Yesil ████   |
  |  20 - 50  | Orta    | Turuncu ████ |
  |  50 - 75  | Yuksek  | Kirmizi ████ |
  |  75 - 100 | Kritik  | Mor     ████ |
  └─────────────────────────────────────┘
```

## 5.5 segmentation.py - Dis Segmentasyonu

Bu modulun detayli aciklamasi Bolum 6'da yapilmaktadir. Burada kisa bir ozet:

**Ne yapar?** PointNet sinir agi kullanarak her 3B noktanin hangi dise ait oldugunu tahmin eder.

```
SEGMENTASYON ISLEMI:

  Giris: 24.000 nokta (x,y,z)      Cikis: 24.000 etiket

  Nokta 1:  (1.2, 3.4, -5.6)  →  Sinif 31 (alt sol santral kesici)
  Nokta 2:  (1.3, 3.5, -5.7)  →  Sinif 31 (alt sol santral kesici)
  Nokta 3:  (5.6, 7.8, -9.0)  →  Sinif 36 (alt sol 1. molar)
  Nokta 4:  (0.1, 2.3, -4.5)  →  Sinif 0  (diseti)
  ...
  Nokta 24000: (3.4, 5.6, -7.8) → Sinif 32 (alt sol lateral kesici)
```

## 5.6 visualization.py - Gorsellestirme

**Ne yapar?** Plotly kutuphanesi ile interaktif 3B grafikler olusturur.

```
GORSELLESTIRME TIPLERI:

  1. Mesafe Haritasi (Distance Heatmap)
     Her nokta mesafe degerine gore renklendirilir
     ┌──────────────────┐
     | ░░░░████████░░░░ |   ░ = Dusuk mesafe (mavi)
     | ░░████████████░░ |   █ = Yuksek mesafe (kirmizi)
     | ░░░░████████░░░░ |
     └──────────────────┘

  2. Risk Haritasi (Risk Heatmap)
     Her dis icin risk skoru cubuk grafigi

     Dis 31: ████████████░░░░░  62/100 [Yuksek]
     Dis 32: █████░░░░░░░░░░░  28/100 [Orta]
     Dis 33: ██░░░░░░░░░░░░░░  12/100 [Dusuk]
     Dis 36: ████████████████  89/100 [Kritik]

  3. Segmentasyon Gorsellestirme
     Her dis farkli renkte gosterilir
     ┌──────────────────┐
     | Her dis farkli   |
     | renkte:          |
     | ██=31 ██=32 ██=33|   Renkler otomatik atanir
     | ██=34 ██=35 ██=36|
     └──────────────────┘
```

## 5.7 train.py - Model Egitim Scripti

**Ne yapar?** PointNet modelini gercek veri ile egitir.

```
EGITIM AKISI:

  [1/4] Veri yukleme
        data_part_1..6 → 1800 tarama
        ↓
  [2/4] PyTorch Dataset olusturma
        Train: 1530 tarama (% 85)
        Val:    270 tarama (% 15)
        ↓
  [3/4] Model olusturma
        PointNet (3.5M parametre)
        ↓
  [4/4] Egitim dongusu (50 epoch)
        ┌──────────────────────────────┐
        |  Her epoch icin:             |
        |  1. Tum egitim verisini gez  |
        |  2. Kayip hesapla            |
        |  3. Agirliklari guncelle     |
        |  4. Her 5 epoch'ta test et   |
        |  5. En iyi modeli kaydet     |
        └──────────────────────────────┘

  Cikti: models/best_model.pth (egitilmis model dosyasi)
```

**Komut satiri parametreleri:**

```
python train.py --epochs 50      # Kac tur egitim yapilacak
                --batch 4        # Ayni anda kac ornek isle
                --lr 0.001       # Ogrenme orani (ne kadar hizli ogren)
                --num-points 24000  # Her ornekten kac nokta al
                --parts 1 2 3 4 5 6  # Hangi data_part'lari kullan
                --val-ratio 0.15     # Validasyon orani (%15)
```

## 5.8 app.py - Streamlit Dashboard

**Ne yapar?** Web tabanli interaktif analiz platformu. Tarayicida acilir.

```
DASHBOARD SAYFALARI:

  ┌─────────────────────────────────────────────────┐
  |  SAYFA 1: GENEL BAKIS                           |
  |  - Veri seti istatistikleri (1900 tarama, vb.)  |
  |  - Ornek mesh metrikleri                        |
  |  - Dis dagilim grafigi                          |
  ├─────────────────────────────────────────────────┤
  |  SAYFA 2: 3B MESH GORSELLESTIRME                |
  |  - Hasta ve cene secimi                         |
  |  - FDI etiketlerine gore renkli 3B model        |
  |  - Nokta sayisi ayarlama                        |
  ├─────────────────────────────────────────────────┤
  |  SAYFA 3: DIS ANALIZI                           |
  |  - Tekil dis secimi (FDI numarasi ile)          |
  |  - Secilen disin 3B gorseli                     |
  |  - Boyut ve vertex istatistikleri               |
  ├─────────────────────────────────────────────────┤
  |  SAYFA 4: DEGISIM SIMULASYONU                   |
  |  - Gercek mesh uzerinde sentetik degisim        |
  |  - Degisim buyuklugu ayari (0.1 - 3.0 mm)      |
  |  - Risk skorlari tablosu                        |
  ├─────────────────────────────────────────────────┤
  |  SAYFA 5: EGITIM BILGISI                        |
  |  - Model egitimi komutlari                      |
  |  - Pipeline aciklamasi                          |
  └─────────────────────────────────────────────────┘

  Nasil calistirilir:
  $ streamlit run app.py
  → Tarayicida http://localhost:8501 adresinde acilir
```

---

# Bolum 6: PointNet Mimarisi Detayli

Bu bolum, projenin en teknik kismi olan PointNet sinir agi mimarisini detayli aciklar.

## 6.1 Neden PointNet?

Normal sinir aglari (CNN gibi) 2B resimleri isler. Ama biz **3B nokta bulutlari** ile calisiyoruz. Temel zorluklar:

```
PROBLEM 1: Siralamaya BAGIMSIZLIK

  Ayni nokta bulutu, farkli siralarda verilebilir:

  Siralama A: [P1, P2, P3, P4]     ┐
  Siralama B: [P3, P1, P4, P2]     ├── Hepsi AYNI sekil!
  Siralama C: [P2, P4, P1, P3]     ┘

  Model, siralama fark etmeksizin ayni sonucu vermeli.
  PointNet bunu MAX POOLING ile cozer.


PROBLEM 2: 3B DONUSUMLERE DAYANIKLILIK

  Ayni sekil farkli acilarda olabilir:

  Duz:    ____         Dondurulmus:  /
         |    |                     / \
         |____|                    /___\

  Model, nesne dondurulse bile ayni sinifi vermeli.
  PointNet bunu T-NET ile cozer.


PROBLEM 3: DEGISEN NOKTA SAYISI

  Farkli mesh'ler farkli sayida noktaya sahip olabilir.
  Ama sinir agi sabit boyutlu giris ister.
  Cozum: Her mesh'ten SABIT 24.000 nokta ornekle.
```

## 6.2 PointNet Mimarisi (Katman Katman)

```
POINTNET SEGMENTASYON MIMARISI:

  GIRIS: (B, 3, N) = (batch_boyutu, xyz_koordinat, nokta_sayisi)
  Ornek: (4, 3, 24000) = 4 tarama, her biri 24.000 nokta

  ┌────────────────────────────────────────────────────────────┐
  |  ADIM 1: INPUT T-NET (Giris Donusumu)                     |
  |                                                            |
  |  Amac: Noktayi standart bir koordinat sistemine cevir      |
  |                                                            |
  |  Giris:  (4, 3, 24000)                                    |
  |  Cikis:  (4, 3, 3) → 3x3 donusum matrisi                 |
  |                                                            |
  |  Bu matris ile noktalar carpilir:                          |
  |  yeni_noktalar = T_matris x noktalar                      |
  |                                                            |
  |  Icerik:                                                   |
  |    Conv1d(3, 64)  → BN → ReLU                             |
  |    Conv1d(64, 128) → BN → ReLU                            |
  |    Conv1d(128, 1024) → BN → ReLU                          |
  |    MaxPool → (4, 1024)                                     |
  |    FC(1024, 512) → BN → ReLU                              |
  |    FC(512, 256) → BN → ReLU                               |
  |    FC(256, 9) → reshape → (4, 3, 3)                       |
  |    + Birim matris (Identity)                               |
  └──────────────────────┬─────────────────────────────────────┘
                         |
                         v  Donusturulmus noktalar (4, 3, 24000)
  ┌────────────────────────────────────────────────────────────┐
  |  ADIM 2: LOKAL OZELLIK CIKARIMI                            |
  |                                                            |
  |  Conv1d(3, 64) → BN → ReLU                                |
  |                                                            |
  |  Giris: (4, 3, 24000)                                     |
  |  Cikis: (4, 64, 24000) ← LOKAL OZELLIKLER                 |
  |                           (her nokta icin 64 boyutlu)      |
  |                           (KAYDEDILIR, sonra kullanilacak) |
  └──────────────────────┬─────────────────────────────────────┘
                         |
                         v  (4, 64, 24000)
  ┌────────────────────────────────────────────────────────────┐
  |  ADIM 3: FEATURE T-NET (Ozellik Donusumu)                 |
  |                                                            |
  |  Ayni T-Net yapisi ama 64x64 donusum matrisi               |
  |                                                            |
  |  Giris: (4, 64, 24000)                                    |
  |  Cikis: (4, 64, 64) → 64x64 donusum matrisi              |
  |                                                            |
  |  Donusturulmus ozellikler: (4, 64, 24000)                 |
  └──────────────────────┬─────────────────────────────────────┘
                         |
                         v  (4, 64, 24000)
  ┌────────────────────────────────────────────────────────────┐
  |  ADIM 4: DERIN OZELLIK CIKARIMI                            |
  |                                                            |
  |  Conv1d(64, 128) → BN → ReLU                              |
  |  Conv1d(128, 1024) → BN                                   |
  |                                                            |
  |  Giris: (4, 64, 24000)                                    |
  |  Cikis: (4, 1024, 24000) ← Her nokta icin 1024 boyut     |
  └──────────────────────┬─────────────────────────────────────┘
                         |
                         v  (4, 1024, 24000)
  ┌────────────────────────────────────────────────────────────┐
  |  ADIM 5: GLOBAL OZELLIK (Max Pooling)                      |
  |                                                            |
  |  24.000 noktanin 1024 boyutlu ozelliklerinden              |
  |  MAX degerini al → TUM SEKLIN OZETI                        |
  |                                                            |
  |  Giris: (4, 1024, 24000)                                  |
  |  Cikis: (4, 1024)        ← Tum sekli ozetleyen vektor     |
  |                                                            |
  |  NEDEN MAX POOLING?                                        |
  |  - Siralama bagimsizligi saglar                            |
  |  - [P1,P2,P3]'un max'i = [P3,P1,P2]'nin max'i             |
  |  - Dolayisiyla noktalarin sirasi farketmez!                |
  └──────────────────────┬─────────────────────────────────────┘
                         |
                         v  Global: (4, 1024)
  ┌────────────────────────────────────────────────────────────┐
  |  ADIM 6: LOKAL + GLOBAL BIRLESTIRME (Concat)              |
  |                                                            |
  |  Global ozelligi (1024) her noktaya kopyala:               |
  |  (4, 1024) → (4, 1024, 24000)                             |
  |                                                            |
  |  Lokal ozelliklerle (64) birlestir:                        |
  |  [Lokal(64) + Global(1024)] = (4, 1088, 24000)            |
  |                                                            |
  |  NEDEN BIRLESTIRIYORUZ?                                    |
  |  - Lokal: Bu noktanin yakinindaki geometri (detay)        |
  |  - Global: Tum seklin genel yapisi (buyuk resim)          |
  |  - Ikisini birlestirerek hem detay hem buyuk resim kullan  |
  └──────────────────────┬─────────────────────────────────────┘
                         |
                         v  (4, 1088, 24000)
  ┌────────────────────────────────────────────────────────────┐
  |  ADIM 7: SEGMENTASYON BASI (Classification Head)           |
  |                                                            |
  |  Conv1d(1088, 512) → BN → ReLU                            |
  |  Conv1d(512, 256) → BN → ReLU                             |
  |  Conv1d(256, 128) → BN → ReLU                             |
  |  Dropout(0.3)       ← Overfitting'i onle                  |
  |  Conv1d(128, 33)    ← 33 sinif icin skor                  |
  |                                                            |
  |  Giris: (4, 1088, 24000)                                  |
  |  Cikis: (4, 33, 24000)                                    |
  |                                                            |
  |  Anlami: Her nokta icin 33 sinifin olasiligi              |
  |  En yuksek olasilik = tahmin edilen sinif                  |
  └──────────────────────┬─────────────────────────────────────┘
                         |
                         v
  ┌────────────────────────────────────────────────────────────┐
  |  SONUC: argmax ile sinif tahmini                           |
  |                                                            |
  |  pred = output.argmax(dim=1)  → (4, 24000)                |
  |                                                            |
  |  Ornek bir nokta icin:                                     |
  |  Sinif 0 (diseti):   0.02                                 |
  |  Sinif 1 (FDI 11):   0.01                                 |
  |  Sinif 2 (FDI 12):   0.01                                 |
  |  ...                                                       |
  |  Sinif 12 (FDI 31):  0.89  ← EN YUKSEK → tahmin = 12     |
  |  ...                                                       |
  |  Sinif 32 (FDI 48):  0.00                                 |
  |                                                            |
  |  Indeks 12 → INDEX_TO_FDI[12] = 31                         |
  |  Sonuc: Bu nokta "Alt sol santral kesici" (FDI 31)         |
  └────────────────────────────────────────────────────────────┘
```

## 6.3 Tensor Boyut Degisimleri Tablosu

Her adimda verinin boyutu nasil degisiyor:

```
B = batch boyutu (ornegin 4)
N = nokta sayisi (24.000)
C = sinif sayisi (33)

Adim                      | Tensor Boyutu     | Aciklama
─────────────────────────────────────────────────────────────
Giris                     | (B, 3, N)         | XYZ koordinatlari
Input T-Net matrisi       | (B, 3, 3)         | Donusum matrisi
Donusturulmus giris       | (B, 3, N)         | Hizalanmis noktalar
Conv1d(3→64)              | (B, 64, N)        | Lokal ozellikler ★
Feature T-Net matrisi     | (B, 64, 64)       | Ozellik donusumu
Donusturulmus ozellikler  | (B, 64, N)        | Hizalanmis ozellikler
Conv1d(64→128)            | (B, 128, N)       | Derin ozellikler
Conv1d(128→1024)          | (B, 1024, N)      | En derin ozellikler
Max Pooling               | (B, 1024)         | Global ozellik
Genisletilmis global      | (B, 1024, N)      | Her noktaya kopyalanmis
Concat [lokal + global]   | (B, 1088, N)      | ★64 + 1024 = 1088
Conv1d(1088→512)          | (B, 512, N)       | Segmentasyon katman 1
Conv1d(512→256)           | (B, 256, N)       | Segmentasyon katman 2
Conv1d(256→128)           | (B, 128, N)       | Segmentasyon katman 3
Conv1d(128→33)            | (B, 33, N)        | Cikis (sinif skorlari)
argmax                    | (B, N)            | Tahmin edilen siniflar
```

## 6.4 Egitim Sureci

### Kayip Fonksiyonu (Loss Function)

```
CROSS ENTROPY LOSS + REGULARIZATION:

  Toplam Kayip = CrossEntropy(tahmin, gercek) + 0.001 * Regularization

  CrossEntropy Ornegi:
  ─────────────────────
  Gercek etiket: Sinif 5 (FDI 15 = Ust sag 2. premolar)

  Model tahmini (olasiliklar):
  Sinif 0:  0.02    Sinif 5:  0.85 ← dogru sinif
  Sinif 1:  0.01    Sinif 6:  0.03
  Sinif 2:  0.01    Sinif 7:  0.02
  Sinif 3:  0.03    ...
  Sinif 4:  0.02    Sinif 32: 0.01

  CrossEntropy = -log(0.85) = 0.163  (dusuk kayip = iyi tahmin!)

  Eger tahmin yanlis olsaydi:
  CrossEntropy = -log(0.02) = 3.912  (yuksek kayip = kotu tahmin!)

  Feature Regularization:
  ──────────────────────
  T-Net matrisinin ortogonal olmasini saglar
  L_reg = ||I - A * A^T||^2
  I = birim matris, A = T-Net ciktisi
```

### Optimizer (Agirlik Guncelleyici)

```
ADAM OPTIMIZER:

  Agirlik guncelleme formulu:
  w_yeni = w_eski - ogrenme_orani * gradyan

  Parametreler:
  - learning_rate = 0.001  (ne kadar buyuk adimlarla ogren)
  - weight_decay = 0.0001  (agirliklarin buyumesini sinirla)

  StepLR Scheduler:
  - Her 20 epoch'ta ogrenme oranini yarilat
  - Epoch 1-20:   lr = 0.001
  - Epoch 21-40:  lr = 0.0005
  - Epoch 41-60:  lr = 0.00025
  - ...

  NEDEN AZALTIYORUZ?
  Baslangicta buyuk adimlar at → yaklasik cozumu bul
  Sonra kucuk adimlar at → hassas ayar yap
```

### Degerlendirme Metrigi: Dice Score (DSC)

```
DICE SIMILARITY COEFFICIENT:

  DSC = 2 * |A ∩ B| / (|A| + |B|)

  A = Model tahmini     B = Gercek etiket
  ∩ = Kesisim (ikisinin de "evet" dedigi noktalar)

  Ornek:
  Gercek:  [0, 0, 1, 1, 1, 0, 0]    (1 = bu dise ait)
  Tahmin:  [0, 0, 1, 1, 0, 0, 0]    (1 = model "bu dis" dedi)

  Kesisim: [0, 0, 1, 1, 0, 0, 0]    |A ∩ B| = 2
  |A| = 2 (tahmin), |B| = 3 (gercek)

  DSC = 2 * 2 / (2 + 3) = 0.80 = %80 benzerlik

  mDSC = Tum siniflarin DSC ortalamalari
  Hedef: mDSC > 0.90 (% 90 dogruluk)
```

---

# Bolum 7: Kurulum ve Calistirma

## 7.1 Python Nedir?

Python, bu projenin yazildigi programlama dilidir. Yapay zeka ve veri biliminde en cok kullanilan dildir.

## 7.2 Gereksinimler

- **Python 3.10 veya uzeri** (python.org'dan indirin)
- **pip** (Python paket yoneticisi, Python ile birlikte gelir)
- **Terminal / Komut Satiri** (macOS: Terminal, Windows: PowerShell)

## 7.3 Adim Adim Kurulum

```bash
# ─── ADIM 1: Proje dizinine gidin ───
# Terminal'i acin ve su komutu yazin:
cd /Users/egearhany/Desktop/inovens-dis
# (Windows'ta: cd C:\Users\...\inovens-dis)

# ─── ADIM 2: Sanal Ortam Olusturun ───
# "Sanal ortam" nedir? Projeye ozel Python ortamidir.
# Baska projelerdeki kutuphanelerle cakismayi onler.
python3 -m venv venv
# Bu komut "venv" adinda bir klasor olusturur.

# ─── ADIM 3: Sanal Ortami Etkinlestirin ───
# macOS / Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
#
# Basarili olursa terminal'de (venv) yazisi gorulur:
# (venv) $ _

# ─── ADIM 4: Kutuphaneleri Yukleyin ───
pip install -r requirements.txt
# Bu komut requirements.txt dosyasindaki tum
# kutuphaneleri otomatik indirir ve kurar.
# (open3d, trimesh, torch, plotly, streamlit, vb.)

# ─── ADIM 5: PyTorch Yukleyin ───
# macOS (Apple Silicon):
pip install torch torchvision
# Linux/Windows (NVIDIA GPU):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 7.4 Demo Calistirma

```bash
# Tum pipeline'i test etmek icin:
python run_demo.py

# Beklenen cikti:
# ============================================================
#   DentalAI - Gercek Veri Demo (OBJ + JSON)
# ============================================================
# [1/7] Veri seti yukleniyor...
#   Toplam: 1900 tarama
# [2/7] Gercek mesh yukleniyor...
#   Vertex: 93,288 | Face: 186,499
# ...
```

## 7.5 Dashboard Baslatma

```bash
# Streamlit web arayuzunu baslatmak icin:
streamlit run app.py

# Tarayicinizda otomatik acilir:
# http://localhost:8501
#
# Acilmazsa tarayiciya bu adresi yapin.
# Durdurmak icin: Terminal'de Ctrl+C
```

## 7.6 Model Egitimi

```bash
# Basit egitim (varsayilan ayarlar):
python train.py

# Hizli test (sadece 2 part, 10 epoch):
python train.py --parts 1 2 --epochs 10

# Tam egitim (tum veri, 100 epoch):
python train.py --epochs 100 --batch 8 --num-points 24000

# NOT: Egitim GPU olmadan cok yavas olabilir.
# Apple Silicon Mac'te MPS hizlandirma otomatik aktiftir.
# NVIDIA GPU'lu bilgisayarda CUDA otomatik aktiftir.
```

## 7.7 Testleri Calistirma

```bash
# 7 test modulunu calistirmak icin:
python tests/test_pipeline.py

# Beklenen cikti:
# Test 1: Veri Seti Yukleme ✓
# Test 2: OBJ Mesh Yukleme ✓
# Test 3: Nokta Ornekleme ✓
# Test 4: Degisim Analizi ✓
# Test 5: Landmark Registration ✓
# Test 6: PointNet Model ✓
# Test 7: PyTorch Dataset ✓
# Tum testler basarili!
```

---

# Bolum 8: Pratik Uygulamalar ve Odevler

## Odev 1: Veri Setini Kesfet (Baslangic Seviye)

**Amac:** Veri setinin yapisini anlamak.

```python
# odev1.py dosyasi olusturun ve su kodu yapin:

import sys
sys.path.insert(0, ".")

from src.data_loader import Teeth3DSDataset

# 1. Veri setini yukle
dataset = Teeth3DSDataset(".", parts=[1, 2])  # Sadece ilk 2 part
stats = dataset.get_statistics()

# 2. Istatistikleri yazdir
print("=== Veri Seti Istatistikleri ===")
print(f"Toplam tarama sayisi:  {stats['total_scans']}")
print(f"Etiketli tarama:       {stats['labeled_scans']}")
print(f"Etiketsiz tarama:      {stats['unlabeled_scans']}")
print(f"Ust cene tarama:       {stats['upper_scans']}")
print(f"Alt cene tarama:       {stats['lower_scans']}")
print(f"Hasta sayisi:          {stats['unique_patients']}")

# 3. Ilk 5 hastayi listele
patients = dataset.get_patient_ids()[:5]
print(f"\nIlk 5 hasta: {patients}")

# 4. Bir hastanin taramalarini gor
for p in patients[:1]:
    scans = dataset.get_patient_scans(p)
    for s in scans:
        print(f"  {s.patient_id} - {s.jaw_type} - {'etiketli' if s.has_labels else 'etiketsiz'}")
```

**Beklenen cikti:**
```
=== Veri Seti Istatistikleri ===
Toplam tarama sayisi:  600
Etiketli tarama:       600
Etiketsiz tarama:      0
Ust cene tarama:       300
Alt cene tarama:       300
Hasta sayisi:          300
```

## Odev 2: Tek Bir Disi Incele (Orta Seviye)

**Amac:** Tek bir disin 3B koordinatlarini incelemek.

```python
# odev2.py dosyasi olusturun ve su kodu yapin:

import sys
sys.path.insert(0, ".")
import numpy as np
from src.data_loader import Teeth3DSDataset

# 1. Veri setinden bir tarama al
dataset = Teeth3DSDataset(".", parts=[1])
scan = dataset.get_labeled_scans()[0]  # Ilk etiketli tarama

# 2. Mesh ve etiketleri yukle
scan.load_mesh()
scan.load_labels()
print(f"Hasta: {scan.patient_id}, Cene: {scan.jaw_type}")
print(f"Vertex sayisi: {scan.num_vertices:,}")
print(f"Face sayisi: {scan.num_faces:,}")

# 3. Mevcut disleri gor
print(f"\nMevcut disler (FDI): {scan.unique_teeth}")
print(f"Dis sayisi: {scan.num_teeth}")

# 4. Bir disi incele (ornegin FDI 31 veya mevcut ilk dis)
from config import FDI_TOOTH_NAMES
fdi_num = scan.unique_teeth[0]  # Ilk mevcut dis
tooth_verts = scan.get_tooth_vertices(fdi_num)
print(f"\n--- FDI {fdi_num}: {FDI_TOOTH_NAMES.get(fdi_num, '?')} ---")
print(f"Vertex sayisi: {len(tooth_verts)}")
print(f"Boyut (mm): X={tooth_verts.ptp(axis=0)[0]:.2f}, "
      f"Y={tooth_verts.ptp(axis=0)[1]:.2f}, "
      f"Z={tooth_verts.ptp(axis=0)[2]:.2f}")
print(f"Merkez (mm): {tooth_verts.mean(axis=0).round(2)}")

# 5. Bellegi temizle
scan.unload()
print("\nBellek temizlendi.")
```

## Odev 3: PointNet ile Tahmin (Ileri Seviye)

**Amac:** Egitilmemis bir PointNet modeli ile tahmin yapmak ve ciktiyi incelemek.

```python
# odev3.py dosyasi olusturun ve su kodu yapin:

import sys
sys.path.insert(0, ".")
import torch
import numpy as np
from src.data_loader import Teeth3DSDataset
from src.segmentation import PointNetSegmentation
from config import NUM_CLASSES, INDEX_TO_FDI, FDI_TOOTH_NAMES

# 1. Veri yukle
dataset = Teeth3DSDataset(".", parts=[1])
scan = dataset.get_labeled_scans()[0]

# 2. Noktalar ornekle
sampled = scan.sample_points(num_points=10000)
points = sampled["points"]          # (10000, 3)
true_labels = sampled["labels"]     # (10000,) - gercek FDI etiketleri
print(f"Orneklenen noktalar: {points.shape}")
print(f"Gercek etiketler: {np.unique(true_labels)}")

# 3. Model olustur (egitilmemis!)
model = PointNetSegmentation(num_classes=NUM_CLASSES)
model.eval()  # Degerlendirme moduna al

# 4. Tahmini yap
points_tensor = torch.from_numpy(points).float().unsqueeze(0)  # (1, 10000, 3)
points_tensor = points_tensor.transpose(2, 1)                   # (1, 3, 10000)

with torch.no_grad():  # Gradyan hesaplama (bellek tasarrufu)
    output, _, _ = model(points_tensor)  # (1, 33, 10000)

# 5. Sonuclari incele
pred_indices = output.argmax(dim=1).squeeze().numpy()  # (10000,)
print(f"\nTahmin edilen indeksler (0-32): {np.unique(pred_indices)}")

# NOT: Model egitilmemis oldugu icin tahminler RASTGELE olacaktir!
# Gercek performans icin egitim yapilmasi gerekir.

print("\n[BILGI] Model henuz egitilmemistir.")
print("Gercek tahminler icin 'python train.py' ile egitim yapin.")

scan.unload()
```

---

# Bolum 9: Sikca Sorulan Sorular

### S: "pip install hata veriyor, ne yapmaliyim?"
**C:** Oncelikle Python surum kontrolu yapin: `python3 --version`. Python 3.10+ olmalidir. Sanal ortamin aktif oldugunu kontrol edin: terminal'de `(venv)` gorunmeli.

### S: "CUDA bulunamadi hatasi aliyorum."
**C:** Bu hata sadece NVIDIA GPU'lu bilgisayarlarda onemlidir. Apple Mac'te bu hata onemsizdir, MPS otomatik kullanilir. GPU'nuz yoksa model CPU'da calisir (yavas ama calisir).

### S: "Veri seti cok buyuk, tum veriyi kullanmak zorunda miyim?"
**C:** Hayir! `--parts 1 2` parametresi ile sadece ilk 2 part'i kullanabilirsiniz (600 tarama). Ogrenme ve test icin yeterlidir.

### S: "Model ne kadar surede egitilir?"
**C:** Donanim ve parametrelere bagli:
- GPU (NVIDIA): ~2-4 saat (50 epoch, tum veri)
- Apple M1/M2: ~4-8 saat
- CPU: ~24-48 saat (onerilmez)
- Test icin `--parts 1 --epochs 5` kullanin (~10 dakika)

### S: "mDSC nedir, ne kadar olmali?"
**C:** Mean Dice Similarity Coefficient = ortalama benzerlik katsayisi. 0 ile 1 arasi bir deger. 0 = tamamen yanlis, 1 = mukkemmel. Bu veri setinde referans degerleri: mDSC > 0.90 iyi, mDSC > 0.95 cok iyi.

### S: "Registration neden gerekli? Taramalar zaten ayni agzi gosteriyor."
**C:** Her taramada tarayicinin pozisyonu farklidir. Iki taramayi ustuste getirmeden degisimleri olcemezsiniz. Registration, iki taramayi ayni koordinat sistemine tasir.

### S: "Neden 33 sinif var? Insanin 32 disi yok mu?"
**C:** 32 dis + 1 diseti (gingiva) = 33 sinif. Diseti de bir sinif olarak ele alinir cunku mesh'te dislerin arasindaki bolge de segmente edilmelidir.

### S: "Lazy loading ne demek?"
**C:** Veriyi ihtiyac olunca yukleme stratejisidir. 1900 mesh'i ayni anda bellege yuklersek ~30 GB RAM gerekir. Bunun yerine sadece islenecek mesh bellege yuklenir, islendikten sonra `unload()` ile serbest birakilir.

### S: "FDI numaralari neden ardisik degil?"
**C:** FDI sistemi agzi 4 kadrana (ceyrek) ayirir. 1. kadran 11-18, 2. kadran 21-28, vb. Bu yuzden 18'den sonra 19 degil 21 gelir. Sinir agi ardisik indeks istediginden `FDI_TO_INDEX` donusumu yapilir.

---

# Bolum 10: Terimler Sozlugu

| Turkce | Ingilizce | Aciklama |
|--------|-----------|----------|
| Yapay Zeka | Artificial Intelligence (AI) | Bilgisayarin ogrenmesi |
| Makine Ogrenmesi | Machine Learning (ML) | Veriden oruuntu ogrenen algoritmalar |
| Derin Ogrenme | Deep Learning (DL) | Cok katmanli sinir aglari |
| Sinir Agi | Neural Network | Beyin esinli hesaplama modeli |
| Vertex (Kose) | Vertex | 3B uzayda bir nokta (x,y,z) |
| Yuz (Ucgen) | Face | 3 vertex'i birlestiren yuzey |
| Normal | Normal | Yuzey yonunu gosteren vektor |
| Mesh (Orfemek) | Mesh | Vertex + Face'lerden olusan 3B model |
| Nokta Bulutu | Point Cloud | Baglantisiz 3B noktalar kumesi |
| Segmentasyon | Segmentation | Veriyi anlamli parcalara bolme |
| Cakistirma | Registration | Iki modeli ayni konuma getirme |
| Egitim | Training | Modelin veriden ogrenme sureci |
| Validasyon | Validation | Egitim sirasinda performans testi |
| Kayip | Loss | Tahmin hatasi (dusuk = iyi) |
| Agirlik | Weight | Modelin ogrendigi degerler |
| Epoch | Epoch | Tum verinin bir kez islenmesi |
| Batch | Batch | Ayni anda islenen ornekler grubu |
| Ogrenme Orani | Learning Rate | Agirliklarin ne kadar degisecegi |
| Overfitting | Overfitting | Modelin ezberleme sorunu |
| Veri Sizintisi | Data Leakage | Egitim verisinin test'e karismasi |
| FDI | FDI | Uluslararasi dis numaralama sistemi |
| Diseti | Gingiva | Dislerin etrafindaki yumusak doku |
| Santral Kesici | Central Incisor | On dis |
| Lateral Kesici | Lateral Incisor | On dis yani |
| Kanin | Canine | Kopek disi |
| Premolar | Premolar | Kucuk azi disi |
| Molar | Molar | Buyuk azi disi |
| Hausdorff Mesafesi | Hausdorff Distance | En kotu durum yuzey sapmasi |
| KDTree | KD-Tree | Hizli en yakin komsu arama yaprisi |
| RANSAC | RANSAC | Rastgele ornekleme ile kaba eslestirme |
| ICP | Iterative Closest Point | Yakin nokta tekrarlama ile ince hizalama |
| FPFH | Fast Point Feature Histogram | Nokta ozellik vektoru |
| T-Net | Transformer Network | Koordinat donusum agi |
| Max Pooling | Max Pooling | En buyuk degeri secme islemi |
| Conv1d | 1D Convolution | 1 boyutlu esikleme islemi |
| BatchNorm | Batch Normalization | Katman cikisini normalize etme |
| ReLU | Rectified Linear Unit | Aktivasyon fonksiyonu (max(0,x)) |
| Dropout | Dropout | Rastgele noron kapatma (overfitting onleme) |
| Adam | Adam Optimizer | Adaptif agirlik guncelleme algoritmasi |
| DSC | Dice Similarity Coefficient | Benzerlik olcum metrigi |
| Streamlit | Streamlit | Python web uygulama cercevesi |
| Plotly | Plotly | Interaktif grafik kutuphanesi |
| Open3D | Open3D | 3B veri isleme kutuphanesi |
| Trimesh | Trimesh | Mesh dosya okuma kutuphanesi |
| PyTorch | PyTorch | Derin ogrenme cercevesi |
| GPU | Graphics Processing Unit | Grafik isleme birimi (hizlandirici) |
| MPS | Metal Performance Shaders | Apple GPU hizlandirma |
| CUDA | CUDA | NVIDIA GPU hizlandirma |

---

> **Bu rehber, DentalAI projesinin universite ogrencileri icin hazirlanmis kapsamli anlatim dokumanidir.**
> Sorulariniz icin proje sahipleri ile iletisime geciniz.
