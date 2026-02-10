
---
# Absenteeism ML Pipeline & Technical Case Study

## Business Case Study: İş Yerinde Devamsızlık (Absenteeism) Analizi

Bu vaka çalışmasında amacımız, bir şirketteki çalışanların devamsızlık alışkanlıklarını analiz ederek, belirli özelliklere sahip bir çalışanın gelecekte kaç saat işe gelmeyeceğini tahmin etmektir. Bu, sadece bir veri analizi değil, doğrudan **üretim kapasitesini ve iş kalitesini** etkileyen stratejik bir karardır.

###  1. Üçlü Araç Stratejisi (The Triple Threat)

Analizimizde en iyi sonucu almak için üç farklı devin gücünü birleştiriyoruz. Neden tek bir araç kullanmıyoruz?

- **SQL (MySQL Workbench):** Veri depolama ve yönetiminde en güçlü araçtır. Veriyi ham haliyle saklamak ve temel düzenlemeleri yapmak için kullanılır.
    
- **Python (Jupyter Notebook):** Matematiksel ve istatistiksel araç seti en geniş olan dildir. Burada **Logistic Regression (Lojistik Regresyon)** algoritmasını kullanarak tahminleme yapacağız.
    
- **Tableau (Tableau Public):** Veri görselleştirmede rakipsizdir. Python'dan aldığımız sonuçları, yöneticilerin tek bakışta anlayabileceği "keskin" grafiklere dönüştüreceğiz.
    

> **Entegrasyon Notu:** Yazılım entegrasyonu karmaşasına girmemek için, bu araçlar arasındaki veri alışverişini **CSV dosyaları** üzerinden, aynı mantıksal akışı koruyarak gerçekleştireceğiz.

---

###  2. İş Probleminin Tanımı: Devamsızlık (Absenteeism)

Günümüzün rekabetçi iş dünyasında artan baskı ve stres, çalışanlarda geçici veya uzun süreli sağlık sorunlarına (örneğin depresyon) yol açabilir. Bu durum devamsızlığa neden olur.

- **Tanım:** "Normal çalışma saatleri içinde iş yerinde bulunmama durumu."
    
- **Odak Noktamız:** Biz bir İnsan Kaynakları (İK) uzmanı değil, **verimlilikten sorumlu** bir analistiz. Bu yüzden devamsızlığın tıbbi boyutuyla değil, **tahmin edilebilirliği** ile ilgileniyoruz.
    
- **Hedef:** Bir çalışanın işe ne kadar uzak yaşadığı, kaç çocuğu veya evcil hayvanı olduğu, eğitim düzeyi gibi verileri kullanarak; bu kişinin belirli bir iş gününde kaç saat devamsızlık yapacağını öngörmek.
    

---

###  3. Veri Kaynakları: Birincil (Primary) vs. İkincil (Secondary) Data

Çalışacağımız verinin doğasını anlamak, analizin güvenilirliği için şarttır:

- **Primary Data (Birincil Veri):** Sizin oluşturduğunuz veridir. Kendi yaptığınız bir anket veya şirketin kendi satış kayıtları buna örnektir.
    
- **Secondary Data (İkincil Veri):** Başkası tarafından toplanmış ve organize edilmiş veridir. İnternetten indirilen veya bir kurumdan satın alınan setler bu sınıfa girer.
    
- **Durumumuz:** Bu çalışmada **Secondary Data** kullanacağız. Elimizdeki veri seti "Raw Data" (Ham Veri) niteliğindedir; yani üzerinde hiçbir işlem yapılmamış, analiz için cilalanmamıştır.
    

---

###  4. Data Pre-processing (Veri Ön İşleme) Gerekliliği

Ham veriyi doğrudan analiz etmek, kirli bir mercekle gökyüzüne bakmaya benzer. Veri ön işleme adımı şunları hedefler:

1. Veri toplama sırasında oluşan hataları gidermek.
    
2. Veriyi anlaşılabilir ve işlenebilir bir formata sokmak.
    
3. İstatistiksel modellerin (Logistic Regression gibi) veriyi hatasız okumasını sağlamak.
    

---

##  Vaka Çalışması Hazırlık Kontrol Listesi

Analize başlamadan önce şu zihinsel ve teknik hazırlıkları tamamladığından emin ol:

- [ ] **İş Mantığı:** Problemi sadece kod olarak değil, şirketin verimliliğini artıracak bir "strateji" olarak görüyor musun?
    
- [ ] **Araç Kurulumu:** MySQL Workbench, Jupyter Notebook ve Tableau Public yazılımların hazır mı?
    
- [ ] **Veri Tanımı:** Çalışacağımız verinin "ikincil" ve "ham" olduğunu, bu yüzden ön işlemeye ihtiyaç duyacağını biliyor musun?
    
- [ ] **Stratejik Plan:** SQL'den Python'a, oradan Tableau'ya uzanan akış şemasına hakim misin?
    
- [ ] **Tahmin Hedefi:** Tahmin etmeye çalıştığımız şeyin "çalışan karakteristiklerine dayalı devamsızlık saatleri" olduğunu netleştirdin mi?
    
---

##  Absenteeism Case Study: Proje Yapısı ve Teknik Akış

Bu bölüm, ham verinin (raw data) işlenmesinden başlayarak, bir makine öğrenmesi modelinin paketlenmesi ve görselleştirilmesine kadar uzanan dört ana evreden oluşur. Projenin omurgasını, veriyi bir "modül" haline getirerek sürdürülebilir kılmak oluşturur.

### 1. Veri Ön İşleme (Data Preprocessing)

Analitik görevlerin en kritik ve zaman alıcı aşamasıdır. Bu evrede ==`Absenteeism_data.csv`== dosyası üzerinde çalışılarak veri, makine öğrenmesi algoritmalarının kabul edebileceği bir forma sokulacaktır.

- **Dönüşüm:** Sürecin sonunda hedef, ==`df_preprocessed.csv`== dosyasına ulaşmaktır.
    
- **Kapsam:** Veri temizleme, eksik değer yönetimi, kategorik değişkenlerin kodlanması ve ölçeklendirme gibi işlemler bu aşamada detaylandırılır. Bu aşamanın sonunda elde edilen dosya, makine öğrenmesi aşaması için standart bir başlangıç noktası teşkil eder.
    

### 2. Makine Öğrenmesi ve Model Geliştirme

Ön işleme aşamasında hazırlanan veri seti kullanılarak, bir bireyin iş yerinden "aşırı devamsızlık" (excessive absence) yapma olasılığını tahmin eden bir model geliştirilecektir.

- **Model Seçimi:** Bu vaka çalışması için **Lojistik Regresyon (Logistic Regression)** modeli kullanılacaktır.
    
- **Yazılım Ürününe Dönüştürme:** Eğitilen model ve ön işleme adımları, ==`absenteeism_module`== adı verilen bir Python modülü olarak kaydedilecektir. Bu, modelin sadece o anki analiz için değil, gelecekteki yeni veriler için de kullanılabilir olmasını sağlar.
    

### 3. Modülün Yüklenmesi ve Tahminleme (Loading the Module)

Geliştirilen `absenteeism_module` sistem tarafından çağrılarak içindeki metotlar test edilir.

- **Amaç:** Yazılım entegrasyonu disiplinine uygun olarak, modelin yeni gözlemler (new observations) üzerinde nasıl tahmin yürüttüğünü simüle etmektir.
    
- **Fonksiyonellik:** Modül, girdi verilerini alır, kendi içinde ön işlemeden geçirir ve sonuç olarak devamsızlık olasılıklarını çıktı olarak verir.
    

### 4. Tableau ile Çıktı Analizi ve İçgörü

Elde edilen tahminlerin ve model girdilerinin arasındaki ilişkiler Tableau üzerinden analiz edilir.

- **Görselleştirme Odağı:** Model girdileri arasındaki üç ayrı bağımlılık (dependencies) incelenerek, basit tabloların sunamadığı derinlemesine iş içgörüleri aranır.
    
- **Karar Destek:** Bu aşama, karmaşık istatistiksel sonuçların yönetici düzeyindeki kararlara temel oluşturacak "dataviz" (veri görselleştirme) ürünlerine dönüştüğü final aşamasıdır.
    

---

##  Proje Metodolojisi Özeti

Proje süresince şu teknik geçişlerin doğruluğu kontrol edilmelidir:

Ham Veri (`.csv`) -> Ön İşleme Süreci -> İşlenmiş Veri (`df_preprocessed.csv`)

İşlenmiş Veri -> Lojistik Regresyon Eğitimi -> Python Modülü
(`absenteeism_module`)

Modül Çıktıları -> Tahminler ve Olasılıklar -> Tableau Görselleştirmeleri

Bu yapı, bir veri bilimcinin sadece model kurmakla kalmayıp, bu modeli iş akışına entegre etme (Deployment/Integration) yeteneğini de geliştirir.

Absenteeism_Case/
├── Data/               # Sadece CSV dosyaları (Ham ve İşlenmiş)
├── Notebooks/          # .ipynb uzantılı çalışma dosyaların
├── Modules/            # Oluşturacağımız absenteeism_module.py dosyası
└── Final_Outputs/      # Tableau için hazır hale getireceğimiz final CSV'ler

- **Data_0**: Ham `Absenteeism_data.csv` dosyasını burada saklayarak orijinal veriyi koruma altına alabilirsin.
    
- **Notebooks_1**: Veri ön işleme ve makine öğrenmesi süreçlerini adım adım dokümante edeceğin çalışma alanın burası olacak.
    
- **Modules_2**: Lojistik regresyon modelini ve ön işleme adımlarını içeren `absenteeism_module` bu klasörde paketlenecek.
    
- **Final_outputs_3**: Tableau'da görselleştirilecek olan nihai çıktılarını burada depolayacaksın.

---
# Reprocessing the Data

## LAB 

### Anaconda Prompt Terminal

```bash
`cd \Users\admin\Documents\Absenteeism_Case`
`jupyter lab`
```

---
##  Veri İçe Aktarma ve DataFrame Yapılandırması

Bu aşama, ham verinin Python ortamına aktarılarak manipülasyona hazır hale getirilmesini kapsar.

### 1. Pandas Modülünün Entegrasyonu

Veri bilimi projelerinde panel verilerle (tabular data) çalışmak için en temel araç **Pandas** kütüphanesidir. Teknik olarak bu kütüphane, veriyi **DataFrame** adı verilen, satır ve sütunlardan oluşan özel bir nesne yapısında saklamamıza olanak tanır.

- **Convention (Gelenek):** Kod yazımını hızlandırmak ve topluluk standartlarına uymak için Pandas kütüphanesi `pd` kısaltması ile içe aktarılır.
    

### 2. .read_csv() Metodu ve Dosya Yolları

Veriyi okumak için kullanılan temel fonksiyon ==`pd.read_csv()`== metodudur. Bu metodun kullanımıyla ilgili kritik teknik detaylar şunlardır:

- **Tırnak Kullanımı:** Dosya ismi belirtilirken tek (`'`) veya çift (`"`) tırnak kullanımı Python tarafından farksız kabul edilir.
    
- **Dosya Uzantısı:** Dosya isminin sonuna mutlaka `.csv` uzantısı eklenmelidir.
    
- **Path (Yol) Yönetimi:** Eğer çalışma dosyası (.ipynb) ile veri dosyası (.csv) aynı klasörde değilse, tam dosya yolu belirtilmelidir. 
    

---

##  Uygulama: Veriyi Yükleme ve Görüntüleme

Jupyter Lab üzerinde `Notebooks_1` klasöründeki dosyanın ilk hücresine şu kodu yazıp çalıştırıyoruz:

Python

```python
import pandas as pd

# pd.read_csv metodu ile veriyi değişkene atıyoruz
# klasör yapına göre yol: Data_name
raw_csv_data = pd.read_csv('../Data_name/Absenteeism_data.csv')

# Değişken ismini yazarak tabloyu görüntülüyoruz
raw_csv_data
```

### Teknik Notlar

- **raw_csv_data:** Bu değişken ismi, verinin henüz "ham" (işlenmemiş) olduğunu hatırlatması için seçilmiştir.
    
- **Execution:** Hücreyi çalıştırdığında (Shift + Enter), verinin tablo formatında ekrana gelmesi, Python'un veriyi başarıyla parse ettiğini gösterir.
    

---

**Analiz:** Veriyi başarıyla yüklediğimize göre, bir sonraki aşamada bu verinin bir kopyasını oluşturup (Checkpoint) üzerinde ilk düzenlemeleri yapmaya başlayacağız.

---

##  Veri İnceleme ve Güvenli Çalışma Alanı (Checkpoint)

Veri ön işlemenin (preprocessing) ilk kuralı, veriyi "gözle muayene" (eyeballing) etmektir. Bu, verinin dikey ve yatay organizasyonunu anlamamızı sağlar.

### 1. Veri Çerçevesinin (DataFrame) Yapısı

- **Sütun Başlıkları (Headers):** Python, varsayılan olarak CSV dosyasının ilk satırını sütun isimleri olarak kabul eder ve veri değerlerinden ayırır.
    
- **İndeksleme (Vertical Organization):** Veriler dikey olarak 0'dan başlayan indekslerle sıralanır. Programlama mantığında sayma işlemi 1 yerine 0'dan başladığı için, 700 satırlık bir veri setinde son indeks numarası 699'dur.
    
- **Yatay Organizasyon:** Veri setimiz `ID` ile başlar ve hedef değişkenimiz olan `Absenteeism time in hours` ile sona erer.
    

### 2. Checkpoint Oluşturma: `.copy()` Metodu

Veri üzerinde manipülasyon yapmaya başlamadan önce orijinal veri setini korumak teknik bir zorunluluktur.

- **Neden Copy?** Eğer doğrudan `raw_csv_data` üzerinde değişiklik yaparsan (örneğin bir sütunu silersen), hata yaptığında orijinal veriye geri dönme şansın kalmaz.
    
- **Uygulama:** Python'da genel kabul görmüş `df` (DataFrame'in kısaltması) ismini kullanarak verinin bir kopyasını oluştururuz.
    

Python

```python
# Verinin bir kopyasını oluşturarak güvenli alana geçiyoruz
df = raw_csv_data.copy()
```

Bu işlemden sonra `df` üzerinde yaptığımız hiçbir hata orijinal `raw_csv_data` değişkenini etkilemez.

---

## Jupyter Lab Görüntüleme ve Analiz İpuçları

Veri setin çok geniş olduğunda Jupyter bazı satır ve sütunları gizleyebilir. Tüm veriyi görmek veya hızlıca analiz etmek için şu teknikleri kullanabilir:

### 1. Sınırsız Görüntüleme Ayarları

Eğer tüm satır ve sütunların görünür olmasını isteniyorsa, `None` anahtar kelimesiyle limitleri kaldırabilir:

Python

```python
# Görüntüleme limitlerini kaldırma
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Veriyi görüntüleme
display(df)
```

### 2. Veri Özeti ve Eksik Değer Kontrolü: `.info()`

Veri setinde eksik (missing) değer olup olmadığını ve veri tiplerini anlamanın en hızlı yolu `.info()` metodudur.

Python

```python
# Verinin teknik özetini alalım
df.info()
```

**Analiz Çıktısı:** `df.info()` komutunu çalıştırdığında her sütun için "700 non-null" ibaresini görüyorsan, bu veri setinde hiç eksik değer olmadığı anlamına gelir. Bu, vaka çalışmamız için oldukça temiz bir başlangıç noktasıdır.

---

###  Sonraki Adım

Notebook dosyanın yeni hücresine şu adımları uygula:

1. `df = raw_csv_data.copy()` satırıyla kopyanı oluştur.
    
2. `df.info()` komutunu çalıştır ve çıktıdaki satır sayılarını kontrol et.
    
3. Çıktıda tüm sütunların `int64` veya `float64` olup olmadığını, yani sayısal olup olmadıklarını gözlemle.
    

Bu kontrolleri yaptıktan sonra, veri setindeki ilk anlamlı müdahalemizi yapacağız. 

---
##  Terminoloji ve Regresyon Analizine Giriş

Veri bilimi projelerinde kullanılan terimler; matematik, programlama ve veri analitiği disiplinleri arasında farklı anlamlara gelebilir. Analizin ilerleyen aşamalarında karmaşayı önlemek adına bu tanımları standartlaştırmak gerekir.

### 1. "Variable" (Değişken) Teriminin Üç Farklı Boyutu

Aynı terim, bağlama göre farklı teknik karşılıklar bulur:

- **Matematik:** Henüz bilinmeyen bir değeri temsil eden $x$ veya $y$ gibi sembollerdir.
    
- **Veri Analitiği (İstatistik):** Zaman içinde veya farklı koşullarda değişen niteliklerdir (Örn: Yaş). Analitik bağlamda bunlar **Independent (Bağımsız)** veya **Dependent (Bağımlı)** değişkenler olarak ayrılır.
    
- **Bilgisayar Programlaması:** Bilginin saklandığı bir depolama lokasyonudur. Örneğin, `raw_csv_data` tüm veri setini içeren bir "programlama değişkenidir".
    

> **Standartlaştırma:** Karmaşayı azaltmak için veri bilimi topluluğu, veri analitiği bağlamındaki değişkenlere **"Features" (Özellikler)**, **"Attributes" (Nitelikler)** veya **"Inputs" (Girdiler)** demeyi tercih eder. Bu vaka çalışmasında biz de sütunlar için "Features" terimini kullanacağız.

### 2. Yapısal İsimlendirme Farklılıkları

- **Data Table (Veri Tablosu):** Analitik dünyasındaki genel tablodur.
    
- **DataFrame:** Bu tablonun Python (Pandas) dilindeki teknik karşılığıdır.
    
- **Arrays (Diziler):** Matematikteki vektör ve matrislerin Python'daki karşılığıdır.
    

---

##  Regresyon Analizi ve Lojistik Regresyon

Analizimizin temelini oluşturacak olan Regresyon Analizi, bir değişkenin değerini diğer değişkenler aracılığıyla açıklayan bir denklemdir.

### Regresyon Denkleminin Bileşenleri

- **Dependent Variable (Bağımlı Değişken / Target):** Değeri, diğer değişkenlere bağlı olan ve tahmin edilmek istenen sonuçtur.
    
- **Independent Variables (Bağımsız Değişkenler / Predictors / Features):** Bağımlı değişkenin değerini açıklayan veya tahmin etmeye yardımcı olan girdilerdir.
    

### Lojistik Regresyon (Logistic Regression) Nedir?

Lojistik regresyon, bağımlı değişkenin **Binary (İkili)** olduğu bir regresyon modelidir. Yani sonuç sadece iki değer alabilir: 0 veya 1, Doğru veya Yanlış, Evet veya Hayır. Amacımız, girdileri kullanarak bu iki sonuçtan birinin gerçekleşme olasılığını yüksek bir tahmin gücüyle hesaplamaktır.

---

##  Veri Seti Üzerinde Mantıksal Eşleştirme

Elimizdeki `df` (DataFrame) yapısını bu teorik çerçeveye oturtalım:

- **Bağımlı Değişken (Target):** `Absenteeism time in hours` sütunudur. Bir çalışanın belirli bir tarihte kaç saat devamsızlık yaptığını gösterir.
    
    - _Örn:_ ID 11 olan çalışan 7 Temmuz 2015'te 4 saat devamsızlık yapmıştır.
        
- **Bağımsız Değişkenler (Features):** ID, Reason for Absence, Date, Age gibi diğer tüm sütunlardır. Bu girdileri kullanarak gelecekteki devamsızlıkları tahmin etmeye çalışacağız.
    

---

##  Uygulama Öncesi Teknik Not

Daha önce yaptığımız `df.info()` kontrolü ile veride eksik değer olmadığını (700 non-null) teyit etmiştik. Bu durum, veri temizleme aşamasının "eksik veri tamamlama" kısmını atlamamıza ve doğrudan veriyi anlamlandırma ile manipüle etme (Preprocessing) aşamasına geçmemize olanak tanır.

**Bir sonraki adım?**

Teorik hazırlığımız tamamlandığına göre, veri ön işlemenin ilk somut adımı olan **"ID" sütununun analizden çıkarılması** ve **"Reason for Absence" sütununun sayısallaştırılması**.

---

##  ID Sütununun Analiz Dışı Bırakılması

Veri setindeki ilk sütun olan `ID`, her bir çalışanı temsil eden benzersiz bir kimlik numarasıdır. Ancak bu bilgi, makine öğrenmesi modeli için yanıltıcı olabilir.

### 1. Nominal Veri ve İstatistiksel Hata

`ID` sütunu sayısal değerler içerse de teknik olarak **Nominal Data** (isimlendirme verisi) kategorisindedir.

- **Sorun:** 5 veya 10 gibi ID değerlerinin büyüklüğü, devamsızlık süresi üzerinde sayısal bir anlam ifade etmez.
    
- **Risk:** Eğer bu sütun modelde kalırsa, algoritma ID numaraları ile devamsızlık süresi arasında sahte (pseudo) bir korelasyon kurmaya çalışabilir ve bu da tahminlerimizin doğruluğunu bozar.
    

### 2. Python'da Sütun Düşürme (Drop) Mantığı

Python'da bir sütunu silmek için `.drop()` metodunu kullanırız. Burada iki kritik teknik detay vardır:

- **Axis (Eksen) Belirleme:** Python varsayılan olarak satırlarda (`axis=0`) arama yapar. Biz sütun silmek istediğimiz için yatay ekseni, yani `axis=1` değerini belirtmek zorundayız.
    
- **Kalıcı Değişiklik:** `.drop()` metodu varsayılan olarak geçici bir kopya döndürür. Değişikliğin `df` değişkenine kalıcı olarak işlenmesi için değişkeni kendisine tekrar atamamız gerekir (`df = df.drop(...)`).
    

Python

```python
# ID sütununu kalıcı olarak düşürüyoruz
# axis=1 sütunları, axis=0 satırları temsil eder
df = df.drop(['ID'], axis=1)

# Kontrol: ID sütunu artık tabloda yer almamalı
df.head()
```

---

## Reason for Absence (Devamsızlık Nedeni) Analizi

Bir sonraki durağımız `Reason for Absence` sütunu. Bu sütun, devamsızlığın arkasındaki nedenleri sayısal kodlarla ifade eder.

### 1. Sütun İnceleme Metotları

Belirli bir sütunu incelemek için `df['Column Name']` söz dizimini kullanırız. Sütundaki değerlerin aralığını ve çeşitliliğini anlamak için şu metotları uygularız:

- **.min() ve .max():** Değerlerin 0 ile 28 arasında değiştiğini gösterir.
    
- **.unique():** Sütundaki tüm benzersiz (tekrarlanmayan) değerleri listeler.
    
- **len():** Benzersiz değerlerin toplam sayısını verir.
    

### 2. "Hawk-Eye" (Şahin Gözü) Kontrolü: Eksik Değer Tespiti

Veri ön işlemede detaylara dikkat etmek, analizin sağlamlığını artırır.

- `.min()` 0 ve `.max()` 28 iken, toplamda 29 farklı değer (0 dahil) bekleriz.
    
- Ancak `len(df['Reason for Absence'].unique())` komutunu çalıştırdığımızda sonuç **28** çıkar.
    
- Bu durum, 0 ile 28 arasında bir sayının veri setinde hiç yer almadığını gösterir.
    

Python

```python
# Benzersiz değerleri alıp sıralıyoruz
unique_reasons = df['Reason for Absence'].unique()
sorted_reasons = sorted(unique_reasons)

print(sorted_reasons)
```

**Teknik Analiz:** Liste incelendiğinde **20** sayısının eksik olduğu görülür. Bu tür detaylar, verinin doğasını anlamak ve ileride yapılacak gruplandırmalar (dummy variables) için kritik öneme sahiptir.

---

###  Uygulama Adımı ve Sonraki Aşama

Notebook dosyanızda şu işlemleri sırasıyla gerçekleştirin:

1. `df = df.drop(['ID'], axis=1)` satırını çalıştırarak kimlik bilgilerini temizleyin.
    
2. `df['Reason for Absence'].unique()` ile değerleri görün.
    
3. `sorted()` fonksiyonu ile 20'nin eksik olduğunu teyit edin.
    

---

##  Kategorik Verilerin Sayısallaştırılması ve Dummy Değişkenler

Veri setindeki `Reason for Absence` sütunu 0 ile 28 arasında sayılar içerir. Ancak bu sayılar matematiksel bir büyüklük ifade etmez; sadece kategorileri temsil eder (Nominal Data).

### 1. Neden Sayısal Kodlar Kullanılır?

Analitik dünyada hastalık isimleri (Örn: "Infectious and parasitic diseases") yerine 1, 2, 3 gibi kodlar kullanılmasının iki ana nedeni vardır:

- **Veritabanı Optimizasyonu:** Uzun metinler (strings) yerine 1-2 basamaklı sayılar saklamak, veri boyutunu küçültür ve depolama maliyetini düşürür.
    
- **Hızlı Referans:** Araştırmacılar bir "lejant" (anahtar tablo) kullanarak sayılar üzerinden hızlıca sınıflandırma yapabilirler.
    

### 2. Dummy Variable (Kukla Değişken) Kavramı

Lojistik regresyon gibi modeller, "neden 2, neden 1'den büyüktür?" sorusunu sayısal olarak sorar. Bu hatayı önlemek için **Dummy Variables** kullanırız.

- **Tanım:** Bir kategorik etkinin varlığını 1, yokluğunu 0 ile gösteren ikili (binary) değişkenlerdir.
    
- **Mantık:** Eğer bir çalışan 1 numaralı nedenle devamsızlık yaptıysa, "Reason 1" sütunu **1**, diğer tüm neden sütunları **0** olur.
    

---

##  Uygulama: `pd.get_dummies` ile Dönüşüm

Pandas kütüphanesindeki `get_dummies` metodu, tek bir sütunu 28 ayrı sütuna (her bir neden için bir sütun) ayırmamızı sağlar.

Python

```python
# Her bir neden için ayrı sütunlar oluşturuyoruz
reason_columns = pd.get_dummies(df['Reason for Absence'])

# Oluşan yeni tabloyu inceleyelim
reason_columns.head()
```

---

##  Veri Bütünlüğü Kontrolü (Data Integrity Check)

Bu vaka çalışmasında bir çalışanın aynı anda **yalnızca bir** nedenle devamsızlık yaptığı varsayılır. Bu kuralın veri setinde korunup korunmadığını matematiksel olarak doğrulamalıyız.

### 1. Yatay Toplam Kontrolü (`axis=1`)

Eğer her satırda sadece bir tane "1" varsa (yani kişi tek bir nedenle gittiyse), o satırdaki tüm dummy sütunlarının toplamı tam olarak **1** olmalıdır.

Python

```python
# Kontrol sütunu oluşturma (Satır bazında toplama)
reason_columns['check'] = reason_columns.sum(axis=1)

# İlk değerleri görelim
reason_columns[['check']].head()
```

### 2. İstatistiksel Sağlama

Tüm satırların doğru olduğunu kanıtlamak için iki yöntem kullanırız:

- **Toplam Sayı:** Tüm `check` sütununu topladığımızda satır sayısına (700) ulaşmalıyız.
    
- **Benzersiz Değerler:** `check` sütununda 1'den başka bir sayı (0 veya 2 gibi) olmamalıdır.
    

Python

```python
# Yöntem A: Toplamın satır sayısına eşitliği
print(reason_columns['check'].sum()) # Çıktı: 700

# Yöntem B: Benzersiz değerlerin kontrolü
print(reason_columns['check'].unique()) # Çıktı: [1]
```

**Analiz:** `unique()` sonucunda sadece **1** değerini görüyorsak, verimiz kusursuzdur. Hiçbir çalışan için birden fazla neden girilmemiş veya hiçbir neden boş bırakılmamıştır.

---

##  Temizlik ve Sonraki Adım

Doğrulamayı tamamladığımıza göre, artık ihtiyacımız olmayan `check` sütununu ve modelde karmaşaya yol açabilecek "0" numaralı (hiçbir neden belirtilmemiş) dummy sütununu yönetmemiz gerekecek.

Python

```python
# Kontrol sütununu siliyoruz
reason_columns = reason_columns.drop(['check'], axis=1)
```

---

Bu aşama, veri ön işlemenin en stratejik noktasıdır. Sadece veriyi temizlemekle kalmıyor, aynı zamanda modelin performansını doğrudan etkileyecek olan **Öznitelik Mühendisliği (Feature Engineering)** ve **Boyut Azaltma (Dimensionality Reduction)** işlemlerini gerçekleştiriyoruz.

---

## Çoklu Doğrusal Bağlantı (Multicollinearity) ve Dummy Tuzağı

Bir kategorik değişkeni dummy değişkenlere dönüştürürken karşılaşılan en büyük risk **Multicollinearity** (Çoklu Doğrusal Bağlantı) sorunudur.

### 1. Neden "Reason 0"ı Düşürüyoruz?

Eğer bir değişkenin $n$ adet kategorisi varsa, bu kategorileri temsil etmek için $n-1$ adet dummy değişken yeterlidir.

- **Mantık:** Eğer bir kişi 1'den 28'e kadar olan hiçbir kategoriye girmiyorsa, onun "0" kategorisinde olduğu zaten matematiksel olarak bellidir.
    
- **Dummy Variable Trap:** Tüm sütunları (0 dahil) modele dahil edersek, sütunlar arasında mükemmel bir doğrusal ilişki oluşur. Bu durum regresyon modelinin katsayılarını hesaplanamaz hale getirir.
    

Python

```python
# 'drop_first=True' parametresi ile 'Reason 0' sütununu otomatik olarak eliyoruz
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
```

---

##  Değişkenlerin Gruplandırılması (Classification)

Elimizde 27 adet dummy sütunu var. 700 gözlemlik bir veri seti için 27 özellik çok fazladır. Bu durumu basitleştirmek için benzer nedenleri 4 ana grupta toplayacağız:

|**Grup**|**Neden Aralığı**|**İçerik Özeti**|
|---|---|---|
|**Grup 1**|1 - 14|Çeşitli ciddi hastalıklar|
|**Grup 2**|15 - 17|Hamilelik ve doğum ile ilgili nedenler|
|**Grup 3**|18 - 21|Zehirlenme ve sınıflandırılmamış belirtiler|
|**Grup 4**|22 - 28|Hafif nedenler (Diş randevusu, fizik tedavi vb.)|

### Teknik Uygulama: `.loc` ve `.max()`

Her satırda sadece bir adet "1" olduğunu bildiğimiz için, belirli sütun aralıklarının maksimum değerini alarak o grubun aktif olup olmadığını saptayabiliriz.

Python

```python
# Grupları oluşturma
reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
```

---

##  Birleştirme (Concatenate) ve Sütun Yönetimi

Şimdi oluşturduğumuz bu 4 grubu ana veri çerçevemiz olan `df` ile birleştireceğiz ve artık ihtiyacımız olmayan orijinal `Reason for Absence` sütununu sistemden çıkaracağız.

### 1. Birleştirme İşlemi

`pd.concat` fonksiyonu ile nesneleri yatay eksende (`axis=1`) yan yana getiriyoruz.

Python

```python
# Orijinal sütunu düşürüyoruz
df = df.drop(['Reason for Absence'], axis=1)

# Yeni grupları df'e ekliyoruz
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
```

### 2. Sütun İsimlerini Yeniden Düzenleme

Yeni eklenen sütunlar 0, 1, 2, 3 gibi isimler alacaktır. Bunları anlamlı hale getirmek için `.columns.values` üzerinden toplu bir isimlendirme yapıyoruz.

Python

```python
# Mevcut sütun isimlerini bir listeye alalım
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
               'Daily Work Load Average', 'Body Mass Index', 'Education',
               'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1',
               'Reason_2', 'Reason_3', 'Reason_4']

# Yeni isimleri atayalım
df.columns = column_names

# Sonucu kontrol edelim
df.head()
```

---

##  Teknik Analiz Notu

Bu işlemler sonucunda verimizi hem sadeleştirdik hem de modelin öğrenme kapasitesini artırdık. Artık devamsızlık nedenleri 27 karmaşık sayı yerine 4 anlamlı biyolojik/sosyal grup üzerinden temsil ediliyor.

---
## Adım 3: İkinci Checkpoint (Önemli!)

Şu ana kadar `Reason for Absence` ve `ID` sütunları üzerinde devasa bir ön işleme yaptık. Bu başarıyı sabitlemek ve geri dönüşü olmayan hatalardan kaçınmak için verinin mevcut halini yeni bir değişkende yedekleyelim.

Python

```python
# Verinin temizlenmiş ve düzenlenmiş bu halini yedekliyoruz
df_reason_mod = df.copy()
```

###  Teknik Analiz ve Gözlem

Artık tablonun sol tarafında temizlenmiş neden gruplarını, sağ tarafında ise tahmin etmeye çalışacağımız çalışma saatlerini görüyorsun. `ID` tamamen yok oldu ve verimiz çok daha "anlamlı" bir hale geldi.

---

##  Sütunların Yeniden Sıralanması (Reordering)

Şu anki `df` yapımızda yeni oluşturduğumuz gruplar tablonun en sonunda yer alıyor. Ancak başlangıçtaki veri setimizde devamsızlık nedenleri en baştaydı. Bu yapıyı korumak ve analizi kolaylaştırmak için sütunları **reorganize** edeceğiz.

### 1. Yeni Sıralama Listesinin Oluşturulması

Önce mevcut sütun isimlerini bir listeye alacağız, ardından listenin sonundaki 4 yeni sütunu kesip başa yapıştıracağız.

Python

```python
# Mevcut sütun isimlerini kontrol edelim (Senin çıktındaki 0, 1, 2, 3 dahil)
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                'Daily Work Load Average', 'Body Mass Index', 'Education',
                'Children', 'Pets', 'Absenteeism Time in Hours', 
                'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']

# Sütunları istediğimiz sıraya göre manuel olarak diziyoruz
column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 
                          'Date', 'Transportation Expense', 'Distance to Work', 'Age',
                          'Daily Work Load Average', 'Body Mass Index', 'Education', 
                          'Children', 'Pets', 'Absenteeism Time in Hours']

# DataFrame'i bu yeni liste ile güncelliyoruz
df = df[column_names_reordered]
```

### 2. Teknik Detay: Indexing vs Reassigning

Burada yaptığımız ==`df = df[column_names_reordered]`== işlemi, Pandas'ta **"Column Indexing"** yoluyla veri çerçevesini yeniden yapılandırmaktır. Bu işlem sırasında veriler kaybolmaz, sadece bellekteki yerleşim sıraları değişir.

---

##  Checkpoint Stratejisi: `df_reason_mod`

"Checkpoint" (Kontrol Noktası) kavramı, büyük projelerde hayat kurtarır. ID'yi sildik, nedenleri grupladık ve sıralamayı bitirdik. Bu, "Reason for Absence" sütunuyla ilgili işimizin bittiği anlamına gelir.

### Neden Şimdi Yedekliyoruz?

- **Hata Payı:** Bir sonraki aşamada (Date/Tarih işleme) yapacağın bir hata, tüm tabloyu bozabilir.
    
- **Hız:** Eğer hata yaparsan, en başa dönüp tüm dummy değişken işlemlerini tekrar çalıştırmak yerine, bu yedeğe dönebilirsin.
    

Python

```python
# 'Reason' işlemleri bittiği için mevcut durumu yedekliyoruz
df_reason_mod = df.copy()

# İlk 5 satırı kontrol ederek sonucu teyit edelim
df.head()
```

---

###  Teknik Analiz ve Gözlem

Hücreyi çalıştırdığında;

- **Sol tarafta:** 1 ve 0 değerlerinden oluşan 4 ana neden grubunu,
    
- **Sağ tarafta:** Hedef değişkenin olan devamsızlık saatlerini görmelisin.
    

Bu işlemle birlikte projenin ilk büyük aşamasını (Reason for Absence Preprocessing) başarıyla tamamlandı.

---
Tarih (Date) sütunu, veri ön işlemenin en teknik ve dikkat gerektiren aşamalarından biridir. Bu bölümde sadece veri tipini değiştirmekle kalmayacak, aynı zamanda "zaman" kavramından modelimiz için anlamlı öznitelikler (features) üreteceğiz.

---

## Date Sütunu ve Timestamp Dönüşümü

Şu anda `df['Date']` sütunundaki veriler Python tarafından "string" (metin) olarak algılanıyor. Metin halindeki bir tarihle matematiksel bir analiz yapılamaz.

### 1. Veri Tipinin Analizi ve Hatanın Tespiti

Veri setindeki ilk değerin tipini kontrol ettiğimizde `<class 'str'>` sonucunu alırız. Bu verileri `Timestamp` (zaman damgası) objesine dönüştürmemiz gerekir.

**Kritik Uyarı:** ==`pd.to_datetime()`== fonksiyonunu doğrudan kullanmak, gün ve ayın yer değiştirmesi gibi ciddi hatalara yol açabilir. Örneğin; 07/10/2015 tarihi, format belirtilmezse Temmuz'un 10'u yerine Ekim'in 7'si olarak okunabilir.

### 2. Formatlı Dönüşüm ve Checkpoint Kullanımı

Hatalı bir dönüşüm yaparsak, daha önce oluşturduğumuz `df_reason_mod` yedeğine dönerek işlemi en baştan (hatasız) başlatabiliriz.

Python

```python
# Hatalı işlemi geri almak için yedeği yüklediğimizi varsayalım
df = df_reason_mod.copy()

# Doğru formatı (%d/%m/%Y) belirterek dönüşümü yapıyoruz
# %d: Gün, %m: Ay, %Y: 4 haneli Yıl
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Veri tipini kontrol edelim
df.info() # 'Date' artık datetime64[ns] olmalı
```

---

## Öznitelik Mühendisliği 1: Ay Değerini Çıkarma (Month Value)

Belirli aylarda (örneğin kış aylarında veya tatil dönemlerinde) devamsızlık oranlarının değişip değişmediğini ölçmek istiyoruz. Bunun için tarihten sadece "ay" bilgisini çekeceğiz.

### Teknik Uygulama: For Döngüsü ve Append

Döngü sayısını (`700`) manuel yazmak yerine `df.shape[0]` kullanarak dinamik hale getirmektir.

Python

```python
list_months = []

# Tüm satırlar boyunca döngü kuruyoruz
for i in range(df.shape[0]):
    # Her tarihin .month özelliğini alıp listeye ekliyoruz
    list_months.append(df['Date'][i].month)

# Yeni sütunu oluşturuyoruz
df['Month Value'] = list_months
```

---

##  Öznitelik Mühendisliği 2: Haftanın Günü (Day of the Week)

İnsanların Pazartesi sendromu mu yaşadığını yoksa Cuma günleri mi erken ayrılmak istediğini anlamak için haftanın günü bilgisine ihtiyacımız var.

### Teknik Uygulama: Fonksiyon ve `.apply()` Metodu

Pandas'ta bir işlemi tüm sütuna verimli bir şekilde uygulamak için bir fonksiyon tanımlayıp `.apply()` metodunu kullanmak en profesyonel yöntemdir.

- **Not:** `.weekday()` metodu haftanın günlerini **0 (Pazartesi)** ile **6 (Pazar)** arasında döndürür.
    

Python

```python
# Tarihi haftanın gününe (0-6) dönüştüren fonksiyon
def date_to_weekday(date_value):
    return date_value.weekday()

# Fonksiyonu tüm 'Date' sütununa uyguluyoruz
df['Day of the Week'] = df['Date'].apply(date_to_weekday)

# Sonucu kontrol edelim
df.head()
```

---

##  Teknik Değerlendirme

Şu an elimizde hem ay bilgisi hem de haftanın günü bilgisi var. Bu iki yeni değişken, regresyon modelimizin devamsızlık modellerini (patterns) anlamasına yardımcı olacak "bağımsız değişkenler" (features) olarak görev yapacak.

**Analiz Notu:** `Timestamp` objesinin içindeki saat bilgisinin `00:00:00` olması normaldir; çünkü orijinal verimizde saat bilgisi bulunmuyordu ve Python eksik kısmı otomatik olarak sıfırla doldurdu.

---
Veri ön işleme (preprocessing) maratonunun son düzlüğüne geldik! Bu aşamada, veri setindeki son birkaç sütunu (Education, Children, Pets) ele alacağız ve projenin geri kalanında "altın standart" olarak kullanacağımız nihai veri çerçevesini oluşturacağız.

---

##  Education, Children ve Pets: Kategorik vs. Sayısal Ayrımı

Bu üç sütun da tam sayılardan (integers) oluşur, ancak analitik anlamları tamamen farklıdır.

### 1. Children ve Pets (Sayısal Veri)

Bu sütunlar, bir kişinin kaç çocuğu veya evcil hayvanı olduğunu tam olarak gösterir. Buradaki "2" değeri "1"den büyüktür ve gerçek bir nicelik ifade eder. Bu nedenle bu sütunlara herhangi bir teknik müdahale yapmıyoruz; oldukları gibi modelde yer alacaklar.

### 2. Education (Kategorik Veri)

`Education` sütunundaki 1, 2, 3 ve 4 sayıları bir miktar değil, birer **eğitim seviyesi etiketidir**:

- **1:** Lise Mezunu (High School)
    
- **2:** Lisans (Graduate)
    
- **3:** Lisansüstü (Postgraduate)
    
- **4:** Doktora/Master (Master/Doctor)
    

#### Neden Gruplandırıyoruz?

`df['Education'].value_counts()` komutunu çalıştırdığımızda şu dağılımı görüyoruz:

- **Lise (1):** 583 kişi
    
- **Diğerleri (2, 3, 4):** Toplam 117 kişi
    

Veri setinin ezici çoğunluğu lise mezunudur. Diğer kategoriler (lisans, yüksek lisans vb.) istatistiksel olarak çok küçük gruplar oluşturduğu için bunları "Yüksek Eğitimli" adı altında tek bir grupta toplamak analizimizin doğruluğunu artıracaktır.

---

##  Uygulama: `.map()` Metodu ile Dönüşüm

Eğitim seviyelerini **Binary (İkili)** bir yapıya dönüştüreceğiz:

- **0:** Lise mezunları
    
- **1:** Lise üstü eğitim alan herkes
    

Python

```python
# Mevcut değerleri kontrol edelim
print(df['Education'].unique())

# Sözlük (Dictionary) yardımıyla değerleri eşleştiriyoruz
# Önemli: Sözlükteki anahtarlar (keys) sütundaki tüm mevcut değerleri kapsamalıdır.
df['Education'] = df['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})

# Sonucu kontrol edelim
print(df['Education'].unique()) # Çıktı: [0 1] olmalı
```

---

##  Nihai Kontrol ve Checkpoint (df_preprocessed)

Artık tüm sütunlarımızı (ID, Reason, Date, Education, vb.) makine öğrenmesi modeline girebilecek şekilde hazırladık. Bağımlı değişkenimiz olan `Absenteeism Time in Hours` üzerinde şu an bir işlem yapmıyoruz çünkü o bizim tahmin etmek istediğimiz "target" değerimizdir ve zaten sayısal olarak temiz bir durumdadır.

### Final Checkpoint: `df_preprocessed`

Bu aşama, projenin "Veri Ön İşleme" kısmının mühürlendiği andır.

Python

```python
# Tüm işlemler bitti, verinin en temiz halini yeni bir isimle kaydediyoruz
df_preprocessed = df.copy()

# Verinin son halini görelim
df_preprocessed.head(10)
```

---

##  Teknik Özet ve Sonuç

Çok kapsamlı ve yorucu bir veri ön işleme sürecini başarıyla tamamlandı. Bu süreçte şu işlemler gerçekleştirildi:

1. **ID:** Gereksiz bilgiyi eledik.
    
2. **Reasons:** 28 farklı nedeni 4 anlamlı gruba indirgedik ve dummy değişkenlere dönüştürdük.
    
3. **Date:** Metin halindeki tarihleri; Ay ve Haftanın Günü gibi sayısal özelliklere parçaladık.
    
4. **Education:** Dengesiz dağılan eğitim seviyelerini 0 ve 1 olarak sadeleştirdik.
    

Artık elimizde, istatistiksel analizlere ve modellemeye hazır, profesyonel standartlarda bir veri seti var.

---
---
Veri bilimi projelerinde **Preprocessing** (Ön İşleme) ve **Machine Learning** (Makine Öğrenmesi) bölümleri genellikle ayrı notebook dosyalarında veya farklı zamanlarda yapılır. Bu yüzden veriyi bir `.csv` olarak dışa aktarmak, emeğimizi sabitlemek için en kritik adımdır.

---

## Veriyi Dışa Aktarma (Exporting)

Aşağıdaki kodu çalıştırarak `df_preprocessed` değişkenini bilgisayarına bir dosya olarak kaydedelim:

Python

```python
# Veriyi CSV formatında kaydediyoruz
# index=False parametresi, Pandas'ın otomatik oluşturduğu satır numaralarını 
# fazladan bir sütun olarak kaydetmemesini sağlar.
df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)
```

###  Teknik Bir İpucu: Neden `index=False`?

Eğer bu parametreyi kullanmazsan, Pandas dosyanın en başına isimsiz bir sütun (0, 1, 2...) ekler. Dosyayı ileride tekrar okuduğunda (read_csv), Pandas bir kez daha indeks atayacağı için tablonun başında gereksiz iki tane sayı sütunu oluşur. Bu karmaşayı önlemek için `index=False` profesyonel bir standarttır.

---

##  Ön İşleme Bölümü Tamamlandı!

Şu ana kadar katettiğimiz yolun teknik özeti:

1. **Ham Veri (Raw Data):** Karmaşık, eksik ve analiz için ham durumdaydı.
    
2. **Temizlik (Cleaning):** ID'yi çıkardık, veriyi yedekledik.
    
3. **Dönüştürme (Transformation):** Devamsızlık nedenlerini grupladık, Tarihleri ayıkladık, Eğitimi binary (0-1) yaptık.
    
4. **Final:** Tüm bu işlemleri `df_preprocessed` adlı temiz bir dosyada topladık.
    

---

###  Bir Sonraki Büyük Adım

Artık elimizde "altın değerinde" temiz bir veri seti 
(`Absenteeism_preprocessed.csv`) var. 

1. **MySQL Entegrasyonu:** Bu temiz veriyi bir veritabanına aktarıp SQL sorgularıyla analiz etmek.
    
2. **Machine Learning:** Lojistik Regresyon modelini kurarak "Bir çalışan neden devamsızlık yapar?" sorusunu tahmin etmeye başlamak.
    

---
---
### 1. Dosyayı Fiziksel Olarak Taşıma

Bu işlem iki şekilde yapabilir:

- **A Yöntemi (Manuel):** Bilgisayarındaki dosya gezginini (Windows Explorer veya macOS Finder) aç. `Notebooks` klasörüne gir, `Absenteeism_preprocessed.csv` dosyasını sağ tıklayıp **Kes** de. Ardından bir üst klasöre çıkıp `Data` klasörüne gir ve **Yapıştır** de.
    
- **B Yöntemi (Python ile):** Eğer her şeyi notebook içinden halletmek istersen, şu kod bloğunu çalıştırabilirsin:
    

Python

```python
import shutil
import os

# Mevcut konum ve hedef konum
kaynak = 'Absenteeism_preprocessed.csv'
hedef = '../Data_name/Absenteeism_preprocessed.csv'

# Dosyayı taşı (Eğer dosya Notebooks içindeyse)
if os.path.exists(kaynak):
    shutil.move(kaynak, hedef)
    print("Dosya başarıyla Data_name klasörüne taşındı.")
else:
    print("Dosya bu dizinde bulunamadı; muhtemelen zaten taşınmış.")
```

---

### 2. Kod Yapısını Güncelleme (Relative Path Mantığı)

Dosyayı taşıdıktan sonra, yeni açacağın "Machine Learning" notebook'unda veriyi okuturken izleyeceğin teknik mantık şudur:

Sizin notebook'unuz `Notebooks` klasörünün içinde olduğu için Python'a şu komutu veriyoruz:

1. **`..`**: "Bulunduğun klasörden (`Notebooks) bir üst dizine (`Absenteeism_Case`) çık."
    
2. **`/Data_name/`**: "Oradaki `Data_name` klasörüne gir."
    
3. **`Absenteeism_preprocessed.csv`**: "Hedef dosyayı oku."
    

**Yeni Başlangıç Code:**

Python

```python
import pandas as pd

# Veriyi merkezi veri deposundan çekiyoruz
df = pd.read_csv('../Data_name/Absenteeism_preprocessed.csv')

# İlk 5 satırı kontrol ederek bağlantıyı teyit edelim
df.head()
```

---

###  Neden Bu Kadar Önemli? (Teknik Vurgu)

Bu yapıya **"Modüler Dosya Mimarisi"** denir. Yarın bir gün veri setin güncellendiğinde (örneğin yeni veriler geldiğinde), sadece `Data_name` içindeki dosyayı değiştirmek yeterli olur. Tüm analiz ve modelleme notebook'ların otomatik olarak yeni veriyi görmeye başlar.

---
---

##  Yeni Bir Notebook Başlatma

Veri Ön İşleme (Preprocessing) notebook'unu kaydedip kapatabilirsin. Şimdi `Notebooks_name` klasörü içinde yeni bir dosya oluşturulabilir. **`Machine_Learning.ipynb`** olabilir.

## Veriyi Yeni Konumundan Okuma

Yeni açılan notebook'un ilk hücresinde, az önceki **"Relative Path" (Göreceli Yol)** mantığını kullanarak veriyi içeri aktaralım.

Python

```python
import pandas as pd

# '../' ifadesi bir üst klasöre çıkar, 'Data_name/' ise veri klasörüne girer
df = pd.read_csv('../Data_name/Absenteeism_preprocessed.csv')

# Bağlantının kurulduğunu teyit edelim
df.head()
```

---

## Lojistik Regresyon İçin "Target" (Hedef) Oluşturma

Makine öğrenmesi kısmındaki ilk görevimiz, tahmin etmek istediğimiz hedefi belirlemektir. Bizim durumumuzda bu, `Absenteeism Time in Hours` (Saat cinsinden devamsızlık) sütunudur. Ancak Lojistik Regresyon için sayısal saatler yerine **"Devamsız" (1)** veya **"Değil" (0)** şeklinde iki kategoriye ihtiyacımız var.

### Teknik Yaklaşım: Medyan Değeri Kullanma

Devamsızlığın "çok" veya "az" olduğunu belirlemek için en adil yöntem **medyan (ortanca)** değerini kullanmaktır.

1. **Medyanı Bul:** Veri setindeki devamsızlık saatlerinin tam ortasını bulacağız.
    
2. **Eşik Değer (Threshold):** Medyanın üzerinde devamsızlık yapanlara **1**, altında veya medyanda kalanlara **0** diyeceğiz.
    

Python

```python
# 1. Medyanı bulalım
median_value = df['Absenteeism Time in Hours'].median()

# 2. 'targets' adında yeni bir seri oluşturalım
# Medyandan büyükse 1, değilse 0 (Lojistik Regresyon hazırlığı)
targets = [1 if x > median_value else 0 for x in df['Absenteeism Time in Hours']]

# 3. Bu listeyi ana df'e yeni bir sütun olarak ekleyelim
df['Excessive Absenteeism'] = targets

# Kontrol edelim
df.head()
```

---

## Analitik Değerlendirme

Neden medyanı kullandık? Eğer ortalama (mean) kullansaydık, çok uç değerler (örneğin 120 saat devamsızlık yapan tek bir kişi) sonucu saptırabilirdi. Medyan kullanarak veri setini tam ortadan ikiye bölmüş olduk ve bu sayede **dengeli bir hedef (balanced target)** elde ettik.

---
---

# Machine Learning

---
Jupyter Notebook üzerinde çok fazla veri çerçevesi (DataFrame) görüntülemek bir süre sonra tarayıcıda yavaşlamalara (lag) neden olabilir; bu yüzden veriyi dışa aktarıp yeni bir dosyada çalışmak profesyonel bir standarttır.

Ayrıca gerçek dünyadaki veri bilimi projelerinde, veri ön işleme (preprocessing) ve modelleme aşamalarının farklı kişiler tarafından yürütülmesi veya aynı veri seti üzerinde farklı modellerin (Lojistik Regresyon, Random Forest vb.) ayrı notebook'larda test edilmesi çok yaygın bir durumdur.

---

## Makine Öğrenmesi Başlangıç: Kütüphaneler ve Veri Yükleme

Yeni notebook'unda (`Absenteeism_Machine_Learning.ipynb`) ilk adım, temel araçlarımızı kuşanmak ve temizlediğimiz veriyi çağırmaktır.

### 1. Kütüphanelerin İçe Aktarılması

Her yeni notebook'ta olduğu gibi, `numpy` ve `pandas` kütüphanelerini import ederek başlıyoruz.

Python

```python
import pandas as pd
import numpy as np
```

### 2. Verinin Yüklenmesi

Dosyayı `Data_0` klasörüne taşıdığımız için "relative path" (göreceli yol) kullanarak veriyi içeri aktarıyoruz.

Python

```python
# Bir üst klasöre çıkıp Data_0 klasöründeki dosyayı okuyoruz
data_preprocessed = pd.read_csv('../Data_0/Absenteeism_preprocessed.csv')

# Veriye hızlıca bir göz atalım
data_preprocessed.head()
```

---

### 1. Neden "True/False" Olarak Kaydedildi?

Pandas, son sürümlerinde `get_dummies` veya `concat` işlemleri sırasında bellek verimliliği sağlamak için kategorik verileri otomatik olarak `bool` (Mantıksal: True/False) tipinde tutabiliyor. `to_csv` yaptığında bu veriler metin olarak kaydedilir.

### 2. Çözüm: "Machine Learning" Notebook'unda Düzeltme

Daha önce hazırlanan `preprocessing` notebook'una dönmeye gerek yok. Yeni açılan **Machine Learning** notebook'unda veriyi okuduktan hemen sonra şu kodu çalıştırmak yeterlidir:

Python

```python
# Veriyi okuduktan hemen sonra boolean (True/False) sütunları tam sayıya (1/0) çeviriyoruz
df[['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']] = df[['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']].astype(int)

# Kontrol edelim
df.head()
```

Bu işlem, veriyi hafızada (RAM) anında günceller ve modelin için hazır hale getirir. Yani fiziksel `.csv` dosyasını değiştirmekle uğraşmak yerine, veriyi **"çalışma anında"** (on-the-fly) düzeltmiş oluyoruz. Bu çok daha profesyonel ve hızlı bir yaklaşımdır.

---

###  Bir Sonraki Adım: Hedef (Target) Oluşturma ve Sütun Seçimi

Ders notlarına sadık kalarak, şimdi tahmin etmek istediğimiz hedefi netleştirelim. `Absenteeism Time in Hours` sütununu kullanarak ikili (binary) sınıflandırma hedefimizi oluşturalım.

1. **Medyanı Belirle:** Veri setindeki orta noktayı bulalım.
    
2. **Sınıflandır:** Medyandan büyük olanlar `1` (Aşırı Devamsız), küçük veya eşit olanlar `0` (Normal) olsun.
    
3. **Sütunu Düşür:** Orijinal saat sütununu artık silebiliriz.
    

Python

```python
# 1. Medyanı bulalım
median_value = df['Absenteeism Time in Hours'].median()

# 2. Hedefleri (Targets) np.where ile hızlıca oluşturalım
# Not: Medyandan yüksekse (>) 1, aksi halde 0
df['Excessive Absenteeism'] = np.where(df['Absenteeism Time in Hours'] > median_value, 1, 0)

# 3. Orijinal saat sütununa artık ihtiyacımız yok, onu kaldıralım
df_with_targets = df.drop(['Absenteeism Time in Hours'], axis=1)

# Kontrol: En sağda yeni sütunumuz görünüyor mu?
df_with_targets.head()
```

---
## Tahmin Modelimizin Mantığı ve Beklentiler

Bu projede **Lojistik Regresyon** (Logistic Regression) kullanacağız. Amacımız; bir çalışanın nedenlerini, yaşını, eğitimini, çocuk/evcil hayvan sayısını ve iş yükünü kullanarak devamsızlık yapıp yapmayacağını tahmin etmek.

Ders notlarındaki teknik öngörülere göre modelimizden şunları bekliyoruz:

- **En Güçlü Belirteç (Predictor):** "Reason for Absence" (Devamsızlık Nedeni) muhtemelen en önemli sütun olacak.
    
- **İş Yükü (Workload):** Kişi ne kadar meşgulse, işi aksatma ihtimalinin o kadar düşeceği varsayılıyor.
    
- **Ailevi Faktörler:** Çocuklar ve evcil hayvanlar, özellikle hastalık durumlarında doktora gitmek ve geri dönmek çok zaman aldığı için devamsızlığı doğrudan etkileyebilir.
    

> **Not:** Regresyon modellerinin en güzel yanı, hangi değişkenin tahmin üzerinde ne kadar "ağırlığı" (importance) olduğunu bize matematiksel olarak göstermesidir. Hangi değişkenlerin gerçekten önemli olduğunu, modelin katsayılarını incelediğimizde net bir şekilde göreceğiz.

---
Bu yaklaşım, veri biliminde **"Parametrelendirme" (Parametrization)** olarak adlandırılan ve kodun esnekliğini, doğruluğunu ve okunabilirliğini artıran çok kritik bir teknik detaydır.

Kodda doğrudan **3** rakamının olmamasının temel sebebi, bu sayının veriye bağlı bir "sonuç" olmasıdır; yani biz **3**'ü hedeflemiyoruz, biz **medyanı** hedefliyoruz ve o anki veri setinde medyan **3**'e denk geliyor.

---

###  Teknik Analiz: Neden 3 Yazmıyoruz?

- **Veri Bağımlılığı:** Eğer doğrudan `> 3` yazarsak, gelecekte veri seti güncellendiğinde veya başka bir şirketin verisiyle çalışıldığında medyan **4** veya **5** olursa, kodumuz yanlış sonuçlar üretir. Fonksiyon kullanmak, kodun her türlü veriye uyum sağlamasını sağlar.
    
- **Sayısal Kararlılık (Numerical Stability):** Lojistik regresyonun sağlıklı çalışması için 0 ve 1 sınıflarının dengeli olması gerekir. Medyanı kullanmak, veriyi matematiksel olarak her zaman en yakın orta noktadan bölmeyi garanti eder.
    
- **Hata Payını Azaltma:** İnsan eliyle yazılan sabit sayılar (hard-coding), projeler büyüdükçe unutulabilir ve büyük hatalara yol açabilir. Parametrik kodlama bu riski sıfıra indirir.
    

---

###  Geliştirilmiş ve Yorumlanmış Kod Bloğu

Python

```python
import numpy as np

# 1. Eşik Değerin (Cutoff Line) Belirlenmesi
# Kodda '3' yazmıyoruz; çünkü 3, verinin medyanıdır. 
# Fonksiyon kullanarak kodu dinamik hale getiriyoruz.
median_value = df['Absenteeism Time in Hours'].median()

# 2. Hedef Değişkenlerin (Targets) Oluşturulması
# np.where tıpkı Excel'deki IF gibidir: (Şart, Doğruysa, Yanlışsa)
# Eğer bir çalışan medyandan (yani 3 saatten) FAZLA devamsızlık yaptıysa '1' (Aşırı),
# aksi takdirde '0' (Normal) değerini atıyoruz.
targets = np.where(df['Absenteeism Time in Hours'] > median_value, 1, 0)

# 3. Yeni Sütunun Veri Çerçevesine Eklenmesi
# Bu işlemle sürekli sayısal veriyi (continuous) ikili sınıfa (binary) dönüştürmüş olduk.
df['Excessive Absenteeism'] = targets

# 4. Kontrol (Checkpoint)
# İlk 5 satıra bakarak eşleşmeyi görelim
df[['Absenteeism Time in Hours', 'Excessive Absenteeism']].head()
```

---

### Veri Dengesi Kontrolü

Kodda **3** yerine medyanı kullanmamızın meyvesini burada topluyoruz. Veri setinin ne kadar dengeli olduğunu şu formülle teyit edebiliriz:

Python

```python
# 1'lerin oranını bulalım (Toplam 1 sayısı / Toplam Satır Sayısı)
# Sonucun 0.45 ile 0.55 arasında çıkması, modelin eğitimi için idealdir.
balance_check = targets.sum() / targets.shape[0]
print(f"Hedeflerin Dağılım Oranı: {balance_check:.2f}")
```

Bu dengeyi kontrol ettikten sonra, artık orijinal saat sütununa ihtiyacımız kalmayacak. Gereksiz sütunları `.drop()` metoduyla temizleyip **`data_with_targets`** adında yeni bir checkpoint oluşturulur.

---
 **Standartlaştırma (Standardization)** adımını teknik derinliğiyle verilere (700 satır, 15 özellik) uygulayalım.

---
 %% Bu sayı (**0.4557**), veri setinin yaklaşık **%46**'sının "1" (Aşırı Devamsız), geri kalan **%54**'ünün ise "0" (Normal) olduğunu gösteriyor.

bu sonuç, modelimizin eğitimi için **"altın oran"** diyebileceğimiz bir dengeyi ifade eder. Eğer bu oran çok düşük (örneğin 0.05) veya çok yüksek (örneğin 0.95) çıksaydı, modelimiz veriyi öğrenmek yerine sadece baskın olan sınıfı tahmin etmeye çalışırdı (bias). Medyan kullanımı sayesinde bu sorunu en baştan çözmüş olduk. %%

---

## Sütun Temizliği ve Checkpoint

Artık hedef değişkenimiz (`Excessive Absenteeism`) hazır olduğuna göre, onu türettiğimiz ham veri sütununu (`Absenteeism Time in Hours`) veri setinden çıkarmalıyız. Bu, modelin "kopya çekmesini" (Data Leakage) önlemek için teknik bir zorunluluktur.

### 1. Sütunu Düşürme ve Yeni Değişken Atama

Bu noktada yeni bir kontrol noktası (checkpoint) oluşturuyoruz:

Python

```python
# Orijinal saat sütununu çıkararak yeni bir değişken oluşturuyoruz
data_with_targets = df.drop(['Absenteeism Time in Hours'], axis=1)

# Bu işlemin yeni bir nesne oluşturup oluşturmadığını (farklı bellek adresi) kontrol edelim
# Sonucun False çıkması bir 'checkpoint' oluşturduğumuzu kanıtlar
data_with_targets is df
```

### 2. Girdileri (Inputs) Ayırma

Şimdi, modelin tahmin yapmak için kullanacağı özellikleri (**Inputs**) seçelim. Burada en kullanışlı ipuçlarından biri olan **negatif indeksleme** (`-1`) yöntemini kullanacağız. Sütun saymakla uğraşmadan, en sondaki hedef sütununu dışarıda bırakacağız.

Python

```python
# Tüm satırları seç (:), sütunlardan ise sonuncusu hariç hepsini al (:-1)
unscaled_inputs = data_with_targets.iloc[:, :-1]

# Sonucu kontrol edelim
unscaled_inputs.head()
```

---
### Negatif İndeksleme

 `data_with_targets.iloc[:, :14]` komutu "statik" bir komuttur; yani sütun sayısına güvenir. 

Python

```python
# Tüm satırları seç (:)
# Sütunlardan ise baştan başla ve en sondaki hedef sütununu hariç tut (:-1)
unscaled_inputs = data_with_targets.iloc[:, :-1]
```

- veri seti 16 sütunlu ise, `:-1` komutu ilk 15 sütunu (girdileri) alır ve 16. sütun olan hedefi (`Excessive Absenteeism`) dışarıda bırakır.
    
- Bu sayede sütun sayın kaç olursa olsun, hedef sütunun en sondaysa kodun her zaman kusursuz çalışır.
    
---
### Uygulama Adımı

Python

```python
# Girdileri seçiyoruz
unscaled_inputs = data_with_targets.iloc[:, :-1]

# Şeklini kontrol edelim (700, 15)
print(unscaled_inputs.shape)
```

---

##  Veriyi Standartlaştırma (Standardization)

Ders notlarında vurgulandığı gibi; her bir değişkenin (sütunun) ortalamasını çıkarıp, sonucun standart sapmaya bölünmesi işlemidir. Matematiksel olarak formül şöyledir:

$$z = \frac{x - \mu}{\sigma}$$

Burada $\mu$ ortalamayı, $\sigma$ ise standart sapmayı temsil eder.

### 1. StandardScaler Nesnesini Tanımlama

Önce boş bir ölçeklendirici oluşturuyoruz. Bu nesne şu an "boş" bir makine gibidir.

Python

```python
from sklearn.preprocessing import StandardScaler

# Boş bir StandardScaler nesnesi oluşturuyoruz
absenteeism_scaler = StandardScaler()
```

### 2. Mekanizmayı Hazırlama (Fit)

Bu adımda `absenteeism_scaler`, senin `unscaled_inputs` içindeki her bir sütunun ortalamasını ve standart sapmasını hesaplar ve bu bilgiyi hafızasına (nesnenin içine) kaydeder. Artık nesne boş değildir!

Python

```python
# Her özelliğin ortalamasını ve standart sapmasını hesaplıyoruz
absenteeism_scaler.fit(unscaled_inputs)
```

### 3. Dönüştürme İşlemi (Transform)

Hesaplanan bu bilgileri kullanarak veriyi gerçekten dönüştürdüğümüz aşamadır.

Python

```python
# Veriyi matematiksel olarak dönüştürüyoruz
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
```

---

##  Sonuçları İnceleme

Bu işlemden sonra verinin yapısı değişir. Hadi kontrol edelim:

Python

```python
# Dönüştürülmüş veriyi görelim (Bir NumPy Array olacaktır)
print(scaled_inputs)

# Şeklini kontrol edelim
print(scaled_inputs.shape)
```

**Analiz:**

- **Görünüm:** Artık o bildiğimiz Pandas tablosu yerine, tüm sayıların birbirine yakın olduğu (genellikle -3 ile +3 arasında) bir **NumPy Array** görünecektir.
---

###  Neden Önemli?

`absenteeism_scaler` nesnesini saklamamız çok kritik. Yarın bir gün yeni bir çalışan verisi geldiğinde, o veriyi de **aynı ortalama ve standart sapma** değerleriyle ölçeklendirmemiz gerekecek. Aksi takdirde modelimiz yanlış tahminler yapar.

----
makine öğrenmesinin en kritik aşamalarından biri olan **"Overfitting" (Aşırı Öğrenme)** ile mücadeleyi ve veriyi bölme stratejisidir. Modelin veriyi ezberlemesini önlemek için verinin bir kısmını "sınav sorusu" olarak saklayacağız.

---

### Teorik Vurgu: Neden Bölüyoruz?

- **Overfitting:** Modelin eğitim verisindeki gürültüyü bile öğrenip, gerçek hayattaki yeni verilerde çuvallamasıdır.
    
- **Çözüm:** Veriyi **Eğitim (Train)** ve **Test** olarak ikiye ayırmak.
    
- **Shuffling (Karıştırma):** Verinin dizilişinden kaynaklanan bağımlılıkları (örneğin haftanın günlerinin sırayla gitmesi) yok etmek için veriyi karıştırıyoruz.
    
- **Random State:** Her çalıştırdığımızda aynı "rastgele" sonucu alarak analizimizin tutarlı kalmasını sağlıyoruz.
    

---

###  Uygulama: `train_test_split`

**80-20** kuralını ve `random_state = 20` parametresini kullanarak kodu yazalım:

Python

```python
# 1. Gerekli kütüphaneyi içe aktarıyoruz
from sklearn.model_selection import train_test_split

# 2. Veriyi bölüyoruz
# inputs: scaled_inputs (700, 14)
# targets: targets (700,)
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, 
                                               train_size = 0.8, 
                                               random_stat=20, 
                                               shuffle =True)

# 3. Şekilleri (Shapes) kontrol edelim
print("Eğitim Girdileri:", x_train.shape, "Eğitim Hedefleri:", y_train.shape)
print("Test Girdileri:", x_test.shape, "Test Hedefleri:", y_test.shape)
```

---

### Sonuç Analizi 

Kodun çıktısı 

- **Toplam Veri:** 700 satır.
    
- **Eğitim Seti (%80):** $700 \times 0.8 = \mathbf{560}$ gözlem.
    
- **Test Seti (%20):** $700 \times 0.2 = \mathbf{140}$ gözlem.
---

### Teknik Checkpoint


1. **`x_train`**: Modelin ders çalışacağı sorular.
    
2. **`y_train`**: Bu soruların doğru cevapları.
    
3. **`x_test`**: Modelin daha önce hiç görmediği sınav soruları.
    
4. **`y_test`**: Sınavın gerçek cevap anahtarı (başarıyı ölçmek için kullanacağız).
    

Veriyi başarıyla bölündü ve artık "algoritmanın kalbine" yani **Logistic Regression** modelini eğitme (fit) aşamasına geçmeye hazırız!

---
### Modeli Kurma ve Eğitme (The Training)

Önce gerekli araçları hazırlayalım ve modelimizi eğiterek veriler arasındaki bağlantıyı (matematiksel regresyonu) kurmasını sağlayalım.

Python

```python
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Modeli tanımlıyoruz (Boş bir kutu gibi düşünebilirsin)
reg = LogisticRegression()

# Modeli eğitiyoruz (Öğrenme burada gerçekleşiyor)
reg.fit(x_train, y_train)
```

### Başarıyı Ölçme (Evaluating Accuracy)

Modelin ne kadar iyi öğrendiğini anlamak için en hızlı yöntem `score` metodudur.  Verilerle bu sonucun **0.80 (%80)** civarında çıkmasını bekliyoruz. Bu, modelin devamsızlık vakalarının %80'ini doğru sınıflandırdığı anlamına gelir.

Python

```python
# Modelin otomatik skorunu alalım
accuracy_score = reg.score(x_train, y_train)
print(f"Model Doğruluk Oranı: {accuracy_score}")
```

---

### Manuel Doğruluk Kontrolü (The Manual Way)

Mantığını kavramak için bu %80'i elimizle hesaplayalım. Bu işlem üç aşamadan oluşur:

1. **Tahmin Et:** Modelin eğitim verilerine bakarak sonuç üretmesini sağla.
    
2. **Karşılaştır:** Modelin tahminlerini (`model_outputs`), gerçek cevap anahtarıyla (`y_train`) kıyasla.
    
3. **Hesapla:** Doğru tahmin sayısını toplam veri sayısına böl.
    

Python

```python
import numpy as np

# 1. Tahminleri alıyoruz
model_outputs = reg.predict(x_train)

# 2. Tahminler ile gerçekleri karşılaştırıyoruz (True/False dizisi)
comparisons = (model_outputs == y_train)

# 3. Manuel skoru hesaplıyoruz (Doğru sayısını toplam sayıya bölüyoruz)
manual_accuracy = np.sum(comparisons) / comparisons.shape[0]

print(f"Manuel Hesaplanan Doğruluk: {manual_accuracy}")
```

---

### Neden Manuel Hesaplama Yaptık?

- **Derin Anlayış:** `score` metodunun arkasında dönen "tahmin et ve kıyasla" mantığını gördük.
    
- **İleri Seviye Hazırlık:** İleride (coefficients) ve olasılıkları (probabilities) incelerken bu karşılaştırma mantığına çok ihtiyacımız olacak.
    

Model %80 başarıyla çalışıyor. 

---

##  Lojistik Regresyon Özet Tablosu Hazırlama Rehberi

Bu aşamada amacımız, modelin "kara kutusunu" açıp hangi değişkenin (Yaş, Nedenler, Masraflar) devamsızlığı ne kadar etkilediğini görmektir.

### 1. Modelin Temel Taşlarını Çıkarma

Öncelikle modelin katsayılarını (weights) ve kesişim noktasını (bias) alıyoruz. Unutma; `sklearn` çıktıları NumPy dizisi olarak verir, biz bunları anlamlı bir tabloya dönüştüreceğiz.

Python

```python
import pandas as pd
import numpy as np

# Özellik isimlerini orijinal veri çerçevesinden alıyoruz (scaled_inputs bir array olduğu için)
feature_name = unscaled_inputs.columns.values

# Yeni bir özet tablo oluşturuyoruz
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)

# Katsayıları (Weights) ekliyoruz. 
# Transpose (devriğini alma) yapıyoruz çünkü coef_ satır vektörü olarak gelir.
summary_table['Coefficient'] = np.transpose(reg.coef_)

# Tabloyu görüntüle (Şu an 0'dan 13'e kadar 14 satır var)
summary_table
```

### 2. Intercept (Kesişim) Ekleme ve İndeks Düzenleme

Intercept'i tablonun en başına (0. indekse) yerleştireceğiz.

Python

```python
# Tüm mevcut indeksleri 1 artırarak 0. indeksi boşaltıyoruz
summary_table.index = summary_table.index + 1

# Boşalan 0. indekse Intercept (Bias) değerini yerleştiriyoruz
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

# Tabloyu indekslerine göre sıralayarak 0'dan 14'e kusursuz dizilimi sağlıyoruz
summary_table = summary_table.sort_index()
```

### 3. Odds Ratio (Olasılık Oranları) Hesaplama

Katsayılar logaritmik değerler olduğu için onları yorumlamak zordur.

==$e^{Coefficient}$== 

işlemini yaparak gerçek dünya etkilerini (Odds Ratio) buluyoruz.

Python

```python
# Üssel fonksiyon kullanarak Odds Ratio hesaplıyoruz
summary_table['Odds ratio'] = np.exp(summary_table.Coefficient)

# Tabloyu en önemli (etkisi en yüksek) değişkenler en üstte olacak şekilde sıralayalım
summary_table = summary_table.sort_values('Odds ratio', ascending=False)

summary_table
```

---

##  Sonuçların Teknik Analizi (Öğretici Notlar)

Tablo incelediğinde şu 3 temel kurala dikkat:

1. **Odds Ratio > 1:** Değişken arttıkça devamsızlık olasılığı **artar**.
    
    - _Örnek:_ `Reason_1` (Zehirli katsayı) en yüksek orana sahipse, bu nedene sahip kişilerin devamsızlık yapma ihtimali çok yüksektir.
        
2. **Odds Ratio < 1:** Değişken arttıkça devamsızlık olasılığı **azalır**.
    
    - _Örnek:_ `Pet` veya `Education` katsayısı negatifse, odds ratio 1'den küçük çıkar. Bu, "Daha çok evcil hayvanı olanlar daha az devamsızlık yapıyor" demektir.
        
3. **Odds Ratio ≈ 1 (veya Coef ≈ 0):** Bu değişken model için **etkisizdir**.
    
    - _Örnek:_ `Daily Work Load Average` (0.97). Bu değişkenin modelde olup olmaması sonucu değiştirmez.

---

### ⚠️ Önemli Hatırlatma

Eğer bu kodları Jupyter'da çalıştırdıktan sonra tabloyu tekrar düzenlemek istersen, **tüm hücreyi en baştan çalıştırdığından emin ol.** Sadece `index + 1` satırını tekrar çalıştırırsan indeksler 15, 16, 17 diye kaymaya devam eder.

---
İşte o meşhur "pürüzün" çözümü!:
Veri biliminde bir **Kukla Değişkeni (Dummy Variable)** standartlaştırmak, onun tüm anlamını yitirmesine neden olur.

Eğer bir kukla değişken 0 ve 1 olarak kalsaydı, "Bu neden varsa devamsızlık olasılığı 7 kat artar" diyebilirdik. Ama onu standartlaştırdığımızda $-0.5$ ve $1.7$ gibi sayılara dönüştüğü için "1 birimlik artış" artık gerçek hayatta hiçbir şeye karşılık gelmez hale geldi.

Bu sorunu çözmek için dersin en kilit "nifty tool"u olan **Custom Scaler (Özel Ölçekleyici)** yapısını kullanacağız.

---

### Custom Scaler Sınıfını Tanımlama!!!

Bu sınıf, `StandardScaler`'ın bir kopyası gibi çalışır ancak sadece bizim seçtiğimiz sütunları ölçeklendirir, diğerlerini (0 ve 1'leri) olduğu gibi bırakır.

Python

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

# Bu sınıf SKLearn'ün standart ölçekleyicisini özelleştirir
class CustomScaler(BaseEstimator, TransformerMixin): 
    
    def __init__(self, columns):
        self.scaler = StandardScaler()
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
```

---

###  Adım 2: Ölçeklenecek Sütunları Seçme

Şimdi, hangi sütunların "sayısal/sürekli" olduğunu ve hangilerinin "kukla/dummy" olduğunu belirlememiz gerekiyor. **Reason** ve **Education** sütunlarını ölçekleme dışında tutacağız.

Python

```python
# Tüm sütun isimlerine bakalım
unscaled_inputs.columns.values

# Ölçeklemek İSTEDİĞİMİZ sütunları seçiyoruz (Dummies ve Education hariç)
columns_to_scale = [
    'Month Value', 'Day of the Week', 'Transportation Expense', 
    'Distance to Work', 'Age', 'Daily Work Load Average', 
    'Body Mass Index', 'Children', 'Pet'
]

# Custom Scaler nesnemizi sadece bu sütunlarla eğitiyoruz
absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)

# Veriyi dönüştürelim
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)

# Kontrol: Şimdi dummy sütunlarının (0, 1) bozulmadığını göreceksin
scaled_inputs
```

---

### Adım 3: Modeli Yeniden Eğitme ve Yeni Özet Tablo

Artık verimiz yorumlanabilir (interpretable) bir formda. Modeli bu yeni `scaled_inputs` ile tekrar eğitelim ve katsayıları inceleyelim.

Python

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Veriyi tekrar böl (Yeni ölçeklenmiş verilerle)
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=20)

# 2. Modeli eğit
reg = LogisticRegression()
reg.fit(x_train, y_train)

# 3. Yeni Özet Tabloyu Oluştur (O meşhur temiz betiğimizle)
feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()

# 4. Odds Ratio hesapla ve yorumlamaya başla!
summary_table['Odds ratio'] = np.exp(summary_table.Coefficient)
summary_table.sort_values('Odds ratio', ascending=False)
```

---

###  Neyi Değiştirdik? (Analiz)

Bu düzeltmeden sonra şunları fark edeceksin:

1. **Doğruluk (Accuracy):** Belki %1'den az bir düşüş yaşanabilir. Bu çok normal çünkü 5 değişkenin ölçeğini değiştirdik.
    
2. **Yorumlanabilirlik (Interpretability):** Artık `Reason_1`'in karşısındaki **Odds Ratio** bize net bir şey söyler: "Eğer sebep Zehirlenme/Enfeksiyon gibi (Reason 1) ise, devamsızlık yapma olasılığı **X kat** artar."
    
3. **Hukuki/İdari Netlik:** Eskiden $-0.57$ olan dummy değerleri artık **1** (var) veya **0** (yok) olarak temsil ediliyor.
    

Artık model sadece "tahmin etmesini" değil, "neden tahmin ettiğini" de açıklayabiliyor.

---

### 1. Neden Bu Kod Değişmeli? (Teknik Analiz)

standart kod şu işlemi yapar:

$$z = \frac{x - \mu}{\sigma}$$

Eğer $x$ bir dummy (0 veya 1) ise, sonuç artık 0 veya 1 değildir. Bu da bizim **Odds Ratio** (Olasılık Oranı) yorumumuzu bozar. Anlatıcının "yeni araç" (nifty tool) dediği şey, bu işlemi sadece seçtiğimiz sütunlara uygulamaktır.

---

### 2. Uygulayacağımız "Yeni ve Doğrulanmış" Betik

 Bu kod, `StandardScaler`'ın tüm yeteneklerine sahiptir ama seçici davranır.

Python

```python

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

class CustomScaler(BaseEstimator, TransformerMixin): 
    
    # parametrelerle __init__ metodu
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        # StandardScaler'a bu parametreleri paslıyoruz
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    # Eğitme metodu
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    # Dönüştürme metodu
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        # Sadece seçtiğimiz sütunları ölçeklendiriyoruz
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        # Seçilmeyen (Dummy) sütunları olduğu gibi bırakıyoruz
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        # Her iki parçayı birleştirip orijinal sütun sırasına diziyoruz
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

```

---

### 3. Hangi Sütunları Ölçekleyeceğiz?

"dummy'leri hariç tutacağız" kısım burası. `unscaled_inputs` içindeki tüm sütunlara bakıyoruz ve sadece sayısal olanları seçiyoruz.

Python

```python
# Tüm sütun listesini alalım
all_columns = unscaled_inputs.columns.values

# Dummy (Reason_1, 2, 3, 4 ve Education) haricindeki sayısal sütunları seçiyoruz
# Not: Bu liste değişebilir, tipik liste şudur:
columns_to_scale = [
    'Month Value', 'Day of the Week', 'Transportation Expense', 
    'Distance to Work', 'Age', 'Daily Work Load Average', 
    'Body Mass Index', 'Children', 'Pet'
]

# Yeni ölçekleyiciyi tanımlıyoruz
absenteeism_scaler = CustomScaler(columns_to_scale)

# Modeli eğitiyoruz (fit)
absenteeism_scaler.fit(unscaled_inputs)

# Dönüştürme işlemini yapıyoruz (transform)
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)

# Sonucu kontrol et: Dummy'lerin 0 ve 1 kaldığını göreceksin!
scaled_inputs
```

---

### Karar ve Teyit: Ne Değişti?

1. **Kod Uyumu:**  `CustomScaler` sınıfı, "pre-prepared code" dediği yapının ta kendisidir. `fit` ve `transform` metodları `StandardScaler` ile aynı şekilde çalışır, böylece kodun geri kalanını (splitting, modeling) bozmazsın.
    
2. **Hukuki/Mantıksal Tutarlılık:** Artık `Reason_1`'in katsayısı (coefficient) doğrudan "neden varsa/yoksa" durumunu temsil edecek.
    
3. **İndeks Güvenliği:** Bu kod bloğunu tek bir hücrede çalıştırdığında, önceki "indeks kayma" sorunu da yaşanmaz çünkü her şey sıfırdan ve doğru sırayla hesaplanır.
    
**Bu değişikliği yaptıktan sonra modelin doğruluğu (accuracy) çok küçük bir miktar düşebilir ama**: **"Yorumlanabilirliği kazandık!"**

---

##  Katsayıların ve Önem Derecelerinin Analizi

Lojistik regresyonda katsayı sıfırdan ne kadar uzaksa (pozitif veya negatif), değişkenin tahmin üzerindeki etkisi o kadar büyüktür.

### 1. En Güçlü Belirleyiciler (Yüksek Etki)

Tablonun en üstünde yer alan ve olasılığı ciddi şekilde artıran değişkenlerdir.

- **Reason 3 (Zehirlenme):** En yüksek katsayıya sahip. Olasılık oranı (Odds Ratio) yaklaşık **20**.
    
    - **Yorum:** Bir çalışan "Zehirlenme" (Reason 3) bildirirse, hiç neden belirtmeyen birine göre "aşırı devamsızlık" yapma olasılığı **20 kat** daha fazladır.
        
- **Reason 1 (Çeşitli Hastalıklar):** Olasılık oranı yaklaşık **14**.
    
    - **Yorum:** Standart hastalık vakalarıdır. Bu gruptaki nedenler devamsızlık olasılığını **14 kat** artırır.
        
- **Reason 2 (Hamilelik ve Doğum):** Olasılık oranı yaklaşık **2**.
    
    - **Yorum:** Diğer nedenlere göre daha düşük bir etkisi var. Transkriptte de belirtildiği gibi; hamilelik düzenli kontroller gerektirse de her zaman "aşırı" devamsızlığa (eşik değerimizin üzerine) yol açmaz.
        

### 2. Standartlaştırılmış Değişkenlerin Yorumlanması

 **Transportation Expense (Ulaşım Masrafı):**

- **Sorun:** Bu değişkeni standartlaştırdığımız için "1 Euro artış" diyemiyoruz.
    
- **Teknik Yorum:** "Ulaşım masrafındaki **1 standart sapmalık** artış, aşırı devamsızlık olasılığını yaklaşık **2 kat** ($Odds \ Ratio \approx 1.9$) artırır."
    
- **Neden Önemli?** Makine öğrenmesi mühendisleri doğruluk (accuracy) için standartlaştırmayı seçerken, istatistikçiler yorumlanabilirlik için orijinal birimleri tercih eder. Biz burada her iki dünyayı da gördük.

### 3. Negatif Katsayılar ve Azaltıcı Etkiler

Katsayısı negatif olan değişkenler, olasılığı düşüren unsurlardır.

- **Pet (Evcil Hayvan):** Odds ratio $0.73$ civarındadır.
    
    - **Matematiksel Yorum:** $1 - 0.73 = 0.27$
        
    - **Hikayeleştirme:** Her bir ek evcil hayvan (veya standart sapma artışı), aşırı devamsızlık olasılığını **%27 oranında azaltır.** Mantık şudur: Çok evcil hayvanı olanın muhtemelen evde ona yardım eden birileri de vardır.
        

### 4. Etkisiz Değişkenler (Gereksizler)

Aşağıdaki değişkenlerin katsayıları sıfıra, olasılık oranları ise 1'e çok yakındır:

- `Daily Work Load Average`
    
- `Distance to Work`
    
- `Day of the Week`
    
- **Sonuç:** Bu faktörlerin işe ne kadar uzak olduğunuzun veya haftanın hangi günü olduğunun "aşırı devamsızlık" (bizim tanımladığımız >3 saat sınırı) üzerinde neredeyse hiçbir etkisi yoktur.
    

---

##  Teknik Uygulama: Katsayıları Yorumlama Fonksiyonu

Modelin çıktılarını daha iyi okumak için şu kod bloğunu kullanabiliriz:

Python

```python
# Katsayıları anlamlandırmak için bir tabloyu tekrar gözden geçirelim
# summary_table zaten elimizde (bir önceki adımda oluşturduğumuz)

# Sadece anlamlı (0'dan uzak) olanları filtreleyip görelim
significant_features = summary_table[abs(summary_table['Coefficient']) > 0.1]

# Intercept (Bias) hakkında not:
# Intercept modelin kalibrasyonudur. Tüm girdiler 0 olduğunda temel olasılığı belirler.
# 'Bias' olarak da adlandırılır.
print(significant_features)
```

### Veri Bilimcisi Notu:
 
 **Prediction (Tahmin)** mi istiyoruz yoksa **Insights (İçgörü)** mü?

- Eğer amacımız sadece kimin devamsızlık yapacağını bilmekse (tahmin), doğruluğu yüksek olan standartlaştırılmış modelle devam ederiz.
    
- Eğer yönetim "Neden devamsızlık yapıyorlar?" diye sorarsa (içgörü), standartlaştırmadan arındırılmış katsayıları açıklarız.
    

---
Bu bölüm, veri biliminde **"Less is More" (Az, çoktur)** felsefesinin en güzel örneği. Modelimizi gereksiz yüklerden kurtarıp daha sade, daha anlaşılır ve daha genelleyebilir hale getiriyoruz.

**"Backward Elimination" (Geriye Doğru Eleme)** mantığını, özellikle `sklearn` dünyasında p-değerleri (p-values) olmadan nasıl yöneteceğimizi ve kodumuzu nasıl daha "akıllı" (parameterized) hale getireceğimizi adım adım inceleyelim.

---

### 1. Gereksiz Değişkenleri Eleme (Data Manipulation)

Önceki analizimizde `Day of the Week`, `Daily Work Load Average` ve `Distance to Work` değişkenlerinin katsayılarının sıfıra çok yakın olduğunu görmüştük. Şimdi onları veri setinden çıkaralım.

Python

```python
# Veri setimizin hedeflerle birleştiği kontrol noktasına geri dönüyoruz
data_with_targets = data_preprocessed.copy()

# Etkisi olmayan 3 sütunu düşürüyoruz
# Not: axis=1 sütun bazlı silme demektir
data_with_targets = data_with_targets.drop(['Day of the Week', 'Daily Work Load Average', 'Distance to Work'], axis=1)

# Girdileri (Inputs) ve Hedefleri (Targets) tekrar ayırıyoruz
unscaled_inputs = data_with_targets.iloc[:, :-1]
targets = data_with_targets.iloc[:, -1]
```

---

### 2. Akıllı Sütun Seçimi: List Comprehension

Kodumuzu daha esnek hale getirmek için **List Comprehension** kullanarak, "ölçeklenmeyecekler dışındaki her şeyi ölçekle" mantığını kuruyoruz.

Python

```python
# Ölçeklemek İSTEMEDİĞİMİZ (Dummy ve Binary) sütunlar
columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education']

# List Comprehension ile ölçeklenecek sütunları dinamik olarak buluyoruz
# "Eğer x, atlanacaklar listesinde yoksa, onu ölçeklenecekler listesine ekle"
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]

# CustomScaler nesnemizi bu dinamik listeyle güncelliyoruz
absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
```

---

### 3. Modelin Yeniden Eğitilmesi ve Analiz

Değişken sayısını azalttık (Dimensionality Reduction). Peki sonuç ne oldu?

Python

```python
# Veriyi tekrar böl ve eğit
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=20)
reg = LogisticRegression()
reg.fit(x_train, y_train)

# Yeni doğruluğu kontrol et
print(f"Yeni Model Accuracy: {reg.score(x_train, y_train)}")
```

###  Neden Bu Kadar Önemli?

- **Karmaşıklık vs. Performans:** 3 değişkeni sildik ama modelin doğruluğu (accuracy) neredeyse hiç değişmedi. Bu, o 3 değişkenin modele hiçbir bilgi (information) katmadığını kanıtlar.
    
- **Aşırı Öğrenme (Overfitting) Riski:** Modelde ne kadar çok gereksiz değişken varsa, model o değişkenlerdeki "gürültüyü" (noise) öğrenmeye o kadar meyillidir. Sade model, gerçek dünyada daha tutarlı çalışır.
    
- **P-Değerleri Meselesi:** `sklearn` mühendisleri, küçük katsayıların zaten p-değerinin yüksek olacağına (istatistiksel olarak anlamsız) işaret ettiğini varsayar. Bu pratik bir mühendislik yaklaşımıdır.
    

---

### Final Kontrolü: Summary Table

Gereksiz değişkenler elendikten sonra `summary_table`'ı tekrar çalıştırdığında, katsayıların birbirine çok yakın kaldığını ama listenin daha derli toplu olduğunu göreceksin. Artık modelin "özünü" (core drivers) bulduk.

Şu an elimizde **optimize edilmiş, sadeleştirilmiş ve yorumlanabilir** bir model var.

**Modelimiz artık savaşa hazır!**

---
 veri biliminde **"Code Robustness" (Kodun Dayanıklılığı)** dediğimiz çok kritik bir konuyu ele alıyor. Hangi sütunların ölçekleneceğini (scaling) manuel yazmak yerine, **List Comprehension** kullanarak dinamik bir şekilde belirlemektir.
### 1. Adım: Sütunları Ayıklama Mantığı (Teorik Hazırlık)

 Ölçekleme (scaling) işlemine dahil edilmemesi gereken (kategorik olanlar: Reasons ve Education) sütunları bir listede tutup, geri kalanları otomatik seçmemizi istiyor.

**Neden?** Çünkü `Reason 1, 2, 3, 4` ve `Education` zaten 0 ve 1 değerlerinden oluşur; bunları standartlaştırmak (mean=0, std=1 yapmak) anlamlarını bozar.

---

### 2. Adım: Dinamik Ölçekleme Kod Bloğu

Python

```python
# 1. Ölçekleme dışı bırakılacak (kategorik) sütunları tanımlıyoruz
# Not:'Education' ve Dummy (Reason) değişkenleri ölçeklenmemeli.
columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education']

# 2. Ölçeklenecek sütunları List Comprehension ile dinamik olarak belirliyoruz
# Bu yapı, unscaled_inputs içindeki tüm sütunlara bakar ve omit listesinde olmayanları seçer.
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]

# 3. Teknik Kontrol: Hangi sütunların seçildiğini görelim
print("Ölçeklenecek Sütunlar:", columns_to_scale)
```

---

### 3. Adım: Scaler'ı Yeni Listeye Göre Güncelleme

Eskiden tüm sütunları veya manuel bir listeyi ölçekliyorken, şimdi oluşturduğumuz bu `columns_to_scale` listesini kullanmalıyız:

Python

```python
# 'manuel düzeltme yerine otomatik seçim' kısmıdır.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# Standart StandardScaler kullanımı örneği:
absenteeism_scaler = StandardScaler()
absenteeism_scaler.fit(unscaled_inputs[columns_to_scale])
```

---

### Neden "Restart & Run All" Yapmalısınız?

- **Hafıza Temizliği:** Önceki denemelerinizden kalan hatalı değişkenler RAM'de asılı kalabilir.
    
- **Akış Doğrulaması:** Kodun en başından en sonuna (Regresyon katsayılarına kadar) sorunsuz aktığını görmeniz gerekir.
    
- **Basitlik Prensibi:** 3 değişkeni (Day of the Week, vb.) sildiğimiz halde başarı oranı (accuracy) değişmiyorsa, **Occam'ın Usturası** prensibi gereği daha basit olan model her zaman daha iyidir.
    

### Ne Yapmanız Gerekiyor?

1. En üstteki hücreye gidip `data_preprocessed = df` yaptığınızdan emin olun.
    
2. Sütunları düşürdüğünüz (`drop`) hücreyi çalıştırın.
    
3. Yukarıda verdiğim **List Comprehension** (`columns_to_scale`) kodunu ölçekleme aşamasına ekleyin.
    
4. Jupyter menüsünden **Kernel -> Restart & Run All** seçeneğine basın.
    

Bu işlemden sonra regresyon sonuçlarınızın (accuracy) çok küçük bir farkla ama daha stabil bir modelle geldiğini göreceksiniz.

---
"Testing" aşaması, makine öğrenmesi sürecinin "doğruluk anı"dır. Eğitim (train) verisinde %77 başarı elde etmek iyidir ancak bu veriler modelin zaten "ezberlediği" sorulardır. Gerçek yetenek, daha önce hiç görmediği **Test** verisindeki performansı ile ölçülür.

İşte bu süreci teknik detaylarını koruyarak ve kod bloklarıyla takip edebileceğiniz analiz:

---

### 1. Test Doğruluğunun (Test Accuracy) Hesaplanması

Eğitim sürecinde `reg.score(x_train, y_train)` kullanarak elde ettiğimiz skoru, şimdi test verileri için tekrarlıyoruz.

Python

```python
# Modelin daha önce hiç görmediği X_test ve y_test verileri üzerinden skorunu alıyoruz
test_accuracy = reg.score(x_test, y_test)

print(f"Test Doğruluğu: {test_accuracy}")
```

**Teorik Analiz:**

- **Overfitting (Aşırı Öğrenme) Kontrolü:** Test doğruluğunuzun (%74), eğitim doğruluğundan (%77) biraz düşük olması çok normal ve sağlıklıdır. Eğer aradaki fark %10-20 olsaydı, modelin veriyi ezberlediğini (overfitting) söyleyebilirdik.
    
- **Değişmezlik Kuralı:** Test skorunu aldıktan sonra modele geri dönüp parametreleri değiştirmemelisiniz. Eğer değiştirirseniz, test setini de dolaylı yoldan eğitime dahil etmiş olursunuz ve test setiniz "tarafsızlığını" yitirir.
    

---

### 2. Tahmin Olasılıkları: `predict_proba`

Sadece 0 veya 1 (gelir/gelmez) sonucunu almak yerine, modelin bu kararı ne kadar "emin" olarak verdiğini görebiliriz. Lojistik regresyon aslında arka planda olasılıklar hesaplar.

Python

```python
# Her gözlem için 0 ve 1 olma olasılıklarını alıyoruz
# Çıktı: [0 olma olasılığı, 1 olma olasılığı] şeklinde bir matristir
predicted_proba = reg.predict_proba(x_test)

# Sadece '1' olma (excessive absenteeism - aşırı devamsızlık) olasılıklarını çekiyoruz
# Tüm satırları al (:) ve 1. indeksteki sütunu al (1)
absenteeism_probability = predicted_proba[:, 1]

# İlk 5 olasılığa göz atalım
print(absenteeism_probability[:5])
```

**Neden `predict_proba`?**

- Standart `predict` metodu, olasılık $> 0.5$ ise `1`, değilse `0` sonucunu verir.
    
- Ancak bazı durumlarda %51 olasılıkla "gelmez" demekle, %99 olasılıkla "gelmez" demek arasında iş kararı açısından büyük fark vardır. Bu olasılık değerleri Tableau analizlerinizde çok daha zengin bir görselleştirme sunacaktır.
    

---

### 3. Modelin Kaydedilmesi ve Entegrasyon Hazırlığı

Eğitim bittiğine göre, bu modeli (ağırlıkları ve katsayıları) bir dosyaya kaydedip her seferinde baştan eğitmeden kullanabiliriz. Bunun için Python'ın ==`pickle`== kütüphanesini kullanacağız.

Python

```python
import pickle

# Modeli 'model' adında bir dosyaya kaydediyoruz
with open('model', 'wb') as file:
    pickle.dump(reg, file)

# NOT: Scaler (ölçekleyici) nesnesini de kaydetmelisiniz! 
# Yeni gelen verileri de aynı şekilde ölçeklemek zorundayız.
with open('scaler', 'wb') as file:
    pickle.dump(absenteeism_scaler, file)
```

### Teknik İş Akışı Özeti

|**Aşama**|**İşlem**|**Amaç**|
|---|---|---|
|**Testing**|`reg.score(x_test, y_test)`|Modelin genel performansını ve overfitting durumunu ölçmek.|
|**Probability**|`reg.predict_proba(x_test)[:,1]`|Sınıflandırma kararının arkasındaki güven oranını (0.0 - 1.0) almak.|
|**Saving**|`pickle.dump()`|Model ve Scaler nesnelerini diske kaydedip tekrar kullanılabilir hale getirmek.|

---

### Sonraki Adımlarımız

Makine öğrenmesi kısmını burada noktalıyoruz. Şimdi sırada şunlar var:

1. **Module Oluşturma:** Teknik olmayan iş arkadaşlarınızın veriyi sadece bir dosya yükleyerek analiz edebilmesi için bir Python modülü (fonksiyonlar dizisi) yazacağız.
    
2. **SQL ve Tableau Entegrasyonu:** Model sonuçlarını veritabanına aktarıp Tableau'da profesyonel bir Dashboard oluşturacağız.
    

Modeli kaydettikten sonra, bu modelin ağırlıklarını (weights) ve `intercept` değerini bir dosyada saklamaya hazır ve bu `pickle` işleminin detaylarına inilebilir.

---
Makine öğrenmesi sürecinin en tatmin edici anlarından birine geldiniz: **Model Deployment (Dağıtım)** hazırlığı. Bir modeli eğitmek işin yarısıdır; onu taşınabilir ve tekrar kullanılabilir hale getirmek ise gerçek hayatta değer yaratan kısımdır.

İşte anlatılanların teknik özeti, mantıksal derinliği ve ödeviniz olan ==`scaler`== kaydetme işleminin çözümü:

### 1. Neden "Pickle" Kullanıyoruz?

**Pickling**, bir Python nesnesini (sizin durumunuzda eğitilmiş regresyon modelini) bir karakter dizisine (byte stream) dönüştürme işlemidir.

- **Kalıcılık:** `reg` nesnesi sadece o anki oturumda RAM üzerinde yaşar. Notebook'u kapattığınızda ölür. Pickle ile onu sabit diske bir dosya olarak "dondururuz".
    
- **Hafiflik:** Dosya boyutu 1 KB'dan bile küçüktür çünkü sadece katsayıları
- ($coefficients$), kesim noktasını ($intercept$) ve parametreleri saklar; devasa veri setini saklamaz.
---

### 2. Modeli Kaydetme (Yazma İşlemi)

Python

```python
import pickle

# 'model' isminde bir dosya açıyoruz. 
# 'wb' (Write Binary): Dosyaya ikili formatta yazacağımızı belirtir.
with open('model', 'wb') as file:
    # reg nesnesini (eğitilmiş modelimizi) bu dosyaya boşaltıyoruz (dump)
    pickle.dump(reg, file)
```

---

### 3. Kritik Ayrıntı: Neden Scaler Nesnesini de Kaydetmeliyiz?

Bu nokta teknik derinlik açısından çok kritiktir. Modeli (`reg`) kaydettiniz ama **ölçekleyiciyi (`scaler`)** unutursanız yeni gelen verileri tahmin edemezsiniz.

- **Veri Uyumu:** Modeliniz, verilerin belirli bir ortalama ($\mu$) ve standart sapma 
- ($\sigma$) ile ölçeklenmiş haline göre eğitildi.
    
- **Kural Seti:** Yeni veri (hiç görülmemiş veri) geldiğinde, modelin bunu anlaması için o yeni verinin de **tam olarak aynı** kurallarla (eğitim setinin ortalaması ve sapmasıyla) ölçeklenmesi gerekir.
    
- **Bağımsızlık:** Scaler'ı kaydederek, modelinizi artık orijinal eğitim verisine 
- (`df`) ihtiyaç duymadan her yerde çalışabilir hale getirirsiniz.
---

### 4. Scaler Nesnesini Kaydetme

 `absenteeism_scaler`  nesnesini kaydetme kodu şöyledir:

Python

```python
# 'scaler' isminde yeni bir dosya oluşturuyoruz
# Not: absenteeism_scaler ismini kendi değişken isminizle kontrol edin
with open('scaler', 'wb') as file:
    pickle.dump(absenteeism_scaler, file)
```

### Teknik Kontrol Listesi

1. **Klasör Kontrolü:** Bu kodları çalıştırdıktan sonra, Notebook dosyanızın bulunduğu klasörde uzantısız veya `.pkl` (isteğe bağlı) olarak `model` ve `scaler` isimli iki yeni dosya görmelisiniz.
    
2. **Güvenlik:** Sadece güvendiğiniz kişilerden gelen pickle dosyalarını açın
3. (`unpickle`), çünkü bu işlem sırasında kötü amaçlı kodlar çalıştırılabilir.

Makine öğrenmesi modelinizi artık "paketlediniz". Bir sonraki adımda bu dosyaları yeni bir ortamda nasıl "uyandıracağımızı" (deployment) göreceğiz.

---
---
Makine öğrenmesi modelini (`reg`) ve veri ölçekleyiciyi (`absenteeism_scaler`) dış dünyaya aktarılabilir dosyalar haline getiriyorsun.

### Modeli ve Ölçekleyiciyi Kaydetme (Final Adımı)

Bu kod bloğunu çalıştırdığında, proje klasöründe iki yeni dosya oluşacaktır. Bu dosyalar, modelin tüm "hafızasını" saklar.

Python

```python
import pickle

# 1. Eğitilmiş Lojistik Regresyon Modelini Kaydetme
# 'wb' -> write binary (ikili formatta yaz)
with open('model', 'wb') as file:
    pickle.dump(reg, file)

# 2. Ölçekleyiciyi (Scaler) Kaydetme
# Bu nesne, yeni verileri aynı istatistiksel kurallarla (mean, std) düzeltmek için şarttır.
with open('scaler', 'wb') as file:
    pickle.dump(absenteeism_scaler, file)
```

### Teknik Analiz: Neden "Aynısı"?

- **Bütünlük:** Görseldeki kodda hem `model` hem de `scaler` dosyaları için `pickle.dump()` kullanılmış. Bu, modelin çalışması için gereken tüm bileşenleri paketlediğini gösterir.
    
- **Serileştirme:**  "karakter dizisine dönüştürme" işlemi tam olarak bu hücre çalıştırıldığında gerçekleşir.
    
- **Güvenli Dosya Modu:** Her iki işlemde de `with open(...)` kalıbı kullanılmış. Bu, Python'da dosya işlemlerini yapmanın en güvenli yoludur; işlem bitince dosyayı otomatik olarak kapatır ve veri kaybını önler.
    

---

### Dikkat Etmen Gereken Önemli Nokta

Kodun çalışması için daha önce tanımladığın **`absenteeism_scaler`** değişken isminin, yukarıdaki hücrelerdeki isimle birebir aynı olduğundan emin olmalısın. Eğer yukarıda ismi sadece `scaler` olarak bıraktıysan, `pickle.dump(scaler, file)` şeklinde güncellemen gerekebilir.

### Bir Sonraki Adım

Bu dosyaları başarıyla oluşturduysan, makine öğrenmesi eğitim defterini (Notebook) kapatmaya hazırsın demektir. Bir sonraki aşamada şunları yapacağız:

1. Yeni bir Python dosyası açacağız.
    
2. Bu dosyaları `pickle.load()` ile "uyandıracağız".
    
3. Tamamen yeni ve ham bir veriyi bu "kara kutu"dan geçirip tahminler alacağız.
---
 Burası projenin **kırılma noktasıdır.** Buraya kadar yaptığımız her şey "Laboratuvar Ortamı"ndaydı. Şimdi ise işi "Fabrika Üretimi"ne, yani gerçek hayata (Production) taşıyoruz.


---

 *"Clumsy" (Hantal) ve "Clever" (Akıllı) yaklaşım farkını ve modül oluşturma süreci:*
### Adım 1: Mantığı Anlamak (Neden Bunu Yapıyoruz?)

- **Hantal Yaklaşım (Clumsy):** Yarın patronun elinde `New_Data.csv` ile gelip "Sonuçlar nerede?" dediğinde, eski Notebook'u açıp, tüm satırları tekrar çalıştırıp, yeni veriyi oraya yapıştırıp hata ayıklamaya çalışmaktır. Bu risklidir ve amatörcedir.
    
- **Akıllı Yaklaşım (Clever - Bizim Yapacağımız):** Kodlarımızı bir "Motor" (Module) haline getirmektir. Yarın yeni veri geldiğinde sadece "Motoru Çalıştır" (Import) ve "Veriyi İşle" komutunu vereceğiz.
    

### Adım 2: Modülün İçeriği (Kodun İskeleti)

 Bu kod, **Nesne Yönelimli Programlama (OOP)** kullanılarak yazılmıştır. İçinde neler olduğunu anlaman çok önemli:

1. **Kütüphaneler:** En başta `pandas`, `numpy`, `pickle`, `sklearn` gibi kütüphaneleri çağırır.
    
2. **Custom Scaler Sınıfı:** Eğitimde kullandığımız o özel ölçekleme sınıfının aynısı burada da var.
    
3. **`absenteeism_model` Sınıfı (YENİ VE KRİTİK KISIM):**
    
    Bu sınıf, bizim asıl makine öğrenmesi motorumuzdur. İçinde 5 temel fonksiyon (method) vardır:
    
    - **`__init__` (Constructor - Kurucu):** Sınıfı çağırdığın an çalışır. Senin az önce kaydettiğin `model` ve `scaler` dosyalarını **unpickle** yapar (okur) ve hafızaya yükler. Yani motoru çalıştırır.
        
    - **`load_and_clean_data`:** Yeni gelen ham veriyi (CSV) alır. Tıpkı eğitimde yaptığımız gibi sütunları düşürür, "Reason"ları gruplar, dummy değişkenleri yaratır ve sıralar.
        
    - **`predicted_probability`:** Modelin "1" olma (aşırı devamsızlık) olasılığını hesaplar.
        
    - **`predicted_output_category`:** Sonucun 0 mı 1 mi olduğuna karar verir.
        
    - **`predicted_outputs`:** Tüm bu sonuçları güzel bir tablo haline getirir.
        

### Adım 3: Uygulama (Notebook'u .py Dosyasına Çevirme)

Şimdi fiziksel olarak **"Integration"** veya **"Module"** adıyla verilmiş olan, içinde yukarıda bahsettiğim uzun kodların olduğu Notebook'u açmalısın.

Eğer o kod zaten elindeyse şu adımları izle:

1. **Notebook'u Aç:** Modül kodlarının (Class yapısının) olduğu Notebook'u aç.
    
2. **Menüye Git:** Sol üstteki menüden **File (Dosya)** seçeneğine tıkla.
    
3. **İndir:** **Download as** (Farklı indir) seçeneğine gel.
    
4. **Format Seç:** Listeden **Python (.py)** seçeneğini seç.
    
5. **Tarayıcı Uyarısı:** Chrome veya tarayıcın "Bu dosya zararlı olabilir" diyebilir. "Sakla" (Keep) de. Çünkü kodu sen yazdın/biliyorsun, güvenli.
    

### Adım 4: Dosya Konumu ve İsimlendirme (ÇOK KRİTİK)

Bu adımda hata yaparsan modül çalışmaz.

1. İndirdiğin dosyanın adı muhtemelen `absenteeism_module.py` olmalı. 
    
2. Bu dosyayı al ve **`model`** ve **`scaler`** dosyalarının olduğu klasörün içine at.
    
3. **Kural:** `New_Data.csv`, `model`, `scaler` ve `absenteeism_module.py` dosyalarının hepsi **AYNI KLASÖRDE** yan yana durmalı.
### Adım 5: Son Kontrol

Bu aşamayı tamamladıysan, elinde bir "İsviçre Çakısı" var demektir. Artık kod yazmakla uğraşmayacaksın, sadece bu çakıyı kullanacaksın.

**Senin Görevin:**

 Modül kodunu bulup (veya yazıp), 
 **==(EK modül.py yazılmıştır modülasyon yapılarak repo edilmiştir).==**

---
---
├── 01_Preprocessing_and_Modeling/
│   ├── Absenteeism_Preprocessing.ipynb  (Mutfak işi - Analizlerin, yorumların)
│   ├── Absenteeism_Modeling.ipynb       (Model eğitimi, katsayı analizi)
│   └── Absenteeism_data.csv             (Ham veri)
├── 02_Deployment/
│   ├── absenteeism_module.py            (Paketlenmiş motor - .py formatında)
│   ├── model                            (Kaydettiğin model dosyası)
│   ├── scaler                           (Kaydettiğin scaler dosyası)
│   ├── New_Data_Predictions.ipynb       (Modülü kullandığın yeni sayfa)
│   └── Absenteeism_new_data.csv         (Tahmin için kullanılan yeni veri)
└── README.md                            (Türkçe rehber ve projenin özeti)

---

# Deployment 

## Bölüm 1: Model Entegrasyonu ve Altyapı Hazırlığı

Makine öğrenmesi modelleri sadece birer kod yığını değildir; onlar eğitildikleri andaki parametreleri saklayan dijital varlıklardır. Analiz aşamasında elde ettiğimiz sonuçları dış dünyaya açmak için iki temel dosyaya ihtiyaç duyduk:

1. **`model`**: Lojistik Regresyon katsayılarımızı içeren `pickle` dosyası.
    
2. **`scaler`**: Verileri standartlaştırmak için kullandığımız istatistiksel parametreleri içeren dosya.
    

Bu dosyaları kullanabilmek için her şeyi otomatize eden bir "motor" yazdık: **`absenteeism_module.py`**.

### 1.1. Modülün Başlığı ve Kodlama Standartları

Modülün her sistemde (Windows, Mac, Linux) ve her dilde (Türkçe karakter desteği ile) sorunsuz çalışması için:

Python

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
```

---

## Bölüm 2: Özel Ölçekleyici (CustomScaler) İnşası

Eğitim sırasında fark ettik ki; standart bir ölçekleyici tüm sütunları (dummy değişkenler dahil) değiştiriyordu. Biz ise sadece sayısal değerleri (Yaş, Yol Masrafı vb.) küçültmek istedik. Bu yüzden modülümüzün kalbine şu özel sınıfı yerleştirdik:

Python

```python
class CustomScaler(BaseEstimator, TransformerMixin): 
    def __init__(self, columns):
        self.scaler = StandardScaler()
        self.columns = columns

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
```

> **Teknik Not:** Bu sınıf, dummy (0-1) değişkenlerin doğallığını bozmadan sadece seçili sütunları ölçeklendirerek modelin doğruluğunu korur.

---

## Bölüm 3: Veri Temizleme Fabrikası (Preprocessing)

Modülün içindeki `load_and_clean_data` fonksiyonu, ham bir CSV dosyasını alıp saniyeler içinde modelin anlayacağı mükemmel bir forma dönüştürür. Burada analiz notebook'unda yaptığın her adımı koda döktük:

1. **Gereksiz Verilerin Atılması:** `ID` sütunu gibi tahmine etkisi olmayan veriler silindi.
    
2. **Nedenlerin Gruplanması:** 28 farklı devamsızlık nedeni, 4 ana gruba indirgendi (Reason_1, 2, 3, 4).
    
3. **Zaman Analizi:** Ham tarihlerden `Month Value` ve `Day of the Week` bilgileri çıkarıldı.
    
4. **Eğitim Durumu:** Veriler "Lise" (0) ve "Yüksek Öğrenim" (1) olarak binary hale getirildi.
    

---

## Bölüm 4: Tahmin Mekanizması ve Çıktılar

Veri temizlendikten sonra modelimiz iki kritik soruya cevap verir:

- **Olasılık (Probability):** Bu kişinin devamsızlık yapma ihtimali yüzde kaç?
    
- **Tahmin (Prediction):** Olasılık %50'den büyükse `1` (Evet), küçükse `0` (Hayır).
    

**Uygulanan Nihai Kod Bloğu:**

Python

```python
# Modülü çağır ve motoru çalıştır
from absenteeism_module import *
model = absenteeism_model('model', 'scaler')

# Yeni veriyi yükle ve işle
model.load_and_clean_data('Absenteeism_new_data.csv')

# Sonuçları al ve CSV olarak kaydet
summary_table = model.predicted_outputs()
summary_table.to_csv('Absenteeism_predictions.csv', index = False)
```

---

## Bölüm 5: Teknik Gözlemlerimiz

Sürecin sonunda iki önemli teknik tespitte bulunduk:

1. **Modern Pandas Etkisi:** Güncel kütüphaneler dummy değişkenleri `1/0` yerine `True/False` olarak gösterebilir. Bu durum matematiksel olarak `True=1` ve `False=0` olduğu için modelin başarısını etkilemez; sadece belleği daha verimli kullanır.
    
2. **Veri Seti Boyutu:** Analizdeki 700 satır yerine "New Data" aşamasında 40 satır görmemiz, modelin artık gerçek hayattaki "yeni gözlemleri" işlediğini kanıtlar.
---
###### Önemli Bilgi: modül.py dosyası ilk taslaktır. Nihai kullanım için modülasyon yapılıp repodadır. ==model_integration== = *modul.py*

---
