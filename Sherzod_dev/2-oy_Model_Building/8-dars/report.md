# 8-dars — savol-javob (Model Building)

## 1. Missing value nima? Ularni aniqlash va to‘ldirish qanday bo‘ladi?

### Missing value nima?

**Missing value** — jadvalda bo‘lishi kerak bo‘lgan, lekin yozilmagan yoki yo‘qolgan qiymatlar. Pandasda odatda `NaN` (raqamli ustunlar), ba’zan `None` yoki bo‘sh qatorlar ko‘rinishida bo‘ladi.

### Aniqlash (pandas)

| Usul | Vazifasi |
|------|----------|
| `df.isnull()` yoki `df.isna()` | Har bir katak uchun `True` / `False` (missing bo‘lsa `True`) |
| `df.isnull().any()` | Har bir ustunda kamida bitta missing bormi |
| `df.isnull().sum()` | Har bir ustundagi missinglar soni |
| `df.info()` | Ustunlar bo‘yicha non-null soni (umumiy ko‘rinish) |

### To‘ldirish (imputation)

- **Bitta qiymat bilan:** `df['ustun'].fillna(qiymat)` — masalan o‘rtacha, median, mode.
- **Sklearn:** `SimpleImputer` — strategiya: `mean`, `median`, `most_frequent` va hokazo.
- **Vaqt qatori:** oldingi/keyingi kuzatuv qiymati bilan to‘ldirish.
- **O‘chirish:** `df.dropna()` — missing juda ko‘p yoki tahlil uchun ma’qul bo‘lmasa.

---

## 2. Train-test split va K-fold cross-validation — farqi va ishlashi

### Train-test split

- Ma’lumot **ikki** qismga bo‘linadi: **training set** (odatda ~70–80%) va **test set** (~20–30%).
- Kod: `sklearn.model_selection` dan `train_test_split` import qilinadi.
- **Maqsad:** model train qismida o‘rganadi; test qismi faqat **yakuniy** baholash (umumlashtirish) uchun alohida saqlanadi.

### K-fold cross-validation

- Ma’lumot **K** ta qismga (fold) bo‘linadi; jarayon **K marta** takrorlanadi: har safar bitta fold **validatsiya**, qolgan **K−1** fold **train**.
- Kod: `KFold` (yoki `StratifiedKFold` klassifikatsiya uchun), `cross_val_score` va boshqa `cross_val_*` funksiyalari.
- **Maqsad:** bitta tasodifiy bo‘linishga bog‘liq bo‘lmagan, **o‘rtacha** ishonchli baho olish; hyperparametr tanlashda ham ishlatiladi.

### Qisqacha farq

| | Train-test split | K-fold CV |
|---|------------------|-----------|
| Bo‘linish | Odatda **1 marta**, 2 to‘plam | **K marta** turli train/validation juftlari |
| Asosiy foyda | Sodda, tez | Bahoning barqarorligi, ma’lumotdan yaxshiroq foydalanish |
| Test | Bitta hold-out test | K-foldda “test” o‘rniga har iteratsiyada validation; alohida test set alohida saqlanishi mumkin |

---

## 3. Scalerlash (masshtablash) usullari

**Nima uchun?** Xususiyatlar (ustunlar) turli o‘lchamlarda bo‘lsa, masofa yoki gradientga asoslangan modellar noto‘g‘ri yo‘nalishi mumkin. Shuning uchun qiymatlarni bir xil masshtabga keltirish **scaling** deyiladi.

### StandardScaler

- Har bir ustun: **o‘rtacha ≈ 0**, **standart tafovut ≈ 1** bo‘ladi:  
  \(z = \dfrac{x - \text{mean}}{\text{std}}\).
- **0 va 1 oralig‘iga** siqilmaydi (bu **MinMax**ning vazifasi).
- Logistic regression, SVM, neural network kabi usullar uchun tez-tez ishlatiladi.

### MinMaxScaler

- Qiymatlarni berilgan oralikka odatda **[0, 1]** ga siqadi: eng kichik → 0, eng katta → 1.
- KNN va ba’zi boshqa usullar uchun qulay bo‘lishi mumkin.

### RobustScaler

- **Median** va **IQR** (interkvartil oralik) asosida masshtablaydi; **outlier**larga `StandardScaler`ga qaraganda sekinroq “ta’sir qiladi”.
- Chetki qiymatlar ko‘p bo‘lgan ma’lumotlarda foydali.

### Muhim eslatma (data leakage)

`fit` faqat **train** (yoki har bir CV fold ichidagi train) ustida qilinadi; **test** / validatsiya ustida faqat `transform` qo‘llanadi.

---

## 4. Pythonda kutubxona (library) nima?

**Kutubxona (library)** — boshqa dasturchilar yoki jamoalar tomonidan yozilgan **tayyor modul va paketlar to‘plami**. Ularni `import` orqali loyihaga ulab, funksiya va klasslarni qayta yozmasdan ishlatish mumkin (masalan: `pandas`, `numpy`, `sklearn`).

---

*Ushbu hujjat `report.ipynb` bilan bir xil mazmundagi qisqa tezislar bo‘yicha tuzilgan.*
