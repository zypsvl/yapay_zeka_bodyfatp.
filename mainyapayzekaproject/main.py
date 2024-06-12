import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer

# Helper function to format the output
def formatter(param):
    pass

# Load dataset
_=formatter('=')
veri = pd.read_csv('bodyfat.csv')
veri.sample(5)
print(formatter("Veri Sayısı :"),veri.shape[0])
print(formatter('Sütun Sayısı :'),veri.shape[1])

hedef = 'BodyFat'
test = veri.sample(52, random_state=30)
eğitim = veri.drop(test.index)

pd.concat((eğitim.describe(), pd.DataFrame([eğitim.skew(), eğitim.kurtosis()], index=['Çarpıklık', 'Basıklık'])))

eğitim.hist(figsize=(18, 15), grid=False, color='#7a38ff')
plt.show()

fig, axes = plt.subplots(3, 5)
sütun = 0
for r in range(3):
    for c in range(5):
        sns.boxplot(eğitim[eğitim.columns[sütun]], ax=axes[r, c], color='#ff008f', linewidth=3, saturation=0.9)
        axes[r, c].set_title(eğitim.columns[sütun])
        sütun += 1

fig.set_size_inches(25, 15)
plt.show()

def kapper(series):
    Q1, Q3 = np.quantile(series, (.25, .75))
    IQR = Q3 - Q1
    Min = Q1 - 1.5 * IQR
    Max = Q3 + 1.5 * IQR
    return np.clip(series, Min, Max)

def df_kapper(data, features):
    for feature in features:
        data[feature] = kapper(data[feature])
    return data

eğitim = df_kapper(eğitim, eğitim.columns)
test = df_kapper(test, test.columns)

eğitim['VücutKitleEndeksi'] = (eğitim['Weight'] / (eğitim['Height'] ** 2)) * 703
test['VücutKitleEndeksi'] = (test['Weight'] / (test['Height'] ** 2)) * 703
eğitim['VücutKitleEndeksi'].hist(color='#7a38ff', grid=False)
plt.show()
eğitim.corr().style.background_gradient(axis=0, cmap='cool', vmin=-1, vmax=1)
eğitim.shape

fig, axes = plt.subplots(4, 4)
sütun = 0
for r in range(4):
    for c in range(4):
        sns.scatterplot(y=eğitim[hedef], x=eğitim[eğitim.columns[sütun]], ax=axes[r, c], color='#ff008f', legend=False, s=100)
        axes[r, c].set_title(eğitim.columns[sütun] + f' Vs {hedef}')
        axes[r, c].set_yticks(range(1, 50, 6))
        sütun += 1

fig.set_size_inches(20, 15)
fig.tight_layout()
plt.show()


X = eğitim.drop([hedef, 'Density'], axis=1)
y = eğitim[hedef]
X_eğitim, X_doğrulama, y_eğitim, y_doğrulama = train_test_split(X, y, test_size=0.2, random_state=42)

ölçekleyici = StandardScaler()
X_eğitim_ölçekli = ölçekleyici.fit_transform(X_eğitim)
X_doğrulama_ölçekli = ölçekleyici.transform(X_doğrulama)

ölçekleyici = StandardScaler()
X_ölçekli = ölçekleyici.fit_transform(X)

modeller = {
    "Doğrusal Regresyon": LinearRegression(),
    "Rastgele Orman": RandomForestRegressor(n_estimators=100, random_state=42),
    "Destek Vektör Regresyonu": SVR(kernel='linear'),
    "K-En Yakın Komşu": KNeighborsRegressor(n_neighbors=5)
}

cv = KFold(n_splits=10, shuffle=True, random_state=42)

sonuçlar = []

for isim, model in modeller.items():
    r2_skorlayıcı = make_scorer(r2_score)
    cv_sonuçları = cross_val_score(model, X_ölçekli, y, cv=cv, scoring=r2_skorlayıcı)
    sonuçlar.append((isim, cv_sonuçları.mean(), cv_sonuçları.std(), cv_sonuçları))

sonuçlar_df = pd.DataFrame(sonuçlar, columns=['Model', 'Ortalama R^2 Skoru', 'Standart Sapma R^2 Skoru', 'CV Sonuçları'])
print(sonuçlar_df)

for isim, model in modeller.items():
    start_time = time.time()
    model.fit(X_eğitim_ölçekli, y_eğitim)
    end_time = time.time()
    training_time = end_time - start_time
    print("=" * 80)
    print("Model:", isim)
    print("Eğitim R^2 Skoru:", model.score(X_eğitim_ölçekli, y_eğitim))
    print("Doğrulama R^2 Skoru:", model.score(X_doğrulama_ölçekli, y_doğrulama))
    print("Eğitim süresi:", training_time, "saniye")

# Hata analizi
fig, axes = plt.subplots(2, 2)
fig.set_size_inches(15, 10)
for ax, (isim, model) in zip(axes.flatten(), modeller.items()):
    model.fit(X_eğitim_ölçekli, y_eğitim)
    y_pred = model.predict(X_doğrulama_ölçekli)
    hatalar = y_doğrulama - y_pred
    ax.hist(hatalar, bins=20, color='#7a38ff')
    ax.set_title(isim)
plt.tight_layout()
plt.show()

# Alternatif metriklerin hesaplanması
def alternative_metrics(y_true, y_pred, tolerance=2.0):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true - tolerance <= y_pred) & (y_pred <= y_true + tolerance))
    tn = np.sum((y_true - tolerance > y_pred) | (y_pred > y_true + tolerance))
    fp = np.sum((y_true - tolerance > y_pred) & (y_pred > y_true + tolerance))
    fn = np.sum((y_true + tolerance < y_pred) | (y_pred < y_true - tolerance))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, recall, specificity

for isim, model in modeller.items():
    model.fit(X_eğitim_ölçekli, y_eğitim)
    y_pred = model.predict(X_doğrulama_ölçekli)
    accuracy, recall, specificity = alternative_metrics(y_doğrulama, y_pred)
    print("=" * 80)
    print(f"Model: {isim}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")

print("=" * 80)
print("Lütfen tahmin yapmak için gerekli verileri girin (Density olmadan):")
veri = {}
for özellik in X.columns:
    if özellik != 'Density':
        veri[özellik] = float(input(f"{özellik}: "))

veri_df = pd.DataFrame([veri])
veri_ölçekli = ölçekleyici.transform(veri_df)

for isim, model in modeller.items():
    tahmin = model.predict(veri_ölçekli)
    print("=" * 80)
    print(f"{isim} Model Tahmini (Density olmadan): {tahmin[0]}")

print("=" * 80)
print("Lütfen tahmin yapmak için gerekli verileri girin:")
veri = {}
for özellik in X.columns:
    veri[özellik] = float(input(f"{özellik}: "))

veri_df = pd.DataFrame([veri])
veri_ölçekli = ölçekleyici.transform(veri_df)

for isim, model in modeller.items():
    tahmin = model.predict(veri_ölçekli)
    print("=" * 80)
    print(f"{isim} Model Tahmini: {tahmin[0]}")
