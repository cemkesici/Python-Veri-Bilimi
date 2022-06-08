#!/usr/bin/env python
# coding: utf-8

# ## DataSet Hakkında Bilgi
# Dataset'i adım sayısı, mesafe gibi verileri tutarak yakılan kaloriyi hesaplamak için kullanacağım.
# 
# TotalSteps,
# 
# TotalDistance,
# 
# TrackerDistance,
# 
# LoggedActivitiesDistance,
# 
# VeryActiveDistance,
# 
# ModeratelyActiveDistance, 
# 
# LightActiveDistance, 
# 
# SedentaryActiveDistance, 
# 
# VeryActiveMinutes, 
# 
# FairlyActiveMinutes, 
# 
# LightlyActiveMinutes, 
# 
# SedentaryMinutes, 
# 
# Calories.
# 
# 
# ### Dataset'in Amacı
#  Dataset'in amacı adım sayısı, mesafe, aktif zaman vs verileri ile yakılan kalori arasındaki bağıntıyı bulmaktır.
# 

# In[1]:


get_ipython().system('pip install skompiler')
get_ipython().system('pip install astor')
get_ipython().system('pip install xgboost')
get_ipython().system('pip install catboost')
get_ipython().system('pip install lightgbm')
#conda install -c conda-forge lightgbm


# In[2]:


import numpy as np
import pandas as pd 
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix,accuracy_score, classification_report, f1_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor,KNeighborsClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import datetime
from skompiler import skompile
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from lightgbm import LGBMRegressor
import sklearn
from warnings import filterwarnings
from sklearn.decomposition import PCA
filterwarnings('ignore')

from datetime import datetime


# ### Dataseti Uygulamaya dahil etme ve ilk bilgiler

# Dataseti ekleme drop etme ve ilk, son değerleri çekme

# In[3]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])#Kullanacağım verileri seçtim
df = hit.copy()#Datasetimin kopyası üzerinde çalışacağım
df= df.dropna()#Datasette boş olan değerleri çıkardım
print(df.head())#İlk beş değeri yazdırdım
print("---"*30)
print(df.tail())#Son beş değeri yazdırdım


# In[4]:


df.info()#Datasetimin bilgisini aldım


# In[5]:


df.shape#Satır sütün


# In[6]:


df.isnull().sum()#boş değer var mı sorgusunu yaptım


# In[7]:


df.describe().T# Datasetteki sütunların sayısal değerlerini aldım


# In[8]:


plt.figure(figsize=(14,8))
sns.heatmap(df.corr(), annot=True, cmap="hot")
plt.show()# figure oluştudum ve Datasetin korolesyonunu aldım(verirler arasında birbirini etkileme oranı diyebilirim)


# In[9]:


sns.pairplot(df, kind  ="reg");


# ### Aykırı değerleri gözlemliyorum

# In[10]:


df=df.select_dtypes(include=['float64','int64'] )
df_tbl=df["TotalSteps"].copy()
df_tbl.head()#TotalsTeps de bulunan aykırı değerlere göz atacağım


# In[11]:


sns.boxplot(x=df_tbl)#Aykırı değerler hakkında bilgi almak için boxplot


# In[12]:


Q1=df_tbl.quantile(0.25)
Q3=df_tbl.quantile(0.75)
IQR=Q3-Q1
print(Q1,IQR,Q3)
ust_sinir=Q3 + 1.5 * IQR
alt_sinir=Q1 - 1.5 * IQR
print("alt sınır: ",alt_sinir)
print("üst sınır: ",ust_sinir)#Verisetini çeyreklere bölerek alt ve üst sınırı belirledim


# In[13]:


aykiri=((df_tbl < (alt_sinir)) | (df_tbl> (ust_sinir)))
df[aykiri]# Alt ve üst sınıra uymayan değerleri ayırdım


# In[14]:


clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)#Aykırı değerleri ayıklamak için oluşturdum


# In[15]:


#her bir gözlem biriminin skoru
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:10]


# In[16]:


np.sort(df_scores)[0:20]
esik_deger = np.sort(df_scores)[9]#9dan sonra kırılım olduğu için seçtim
aykiri_tf = df_scores > esik_deger
aykiri_tf# aykırı değerleri çektim


# In[17]:


#aykırı gözlemler
df[df_scores < esik_deger] # Üretilen score değerleri eşikten (dokuzuncu veriden) küçük mü diye kontrol ettim


# In[18]:


df  = df[df_scores > esik_deger]
df


# In[19]:


df.shape


# In[20]:


df=df.select_dtypes(include=['float64','int64'] )
df_tbl=df["TotalSteps"].copy()
df_tbl.head()


# In[21]:


sns.boxplot(x=df_tbl)


# ### Linear Regression
# Sadece TotalSteps ve Calories arasıdaki bağıntıyı kontrol ettim

# #### Model
# Model kurlumunu ve Dataset için gerekli olan ayrımları yaptım.
# Eksik veri ayıklama, test verilerini ayırma işini burada hallettim

# In[22]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df= df.dropna()#Datasette boş olan değerleri çıkardım

y = df.drop(["TotalSteps"], axis = 1)#total_stepsi çıkararak y sütununa ekledik
X_ = df.drop(["Calories"], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(X_ , y , test_size=0.3 , random_state=42)# test verilerini oluşturduk

print("X_train", x_train.shape)
print("y_train",y_train.shape)
print("X_test",x_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[23]:


sns.jointplot(x = "TotalSteps", y = "Calories", data = df, kind = "reg")#Aralarındaki bağıntı


# In[24]:


X = df[["TotalSteps"]]
X[0:5]#Adımın ilk beş değeri


# In[25]:


X = sm.add_constant(data=X)
X[0:5]#Daha iyi gözlem  için dayanak olması içina const ekledim


# In[26]:


from warnings import filterwarnings
filterwarnings('ignore')
#uyarıları kapatmak icin


# In[27]:


#bağımlı değişkenimizi de alıyoruz
y = df["Calories"]
y[0:5]# Caloriesin ilk beş değeri


# In[28]:


#modelin kurulması
lm = sm.OLS(y,X)
#modelin fit edilmesi
model = lm.fit()
#model ciktilarinin alinmasi
model.summary()


# In[29]:


#sadece katsayıları görelim 
model.params


# In[30]:


model.mse_model


# In[31]:


#düzeltilmiş rkare değeri
model.rsquared_adj


# In[32]:


#modelden tahmin edilen y değerleri
model.fittedvalues[0:5]


# In[33]:


#gercek y degerleri
y[0:5]


# In[34]:


#modelin görsel olarak ifade edilmesi
g = sns.regplot(df["TotalSteps"], df["Calories"], ci=None, scatter_kws={'color':'r', 's':9})
g.set_title("Model")
g.set_ylabel("Adım Sayısı")
g.set_xlabel("Kalori")
import matplotlib.pyplot as plt
plt.xlim(-300,36100)
plt.ylim(bottom=0);


# In[35]:


import statsmodels.formula.api as smf
lm = smf.ols("Calories ~ TotalSteps", df)
model = lm.fit()
model.summary()


# In[36]:


mse = mean_squared_error(y, model.fittedvalues)
#gerçek değerler ile tahmin edilen değerler arasındaki farkların karelerinin ortalaması
mse


# In[37]:


#karşılaştırma tablosu
k_t = pd.DataFrame({"gercek_y": y[0:10],
                   "tahmin_y": model.predict(X)[0:10]})
k_t


# In[38]:


#artıkların görselleştirilmesi
#bu hatalar verisetindeki aykırı değerlerden mi kaynaklanıyor gibi soruların sorulabileceği 
#ve bu sorulara yanıt aramaya bizi sevk edecek gözlemler
plt.plot(model.resid)


# In[39]:


k_t["hata"] = k_t["gercek_y"] - k_t["tahmin_y"]
k_t["hata_kare"] = k_t["hata"]**2
k_t


# In[40]:


#toplam hata
np.sum(k_t["hata_kare"])


# In[41]:


#hata kareler ortalaması
np.mean(k_t["hata_kare"])


# In[42]:


#hata kareler ortalamasının karekoku
np.sqrt(np.mean(k_t["hata_kare"]))


# In[43]:


rmseTrain=np.sqrt( mean_squared_error(y, model.fittedvalues))
print("Eğitim Hatası: ",rmseTrain)


# ### Multi Linear Regression

# #### Model
# Bütün verileri kullanarak Linear Reg yaptım

# In[44]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.3 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# #### Statsmodel

# In[45]:


lm = sm.OLS(y_train, X_train)
model = lm.fit()
model.summary()


# #### TotalDistance değerini çıkarıp tekrar deniyoruz

# In[46]:


hit = pd.read_csv("Kalori.csv",usecols = [2,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.25 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[47]:


lm = sm.OLS(y_train, X_train)
model = lm.fit()
model.summary()


# #### Model kurulumunu ve test, eğitim verilerini girdim

# In[48]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.3 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[49]:


lm = LinearRegression()
model = lm.fit(X_train, y_train)


# Modelin hata çıktılarını aldım

# In[50]:


y_pred = model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(model.intercept_)
print(model.coef_)


# In[51]:


df.head()


# #### Model Tuning

# In[52]:


X = df.drop('Calories', axis=1)
y = df["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.25, random_state=42)
lm = LinearRegression() 
lojj_tunned = lm.fit(X_train, y_train)


# In[53]:


rmseTrain=np.sqrt(mean_squared_error(y_train, lojj_tunned.predict(X_train)))
print("Eğitim Hatası: ",rmseTrain)
rmseTest=np.sqrt(mean_squared_error(y_test, lojj_tunned.predict(X_test)))
print("Test Hatası: ",rmseTest)
lojj_tunned.score(X_train, y_train)


# In[54]:


#eğitim verileri için çapraz doğrulama ile elde edilmiş ortalama r2 skoru
cross_val_score(model, X_train, y_train, cv = 10, scoring = "r2").mean()


# In[55]:


#neg_mean_squared_error'da sonuc negatif olacağı için üstte (-) ile çarpıyoruz 
rmseTrain=np.sqrt(-cross_val_score(model, 
                X_train, 
                y_train, 
                cv = 10, 
                scoring = "neg_mean_squared_error")).mean()
print("Eğitim Hatası: ",rmseTrain)
rmseTest=np.sqrt(-cross_val_score(model, 
                X_test, 
                y_test, 
                cv = 10, 
                scoring = "neg_mean_squared_error")).mean()
print("Test Hatası: ",rmseTest)


# In[56]:


#karşılaştırma tablosu
k_t = pd.DataFrame({"Gercek": y[0:10],
                   "Tahmin": lojj_tunned.predict(X)[0:10]})
k_t["hata"] = k_t["Gercek"] - k_t["Tahmin"]
k_t["hata_kare"] = k_t["hata"]**2
k_t


# ### PCR (Temel Bileşen Regresyonu - Principal Component Regression)

# In[57]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.3 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[58]:


pca = PCA()
#indirgenmiş veriseti (transform indirgeme islemi)(scale islemi veri standardizasyonu yapmayi saglar)

X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_test = pca.fit_transform(scale(X_test))
#ilk gözlemin tüm bileşenlerde incelenmesi
X_reduced_train[0:1,:]
#Normalde PCA n_components yani bileşen sayısı parametresini alır. Eğer parametreyi vermezsek bütün bileşenleri kullanır.


# In[59]:


#açıklanan varyans oranı
np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)
#1. bilesen verisetindeki varyansın %38ini ifade ettiğini belirtir.
#2. bileşenin kendinden önceki bileşenle birlikte verisetindeki varyansın %59unu ifade ettiğini belirtir.
#3. bileşenin kendinden önceki bileşenlerle birlikte verisetindeki varyansın %70unu ifade ettiğini belirtir.
#...

#burada 10 bileşenle verisetindeki varyansın %97'sini ifade edebiliyoruz.


# In[60]:


lm = LinearRegression()
pcr_model = lm.fit(X_reduced_train, y_train)
print(pcr_model.intercept_)
print(pcr_model.coef_)


# #### Tahmin

# In[61]:


y_pred = pcr_model.predict(X_reduced_train)
y_pred[0:5]


# In[62]:


y_pred = pcr_model.predict(X_reduced_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = pcr_model.predict(X_reduced_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = pcr_model.predict(X_reduced_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = pcr_model.predict(X_reduced_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(pcr_model.intercept_)
print(pcr_model.coef_)


# In[63]:


df["Calories"].mean()


# #### Model Tuning

# In[64]:


pca2 = PCA()
X_reduced_test = pca2.fit_transform(scale(X_test))


# In[65]:


lm = LinearRegression()
#tüm bileşenlerle ile deneyelim
pcr_tuned = lm.fit(X_reduced_train, y_train)
y_pred = pcr_tuned.predict(X_reduced_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))


# In[66]:


#cross validation sayesinde olası bileşen sayılarıni gonderip optimum bileşen sayısını belirleyebiliriz 
from sklearn import model_selection
cv_10 = model_selection.KFold(n_splits = 10,
                             shuffle = True,
                             random_state = 42)
lm = LinearRegression()
RMSE = []


for i in np.arange(1, X_reduced_train.shape[1] + 1):
    
    score = np.sqrt(-1*model_selection.cross_val_score(lm, 
                                                       X_reduced_train[:,:i], 
                                                       y_train.ravel(), 
                                                       cv=cv_10, 
                                                       scoring='neg_mean_squared_error').mean())
    RMSE.append(score)


# In[67]:


import matplotlib.pyplot as plt
plt.plot(RMSE, '-v')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Maaş Tahmin Modeli İçin PCR Model Tuning');


# en iyi değer 10 gibi görünüyor

# In[68]:


y_pred = pcr_tuned.predict(X_reduced_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = pcr_tuned.predict(X_reduced_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = pcr_tuned.predict(X_reduced_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = pcr_tuned.predict(X_reduced_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(pcr_tuned.intercept_)
print(pcr_tuned.coef_)


# ### PLS (Kısmi En Küçük Kareler Regresyonu)

# In[69]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.3 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[70]:


#tum bilesenler alindi
pls_model = PLSRegression().fit(X_train, y_train)


# In[71]:


X_train.head()


# In[72]:


y_pred = pls_model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = pls_model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = pls_model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = pls_model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(pls_model.coef_)


# #### Model Tuning

# In[73]:


#CV
cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)


#Hata hesaplamak için döngü
RMSE = []

for i in np.arange(1, X_train.shape[1] + 1):
    pls = PLSRegression(n_components=i)
    score = np.sqrt(-1*cross_val_score(pls, X_train, y_train, cv=cv_10, scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

#Sonuçların Görselleştirilmesi
import matplotlib.pyplot as plt
plt.plot(np.arange(1, X_train.shape[1] + 1), np.array(RMSE), '-v', c = "r")
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Salary');


# In[74]:


pls_model = PLSRegression(n_components = 10).fit(X_train, y_train)


# In[75]:


y_pred = pls_model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = pls_model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = pls_model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = pls_model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(pls_model.coef_)


# ### Ridge Regression

# In[76]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.3 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[77]:


#alpha=lambda
ridge_model = Ridge(alpha = 0.1).fit(X_train, y_train)
ridge_model


# In[78]:


ridge_model.coef_


# In[79]:


#belirlenen aralıkta lambda değerleri oluşturuyoruz
lambdalar = 10**np.linspace(10,-2,100)*0.5 

ridge_model = Ridge()
katsayilar = []

for i in lambdalar:
    ridge_model.set_params(alpha = i)
    ridge_model.fit(X_train, y_train) 
    katsayilar.append(ridge_model.coef_) 
    

import matplotlib.pyplot as plt    
ax = plt.gca()
ax.plot(lambdalar, katsayilar) 
ax.set_xscale('log') 

plt.xlabel('Lambda Değerleri')
plt.ylabel('Katsayılar')
plt.title('Ridge Katsayıları');


# In[80]:


y_pred = ridge_model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = ridge_model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = ridge_model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = ridge_model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(ridge_model.intercept_)
print(ridge_model.coef_)


# #### Model Tuning

# In[81]:


lambdalar = 10**np.linspace(10,-2,100)*0.5 


# In[82]:


lambdalar[0:5]


# In[83]:


ridge_cv = RidgeCV(alphas = lambdalar, 
                   scoring = "neg_mean_squared_error",
                   normalize = True)


# In[84]:


ridge_cv.fit(X_train, y_train)


# In[85]:


ridge_cv.alpha_ #optimum lambda


# In[86]:


ridge_tuned = Ridge(alpha = ridge_cv.alpha_, 
                   normalize = True).fit(X_train,y_train)


# In[87]:


y_pred = ridge_model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = ridge_model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = ridge_model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = ridge_model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(ridge_model.intercept_)
print(ridge_model.coef_)


# ### Lasso Regression

# In[88]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.3 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[89]:


lasso_model = Lasso(alpha = 0.1).fit(X_train, y_train)
lasso_model


# In[90]:


y_pred = lasso_model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = lasso_model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = lasso_model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = lasso_model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(lasso_model.intercept_)
print(lasso_model.coef_)


# In[91]:


lasso = Lasso()
lambdalar = 10**np.linspace(10,-2,100)*0.5 
katsayilar = []

for i in lambdalar:
    lasso.set_params(alpha=i)
    lasso.fit(X_train, y_train)
    katsayilar.append(lasso.coef_)
  
import matplotlib.pyplot as plt  
ax = plt.gca()
ax.plot(lambdalar*2, katsayilar)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# In[92]:


lasso_model.predict(X_test) # Tahmin


# In[93]:


y_pred = lasso_model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = lasso_model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = lasso_model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = lasso_model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(lasso_model.intercept_)
print(lasso_model.coef_)


# #### Model Tuning

# In[94]:


lasso_cv_model = LassoCV(alphas = None, 
                         cv = 10, 
                         max_iter = 10000, 
                         normalize = True)


# In[95]:


lasso_cv_model.fit(X_train,y_train)


# In[96]:


lasso_cv_model.alpha_


# In[97]:


lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_)


# In[98]:


lasso_tuned.fit(X_train, y_train)


# In[99]:


y_pred = lasso_tuned.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = lasso_tuned.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = lasso_tuned.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = lasso_tuned.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(lasso_tuned.intercept_)
print(lasso_tuned.coef_)


# ### ElasticNet (eNet) Regression

# In[100]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.3 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[101]:


enet_model = ElasticNet().fit(X_train, y_train)


# In[102]:


enet_model.alpha


# In[103]:


y_pred = enet_model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = enet_model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = enet_model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = enet_model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(enet_model.intercept_)
print(enet_model.coef_)


# In[104]:


enet_model.predict(X_test)


# #### Model Tuning

# In[105]:


enet_cv_model = ElasticNetCV(cv = 10, random_state = 42).fit(X_train, y_train)


# In[106]:


enet_cv_model.alpha_


# In[107]:


enet_cv_model


# In[108]:


enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(X_train,y_train)


# In[109]:


y_pred = enet_tuned.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = enet_tuned.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = enet_tuned.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = enet_tuned.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(enet_tuned.intercept_)
print(enet_tuned.coef_)


# ### En Yakın Komşu
# Gözlem birimlerinin birbirine olan benzerlikleri üzerinden tahmin yapar

# In[110]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X = df.drop(["Calories"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.25 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[111]:


knn_model = KNeighborsRegressor().fit(X_train, y_train)


# In[112]:


y_pred = knn_model.predict(X_test)

print("test hatası:" , np.sqrt(mean_squared_error(y_test, y_pred)))

RMSE = [] 
#cross validation yapmadan hatalara bir bakalim
for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_train) 
    rmse = np.sqrt(mean_squared_error(y_train,y_pred)) 
    RMSE.append(rmse) 
    print("k =" , k , "için RMSE değeri: ", rmse)


# In[113]:


#GridSearchCV ile optimum k sayisinin belirlenmesi
knn_params = {'n_neighbors': np.arange(1,50,1)}
knn = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn, knn_params, cv = 10)
knn_cv_model.fit(X_train, y_train)


# In[114]:


knn_cv_model.best_params_["n_neighbors"]# en iyi değer


# In[115]:


RMSE = [] 
RMSE_CV = []
for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_train) 
    rmse = np.sqrt(mean_squared_error(y_train,y_pred)) 
    rmse_cv = np.sqrt(-1*cross_val_score(knn_model, X_train, y_train, cv=10, 
                                         scoring = "neg_mean_squared_error").mean())
    #cross validation olmadan hatalar
    RMSE.append(rmse) 
    #cross validation kullanilarak alindan hatalar
    RMSE_CV.append(rmse_cv)
    print("k =" , k , "için RMSE değeri: ", rmse, "RMSE_CV değeri: ", rmse_cv )


# In[116]:


knn_tuned = KNeighborsRegressor(n_neighbors =33)
knn_tuned.fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, knn_tuned.predict(X_test)))


# In[117]:


y_pred = knn_tuned.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = knn_tuned.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = knn_tuned.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = knn_tuned.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# ### Destek Vektör

# In[118]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.25 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[119]:


svr_model = LinearSVR().fit(X_train, y_train)


# In[120]:


y_pred = svr_model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = svr_model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = svr_model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = svr_model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(svr_model.intercept_)
print(svr_model.coef_)


# #### Model Tuning

# In[121]:


svr_params = {"C": np.arange(0.1,2,0.1)}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv = 10).fit(X_train,y_train)


# In[122]:


svr_cv_model.best_params_


# In[123]:


svr_tuned = LinearSVR(C = pd.Series(svr_cv_model.best_params_)[0]).fit(X_train, y_train)


# In[124]:


y_pred = svr_tuned.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = svr_tuned.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = svr_tuned.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = svr_tuned.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)
print(svr_tuned.intercept_)
print(svr_tuned.coef_)


# In[125]:


#bir veriseti olusturup lineer regresyon ile SVR'yi gorsel olarak karsilastiralim
np.random.seed(3)

x_sim = np.random.uniform(2, 10, 145)
y_sim = np.sin(x_sim) + np.random.normal(0, 0.4, 145)
#aykırı gözlemler
x_outliers = np.arange(2.5, 5, 0.5)
y_outliers = -5*np.ones(5)

x_sim_idx = np.argsort(np.concatenate([x_sim, x_outliers]))
x_sim = np.concatenate([x_sim, x_outliers])[x_sim_idx]
y_sim = np.concatenate([y_sim, y_outliers])[x_sim_idx]


# In[126]:


#lineer regresyon
ols = LinearRegression()
ols.fit(np.sin(x_sim[:, np.newaxis]), y_sim)
ols_pred = ols.predict(np.sin(x_sim[:, np.newaxis]))

#SVR
eps = 0.1 #default degeri
#rbf=radial bases function (dogrusal olmayan bir form)
svr = SVR(kernel='rbf', epsilon = eps)
svr.fit(x_sim[:, np.newaxis], y_sim)
svr_pred = svr.predict(x_sim[:, np.newaxis])


# In[127]:


plt.scatter(x_sim, y_sim, alpha=0.5, s=26)
plt_ols, = plt.plot(x_sim, ols_pred, 'g')
plt_svr, = plt.plot(x_sim, svr_pred, color='r')
plt.xlabel("Bağımsız Değişken")
plt.ylabel("Bağımlı Değişken")
plt.ylim(-5.2, 2.2)
plt.legend([plt_ols, plt_svr], ['EKK', 'SVR'], loc = 4);


# In[128]:


svr_rbf = SVR(kernel="rbf").fit(X_train, y_train)


# In[129]:


y_pred = svr_rbf.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = svr_rbf.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = svr_rbf.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = svr_rbf.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# In[130]:


svr_params = {"C": [6850000]}
svr_cv_model = GridSearchCV(svr_rbf,svr_params, cv = 10)
svr_cv_model.fit(X_train, y_train)


# In[131]:


svr_cv_model.best_params_


# In[132]:


svr_tuned = SVR(kernel="rbf", C = pd.Series(svr_cv_model.best_params_)[0]).fit(X_train, 
                                                                        y_train)


# In[133]:


y_pred = svr_tuned.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = svr_tuned.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = svr_tuned.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = svr_tuned.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# ### Yapar Sinir Ağları

# In[134]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.25 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[135]:


#değişken standartlaştırması
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[136]:


mlp_model = MLPRegressor().fit(X_train_scaled, y_train)
print(mlp_model)
print(mlp_model.n_layers_)
print(mlp_model.hidden_layer_sizes)


# In[137]:


y_pred = mlp_model.predict(X_train_scaled)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = mlp_model.predict(X_test_scaled)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = mlp_model.predict(X_test_scaled)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = mlp_model.predict(X_train_scaled)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# #### Model Tuning

# In[138]:


mlp_params = {'alpha': [0.7],'hidden_layer_sizes': [(500,250)],'activation': ['relu']}


# In[139]:


t1 =  datetime.now()
mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv = 10)
mlp_cv_model.fit(X_train_scaled, y_train)
t2 =  datetime.now()
print(t2-t1)


# In[140]:


mlp_model = MLPRegressor().fit(X_train_scaled, y_train)


# In[141]:


mlp_tuned = MLPRegressor(activation=mlp_cv_model.best_params_['activation'],
                         alpha = mlp_cv_model.best_params_['alpha'], 
                         hidden_layer_sizes = mlp_cv_model.best_params_['hidden_layer_sizes'])
print(mlp_tuned)
print(mlp_tuned.activation)


# In[142]:


mlp_tuned.fit(X_train_scaled, y_train)


# In[143]:


y_pred = mlp_tuned.predict(X_train_scaled)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = mlp_tuned.predict(X_test_scaled)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = mlp_tuned.predict(X_test_scaled)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = mlp_tuned.predict(X_train_scaled)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# ### Karar Ağaçları / Classification and Regression Trees (CART)
# Temel amaç veriseti içerisindeki karmaşık yapıları basit karar yapılarına dönüştürmektir.
# Heterojen verisetleri belirlenmiş bir hedef değişkene göre homojen alt gruplara ayrılır.
# 

# In[144]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.25 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[145]:


cart_model = DecisionTreeRegressor()
cart_model.fit(X_train, y_train)


# In[146]:


#ilkel test hatasina bir bakalim
cart_model = DecisionTreeRegressor()
cart_model.fit(X_train, y_train)
y_pred = cart_model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = cart_model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = cart_model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = cart_model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# #### Model Tuning

# In[147]:


#derleme uzun sürmesin diye bulduğum en uygun aralığı kapsayacak değerleri yazdım
cart_params = {"min_samples_split": [44],
               "max_leaf_nodes": [22]}
cart_cv_model = GridSearchCV(cart_model, cart_params, cv = 10)
cart_cv_model.fit(X_train, y_train)
print(cart_cv_model.best_params_)
cart_tuned = DecisionTreeRegressor(max_leaf_nodes = cart_cv_model.best_params_['max_leaf_nodes'], 
                                   min_samples_split = cart_cv_model.best_params_['min_samples_split'])
cart_tuned.fit(X_train, y_train)
y_pred = cart_tuned.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = cart_tuned.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = cart_tuned.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = cart_tuned.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# ### Random Forests
# Temeli birden çok karar ağacının ürettiği tahminlerin bir araya getirilerek değerlendirilmesidir.
# 

# In[148]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.25 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[149]:


rf_model = RandomForestRegressor(random_state = 42)
rf_model.fit(X_train, y_train)


# In[150]:


y_pred = rf_model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = rf_model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = rf_model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = rf_model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# #### Model Tuning

# In[151]:


#bulduğum en iyi parametreleri uzun sürmemesi için yazdım
rf_params = {'max_depth': [13],
            'max_features': [8],
            'n_estimators' : [2400]}
rf_model = RandomForestRegressor(random_state = 42)
rf_cv_model = GridSearchCV(rf_model, 
                           rf_params, 
                           cv = 10, 
                            n_jobs = -1)
rf_cv_model.fit(X_train, y_train)


# In[152]:



rf_cv_model.best_params_


# In[153]:


rf_tuned = RandomForestRegressor(max_depth  = rf_cv_model.best_params_['max_depth'], 
                                 max_features = rf_cv_model.best_params_['max_features'], 
                                 n_estimators =rf_cv_model.best_params_['n_estimators'])


# In[154]:


rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = rf_tuned.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = rf_tuned.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = rf_tuned.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# In[155]:


#degiskenlerin onem duzeyine bir bakalim.
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)


# In[156]:


Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Değişken Önem Düzeyleri")


# ### Gradient Boosting Machines (GBM)
# AdaBoost'un sınıflandırma ve regresyon problemlerine kolayca uyarlanabilen genellenmiş bir versiyonudur.
# 
# Adaptive Boosting (AdaBoost) zayıf sınıflandırıcıların bir araya gelerek güçlü bir sınıflandırıcı oluşturması için geliştirilen bir yöntemdir.

# In[157]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.25 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[158]:


gbm_model = GradientBoostingRegressor()
gbm_model.fit(X_train, y_train)


# In[159]:


y_pred = gbm_model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = gbm_model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = gbm_model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = gbm_model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# #### Model Tuning

# In[160]:


gbm_params = {
    'learning_rate': [0.15],
    'max_depth': [65],
    'n_estimators': [2030],
    'subsample': [0.65],
}


# In[161]:


#algoritmanın calisma zamanina da bir bakalim
t1 =  datetime.now()
gbm = GradientBoostingRegressor()
gbm_cv_model = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)
gbm_cv_model.fit(X_train, y_train)

t2 =  datetime.now()
print(t2-t1)


# In[162]:


gbm_cv_model.best_params_


# In[163]:


gbm_tuned = GradientBoostingRegressor(learning_rate = gbm_cv_model.best_params_['learning_rate'],  
                                      max_depth = gbm_cv_model.best_params_['max_depth'], 
                                      n_estimators = gbm_cv_model.best_params_['n_estimators'], 
                                      subsample = gbm_cv_model.best_params_['subsample'])

gbm_tuned = gbm_tuned.fit(X_train,y_train)


# In[164]:


y_pred = gbm_tuned.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = gbm_tuned.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = gbm_tuned.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = gbm_tuned.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# In[165]:


Importance = pd.DataFrame({"Importance": gbm_tuned.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Değişken Önem Düzeyleri")


# ### XGBoost (eXtreme Gradient Boosting)
# XGBoost, GBM'in hız ve tahmin performansını arttırmak üzere optimize edilmiş; ölçeklenebilir ve farklı platformlara entegre edilebilir halidir.

# In[166]:


hit = pd.read_csv("Kalori.csv",usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14])
df = hit.copy()
df.dropna(inplace=True)
y = df["Calories"]
X_ = df.drop(["Calories"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_ , y , test_size=0.25 , random_state=42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)
training = df.copy()
print("training", training.shape)


# In[167]:


#pandas ya da numpy yerine kendi dataframeini kullanırsaniz daha performanslı calıstigi soylenmis.
DM_train = xgb.DMatrix(data = X_train, label = y_train)
DM_test = xgb.DMatrix(data = X_test, label = y_test)


# In[168]:


#pandas dataframei ile de kullanılabilir.
xgb_model = XGBRegressor().fit(X_train, y_train)


# In[169]:


y_pred = xgb_model.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = xgb_model.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = xgb_model.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = xgb_model.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# #### Model Tuning

# In[170]:


xgb_grid = {
     'colsample_bytree': [0.51], 
     'n_estimators':[1885],
     'max_depth': [41],
     'learning_rate': [0.38]
}


# In[171]:


# bulduğum en iyi parametreleri uygulama derlemesi uzun sürmemesi için yazdım
t1 =  datetime.now()

xgb = XGBRegressor()
xgb_cv = GridSearchCV(xgb, 
                      param_grid = xgb_grid, 
                      cv = 10, 
                      n_jobs = -1,
                      verbose = 2)

xgb_cv.fit(X_train, y_train)

t2 =  datetime.now()
print(t2-t1)
print(xgb_cv.best_params_)


# In[172]:


xgb_tuned = XGBRegressor(colsample_bytree = xgb_cv.best_params_['colsample_bytree'], 
                         learning_rate = xgb_cv.best_params_['learning_rate'], 
                         max_depth = xgb_cv.best_params_['max_depth'], 
                         n_estimators = xgb_cv.best_params_['n_estimators']) 

xgb_tuned = xgb_tuned.fit(X_train,y_train)


# In[173]:


y_pred = xgb_tuned.predict(X_train)
print("Train R2 Score: ", r2_score(y_train, y_pred))
print("----"*30)
y_pred = xgb_tuned.predict(X_test)
print("Test R2 Score: ", r2_score(y_test, y_pred))
print("----"*30)
y_pred = xgb_tuned.predict(X_test)
print("Ortalama Test Hatası: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("----"*30)
y_pred = xgb_tuned.predict(X_train)
print("Ortalama Eğitim Hatası: ", np.sqrt(mean_squared_error(y_train,y_pred)))
print("----"*30)


# ### Tüm Modellerin Karşılaştırılması

# In[174]:


modeller = [
    lojj_tunned,
    pls_model,
    ridge_tuned,
    lasso_tuned,
    enet_tuned,
    knn_tuned,
    svr_tuned,
    mlp_tuned,
    cart_tuned,
    rf_tuned,
    gbm_tuned,
    xgb_tuned    
]

for model in modeller:
    if(model!=mlp_tuned):
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test_scaled)    
        
    dogruluk = r2_score(y_test, y_pred)
    isimler = model.__class__.__name__
    print("-"*30)
    print(isimler + ":" )
    print("R2 Skor: {:.4%}".format(dogruluk))


# RandomForestRegressor en iyi sonucu veriyor

# In[175]:


sonuc = []

sonuclar = pd.DataFrame(columns= ["Modeller","Accuracy"])

for model in modeller:
    
    if(model!=mlp_tuned):
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test_scaled)

    dogruluk = r2_score(y_test, y_pred)    
    isimler = model.__class__.__name__
    sonuc = pd.DataFrame([[isimler, dogruluk*100]], columns= ["Modeller","Accuracy"])
    sonuclar = sonuclar.append(sonuc)
    
    
sns.barplot(x= 'Accuracy', y = 'Modeller', data=sonuclar, color="r")
plt.xlabel('R2 Score')
plt.title('Modellerin Doğruluk Oranları');    


# ### Sonuç

#    Seçilen modeller arasında verisetimize en uygun çalışanı, Random Forest Regressor oldu. Verisetim de Kalori yakmayı en çok etkileyen veriyi elde etmeye çalıştım. Antremanın sonlarına doğru aktifliğin en çok olduğu VeryActiveMinutes adında ki değerlerin ciddi oranda etkilediğini gözlemledim. Her ne kadar toplam adım, toplam mesafe, süre gibi değişkenler çok önemli gözükmese de VeryActiveMinutes değeri bu değerlerin sonucu ortaya çıkıyor. Yani bağımsız değişken olarak aldığımız değişkende aslında başka bir dizi değerin bağımlı değişkeni konumundadır. Veri setini yorumlama ve bir bütün olarak gözlemlemek için çok büyük kolaylık sağlıyor.

# Yukarıda modelleri hem grafik olarak hem de yazılı olarak belirttim.

# In[176]:


modeller = [
    lojj_tunned,
    pls_model,
    ridge_tuned,
    lasso_tuned,
    enet_tuned,
    knn_tuned,
    svr_tuned,
    mlp_tuned,
    cart_tuned,
    rf_tuned,
    gbm_tuned,
    xgb_tuned    
]

for model in modeller:
    if(model!=mlp_tuned):
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test_scaled)    
        
    dogruluk =  np.sqrt(mean_squared_error(y_test, y_pred))
    isimler = model.__class__.__name__
    print("-"*30)
    print(isimler + ":" )
    print("Hata katsayıları : {:.4}".format(dogruluk))


# Linear Regression, Pl
