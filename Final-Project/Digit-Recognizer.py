import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
import numpy as np

df=pd.read_csv('Digit-Recognizer-Data/train.csv')
print df.describe()
print df.isnull().sum().sum()

x_train=np.reshape(np.array(df.iloc[0:29400, 1:783]), [29400, 782])
y_train=np.reshape(np.array(df.iloc[0:29400, 0]), [29400, 1])

x_test=np.reshape(np.array(df.iloc[29400:42001, 1:783]), [12600, 782])
y_test=np.reshape(np.array(df.iloc[29400:42001, 0]), [12600, 1])

x_scaled=StandardScaler().fit_transform(x_train)
x_norm=normalize(x_train, norm='max')
#x_pca=x_train
#pca=PCA(n_components=781)
#y_pca=np.asarray(pca.fit_transform(x_pca), dtype="|S6")

rf=RandomForestClassifier(n_estimators=64)
rf_scaled=RandomForestClassifier(n_estimators=64)
rf_norm=RandomForestClassifier(n_estimators=64)
#rf_pca=RandomForestClassifier(n_estimators=64)

rf.fit(x_train, np.ravel(y_train))
rf_scaled.fit(x_scaled, np.ravel(y_train))
rf_norm.fit(x_norm, np.ravel(y_train))
#rf_pca.fit(x_pca, y_pca)

print('Score with Random Forest: ')
print rf.score(x_test, y_test)
print('Score with scaling: ')
print rf_scaled.score(x_test, y_test)
print('Score with normalization: ')
print rf_norm.score(x_test, y_test)
#print rf_pca.score(x_test, y_test)
