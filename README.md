import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('/content/iris.csv')
df
%matplotlib inline
img=mpimg.imread('/content/gayu.jpg')
plt.figure(figsize=(10,10))
plt.imshow(img)
x = df.iloc[:, :4].values  # Replace dataset with df
y = df['Species'].values   # Replace dataset with df and correct the column name to 'Species'
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.naive_bayes import GaussianNB # Corrected the typo in the module name
nvclassifier = GaussianNB()  # Corrected the typo in the class name
nvclassifier.fit(x_train, y_train)
from sklearn.metrics import confusion_matrix
y_pred = nvclassifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
a= cm.shape
corrpred=0
falsepred=0.1
for row in range(a[0]):
  for c in range(a[1]):
    if row==c:
      corrpred+=cm[row,c]
    else:
      falsepred+=cm[row,c]
print('correct predictions:',corrpred)
print('False predictions',falsepred)
print('\n\nAccuracy of the KNN Classification is :', corrpred/(cm.sum()))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)
classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
y_pred
