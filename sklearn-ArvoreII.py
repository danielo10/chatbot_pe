import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import tree
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['STATUS','Quartos','Banheiros','sqft_living','sqrf_lot','Floor','Condition','sqft_above','sqft_basement']
casas = pd.read_csv("ArvoreIISK.csv", header=None, names=col_names)
#casas = casas.drop(['STATUS','Quartos','Banheiros','Nota','sqft_living','sqrf_lot','Floor','Condition','sqft_above','sqft_basement'], axis=1)
casas.head()
feature_cols = ['Quartos','Banheiros','sqft_living','sqrf_lot','Floor','Condition','sqft_above','sqft_basement']
X = casas[feature_cols] # Features
y = casas.STATUS # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1)
#y_test = pd.read_csv("ConferenciaIISK.csv")
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


with open("ArvoreII.dot", 'w') as f:
    f = tree.export_graphviz(clf,
                             out_file=f,
                             max_depth=20,
                             impurity=True,
                             feature_names = feature_cols,
                             class_names = ['Abaixo','Media','Acima'],
                             rounded= True,
                             filled = True)
