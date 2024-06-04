import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.discriminant_analysis import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as skmet

# Load the dataset
df = pd.read_csv('wdbc.csv', header=None)
df.columns = ['id','diagnosis']+[f'feature_{i}' for i in range(30)]
# df.info()


# Split the dataset 
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']
# print(X.shape)
# print(y.shape)

# Encoding the class labels (M = 1, B = 0)
le = LabelEncoder()
y = le.fit_transform(y)

# Standardize the data
standardize = StandardScaler()
X = standardize.fit_transform(X)


def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    y_pred = cross_val_predict(model, X, y, cv=5)
    print("Confusion Matrix:")
    print(skmet.confusion_matrix(y, y_pred))
    print("\nClassification Report:")
    print(skmet.classification_report(y, y_pred))
    print("Cross-Validation Scores:")
    print(scores)


# Testing different models
# KNN
print("====================================")
print("KNN Results")
print("====================================")
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn, X, y)

# Decision Tree
print("====================================")
print("Decision Tree Results")
print("====================================")
tree_clf = DecisionTreeClassifier(random_state=32)
evaluate_model(tree_clf, X, y)

# SVM
print("====================================")
print("SVM Results")
print("====================================")
svm = SVC(kernel='linear', random_state=32)
evaluate_model(svm, X, y)

# Naive Bayes
print("====================================")
print("Naive Bayes Results")
print("====================================")
nb = GaussianNB()
evaluate_model(nb, X, y)