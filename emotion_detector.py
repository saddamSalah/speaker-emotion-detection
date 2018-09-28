import pandas as pd
import numpy as np
from utils.feature_extractor import get_mfcc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
input_data = pd.read_csv('utils/training_files.csv')
train_data = pd.DataFrame()
train_data['Path'] = input_data['Path']
input_data = train_data['Path'].apply(get_mfcc)
input_data.to_csv('features.csv')
print('done loading train mfcc')

train_data = pd.read_csv('features.csv')
train_data['Label'] = input_data['Label']
X, y = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
y = LabelEncoder().fit_transform(y)

# Anger ---> 0
# Disgust --> 1
# Fear -----> 2
# Happy ----> 3
# Neutral --> 4
# Sad ------> 5
# Surprise --> 6


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
A_count = D_count = F_count= H_count= N_count= S_count= Su_count = 0
for i in range(len(y_test)):
    if y_test[i] == 0:
        A_count = A_count + 1
    elif y_test[i] == 1:
        D_count = D_count + 1
    elif y_test[i] == 2:
        F_count = F_count + 1
    elif y_test[i] == 3:
        H_count = H_count + 1
    elif y_test[i] == 4:
        N_count = N_count + 1
    elif y_test[i] == 5:
        S_count = S_count + 1
    else:
        Su_count = Su_count + 1
print(str(A_count) + ' ' + str(D_count) + ' ' + str(F_count) + ' ' + str(H_count) + ' ' + str(N_count) + ' ' +
      str(S_count) + ' ' + str(Su_count))

pipe_model = Pipeline([('standard_scalar', StandardScaler()),
                       # ('pca', PCA(n_components=10)),
                        ('svc', GradientBoostingClassifier())
                       ])
pipe_model.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_model.score(X_test, y_test))
print('Training Accuracy: %.3f' % pipe_model.score(X_train, y_train))
print('Test F1 score: %.3F' % f1_score(y_test, pipe_model.predict(X_test), average='weighted'))




scores = cross_val_score(estimator=pipe_model,
                         X=X_train,
                         y=y_train,
                         n_jobs=1,
                         cv=10)
print('Cross validation scores: %s' % scores)


FILE_NAME = 'mix_emotion_detector.sav'
pickle.dump(pipe_model, open(FILE_NAME, 'wb'))
pipe_model2 = pickle.load(open(FILE_NAME, 'rb'))

C = confusion_matrix(y_test, pipe_model.predict(X_test))
df = pd.DataFrame(C, index=['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
plt.figure(figsize=(10, 7))
sns.heatmap(df, annot=True, fmt='g')
plt.show()
print(C)