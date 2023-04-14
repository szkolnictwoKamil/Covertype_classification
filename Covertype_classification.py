import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow as tf
from collections import defaultdict


def heuristic_method(x_train, y_train, x_test, y_test):
    '''
    The heuristic method calculates the means of the features for every tree type and then does the same for every record.
    It then decides which type to choose by taking the minimal difference of means. It returns the predicted classes for every row.
    '''
    df_train = x_train
    df_train['class'] = y_train
    df_grupped = df_train.groupby('class')[[i for i in range(0,54)]].mean()
    df_mean = list(df_grupped.mean(axis=1)) # training means

    df_test = x_test
    df_test['class'] = y_test
    df_test['mean_1'] = df_test.mean(axis=1)
    df_test = df_test.loc[:, ['class','mean_1']]
    df_test = df_test.reset_index()
    del df_test['index']
    df_test

    pred = []
    for (_, row) in df_test.iterrows():
        diffrences = [abs(row['mean_1'] - i) for i in df_mean] #choosing the class based on the minimal difference of means
        pred.append(diffrences.index(min(diffrences))+1)
      
    return pred  

def random_forest(x_train, y_train, x_test, y_test):
    '''
    Function calculates the predicted class using the random forest classifier. 
    '''
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rfc.fit(x_train, y_train)
    pred = rfc.predict(x_test)
    return pred
    
def knn_method(x_train, y_train, x_test, y_test):
    '''
    Function calculates the predicted class using the knn (with 3 neighbours) classifier. 
    '''
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    return pred


def create_model():
    '''
    function to create a neural network. The number of hidden layers and neurons were chosen manually after conducting a few tests.
    '''
    model = Sequential()
    model.add(Dense(54, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(35, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def choose_best_hyperparameters():
    '''
    Function that prints out the best hyperparameters (batch size and number of epochs for the chosen neural network)
    '''
    keras_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)
    batch_size = [250, 500, 1000]
    epochs = [50, 70, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(x, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # best results were {'batch_size': 250, 'epochs': 100}
   
def neural_network(x_train, y_train, x_test, y_test):
    '''
    Function calculates the predicted class using neural networks. 
    '''
    model = create_model()
    model_history = model.fit(x_train, y_train, validation_split=0.33, batch_size=250, epochs=100)
    y_pred = model.predict(x_test)
    pred = np.argmax(y_pred, axis=1)
    return pred
 
def evaluate_model(y_test, y_pred):
    '''
    Function to calculate the metrics.
    '''
    # Calculate accuracy, precision, recall, f1-score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred,average='macro')
    rec = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

if __name__ == '__main__':
    # uploading the file
    data = pd.read_csv('covtype.data', sep=',', header=None)
    
    # Let us scale numerical data to avoid outlier presence significantly affecting the model
    scaler = StandardScaler()
    data.iloc[:,:10] = scaler.fit_transform(data.iloc[:,:10])

    # Let us separate independent and dependent variables
    x = data.iloc[:,:-1]
    y = data.iloc[:, -1]

    # oversampling the training set to handle class imbalance
    smote = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
    x,y = smote.fit_resample(X_train,y_train)
    
    model = input('Please enter the number indicating the method you would like to choose: \n 0: heuristic \n 1: random forest \n 2: knn, \n 3: neural network')
    models =[heuristic_method, random_forest, knn_method, neural_network]
    method = models[int(model)]
    pred = method(x,y,X_test,y_test)
    results = evaluate_model(y_test, pred)
    print(results)

