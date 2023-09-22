from csv import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def loaddata():
    #Process csv
    file1 = open('/Users/kiran/Desktop/1ACode/Practice/Iris_Classifier/iris0.csv',newline='')
    file2 = open('/Users/kiran/Desktop/1ACode/Practice/Iris_Classifier/iris1.csv',newline='')
    file3 = open('/Users/kiran/Desktop/1ACode/Practice/Iris_Classifier/iris2.csv',newline='')
    iris1 = reader(file1, delimiter=',',)
    iris2 = reader(file2, delimiter=',',)
    iris3 = reader(file3, delimiter=',',)
    irislist1 = [row for row in iris1]
    irislist2 = [row for row in iris2]
    irislist3 = [row for row in iris3]
    #Make training and test data
    iristrain = []
    for x in [irislist1,irislist2,irislist3]: iristrain.extend(x[0:-6])
    iristest = []
    for x in [irislist1,irislist2,irislist3]: iristest.extend(x[-6:])
    #Process train data
    trainarr = np.array([[x for x in iristrain[y]] for y in range(len(iristrain)-1)])
    x_train = trainarr[:,0:4].astype(float)
    y_train = trainarr[:,4]
    y_train[y_train=='Iris-virginica']=2
    y_train[y_train=='Iris-versicolor']=1
    y_train[y_train=='Iris-setosa']=0
    y_train = y_train.astype(float)
    #Process test data
    testarr = np.array([[x for x in iristest[y]] for y in range(len(iristest)-1)])
    x_test = testarr[:,0:4].astype(float)
    y_test = testarr[:,4]
    y_test[y_test=='Iris-virginica']=2
    y_test[y_test=='Iris-versicolor']=1
    y_test[y_test=='Iris-setosa']=0
    y_test = y_test.astype(float)
    return x_train,y_train,x_test,y_test

#Create Model
def makemodel():
    model = tf.keras.models.Sequential([])
    model.add(tf.keras.layers.Input(shape=(4,)))
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dense(50,activation='softmax'))
    model.add(tf.keras.layers.Dense(25,activation='elu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(3,activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
def save(model) :
    model.save('model')
def loadmodel():
    return tf.keras.models.load_model('model')
def evaluate(model,x_test,y_test):
    model.evaluate(x_test,y_test)
def fit(model,x,y):
    model.fit(x,y,epochs=300, batch_size=int(x.shape[0]/3))
def predict(model,a,b,c,d):
    predicts= list(model.predict(np.array([[a,b,c,d]]))[0]*100)
    print(predicts)
    number = predicts.index(max(predicts))
    name = ""
    if number == 0: name ='Iris-setosa'
    elif number == 1: name = 'Iris-versicolor'
    elif number == 2: name = 'Iris-virginica'
    result = name,(str(max(predicts))+"% probability")
    return result

a,b,c,d = loaddata()
model = makemodel()
fit(model,a,b)
acc = model.history.history['accuracy']
loss = model.history.history['loss']
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.plot([x for x in range(len(acc))],acc,label = 'Accuracy',color='b')
plt.plot(loss, label='Loss', color='y')
plt.legend()
plt.show()
evaluate(model,c,d)