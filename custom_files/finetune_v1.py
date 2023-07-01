import glob
import os

import matplotlib.pyplot as plt
from custom_files.utils import saveload
import numpy as np
import tensorflow as tf


detmodel = "retinaface"
embmodel1 = "Facenet512"  #512
embmodel2 = "SFace"  # 128
embmodel3 = "ArcFace"  # 512
#embmodel4 = "Dlib"  # 512
embmodel5 = "VGG-Face"  #2622
align = False
deepalign = not align
model_embeddings = '../deepface/{}_align{}_{}_{}_{}_{}_embeddings'.format(detmodel,align, embmodel1,embmodel2, embmodel3, embmodel5)

# prepare train data
if not os.path.isfile('finetune_5492.pickle'):
    [train_embeddings] = saveload('load',model_embeddings,1)
    [test_embeddings] = saveload('load',model_embeddings+'_pred_5492n',1)

    names = []
    embs = []
    for name, emb in train_embeddings:
        names.append(name)
        embs.append(emb)

    unique_names = list(set(names))
    unique_labels = np.arange(len(unique_names))

    for name, emb,facemat in test_embeddings:
        names.append(name)
        embs.append(emb)

    labels = []
    for name in names:
        idx = unique_names.index(name)
        labels.append(unique_labels[idx])

    s = np.arange(len(labels))
    np.random.seed(2)
    np.random.shuffle(s)
    embs = np.vstack(embs)[s]
    labels = np.array(labels)[s]

    saveload('save','finetune_{}'.format(len(labels)),[embs,labels])
else:
    [embs,labels] = saveload('load','finetune_5492',1)

print(embs[0])
print(labels)

trsplit = int(0.8*len(labels))
x = tf.cast(embs[:trsplit],dtype=tf.float32)
y = tf.cast(labels[:trsplit],dtype=tf.int64)
vx = tf.cast(embs[trsplit:],dtype=tf.float32)
vy = tf.cast(labels[trsplit:],dtype=tf.int64)

N = 1024
batchsize = 256
model_name = 'toplayer_n{}_h{}_b{}'.format(y.shape[0], N, batchsize)

# create model
train = True
# if glob.glob("./model_weights/toplayer*"):
#     train= False
# else:
#     train = True

class FaceRecogEnsemModelTop(tf.keras.Model):
    def __init__(self, num_class=len(np.unique(labels))):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(N, activation='relu')
        self.fc2 = tf.keras.layers.Dense(N, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_class, activation='softmax')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        z = self.classifier(x)
        return z


topmodel = FaceRecogEnsemModelTop()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
lossfunc = tf.keras.losses.sparse_categorical_crossentropy
topmodel.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics='accuracy')

unt_yhat = topmodel(x)
unt_pred = np.argmax(unt_yhat,axis=1)
unt_acc = np.sum(y == unt_pred)/len(y)
print("Untrained model, accuracy: {:5.2f}%".format(100 * unt_acc))

# train model
if train:
    history = topmodel.fit(x=x, y=y, epochs=10, batch_size=batchsize, validation_data=(vx,vy))
    topmodel.save_weights('./model_weights/'+model_name)
    print("Model saved")

    plt.figure()
    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.subplot(122)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.suptitle(model_name+' tr {:5.2f}, val {:5.2f}'.format(100 * history.history['accuracy'][-1], 100 * history.history['val_accuracy'][-1]))
    plt.savefig('./train_fig/'+model_name+'.png')
    plt.show()
    print("Trained model, train on {} accuracy: {:5.2f}%".format(y.shape[0], 100 * history.history['accuracy'][-1]))
    print("Trained model, val accuracy: {:5.2f}%".format(100 * history.history['val_accuracy'][-1]))


class TopModel(tf.keras.Model):
    def __init__(self, num_class=len(np.unique(labels))):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(N, activation='relu')
        self.fc2 = tf.keras.layers.Dense(N, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_class, activation='softmax')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        z = self.classifier(x)
        return x, z


model = TopModel()
#model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics='accuracy')
model.load_weights('./model_weights/'+model_name)
featlayer, yhat = model(embs)
pred = np.argmax(yhat,axis=1)
acc = np.sum(labels == pred)/len(labels)
print("Trained model, full x accuracy: {:5.2f}%".format(100 * acc))
