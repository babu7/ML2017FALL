#!/usr/bin/env python3
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def read_dataset():
    tmp, tmp_, train_pixels, train_labels_ = makedata(0)
    train_labels = []
    for i in train_labels_:
        train_labels.append(np.where(i>0)[0][0])
    return np.asarray(train_pixels), np.asarray(train_labels)


def main():
    model_path = 'weights_63509.h5'
    emotion_classifier = load_model(model_path)
    np.set_printoptions(precision=2)
    # dev_feats,te_labels = read_dataset()
    import cnn_train
    face = cnn_train.load_data()
    face.labels = face.labels.argmax(axis=1)
    dev_feats,te_labels = face.inputs, face.labels
    print(face.inputs.shape)
    print(face.labels.shape)
    predictions = emotion_classifier.predict(dev_feats)
    predictions = predictions.argmax(axis=-1)
    print (predictions)
    #te_labels = get_labels('../test_with_ans_labels.pkl')
    print (te_labels)
    conf_mat = confusion_matrix(te_labels,predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    # plt.show()
    img = plt.gcf()
    img.savefig('confu_matrix.png')

if __name__=='__main__':
    main()
