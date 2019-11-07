# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 20:12:34 2019

@author: Steven
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from skimage.color import rgb2lab
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

OUTPUT_TEMPLATE = (
    'Bayesian classifier:    {bayes_rgb:.3f} {bayes_convert:.3f}\n'
    'kNN classifier:         {knn_rgb:.3f} {knn_convert:.3f}\n'
    'Rand forest classifier: {rf_rgb:.3f} {rf_convert:.3f}\n'
)


# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 114, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'pink': (255, 187, 187),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])


def plot_predictions(model, lum=71, resolution=256):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of LAB colour values, with L=lum
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    # convert to RGB for consistency with original input
    X_grid = lab2rgb(lab_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((-1, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at L=%g' % (lum,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.ylabel('B')
    plt.imshow(X_grid.reshape((hei, wid, -1)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)

def rgbtolab(X):
    X = X.reshape(1,-1,3)
    X = rgb2lab(X)
    X = X.reshape(-1,3)
    
    return X


def main():
    data = pd.read_csv(sys.argv[1])
    X = data[['R', 'G', 'B']].values / 255
    y = data['Label'].values

    # TODO: create some models
    
    #rgb = X[['R','G','B']]/255
    #names = y['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # TODO: build model_rgb to predict y from X.
    
    model_rgb = GaussianNB()
    
    # TODO: print model_rgb's accuracy score
    
    #model_rgb.fit(X_train, y_train)
    #print(model_rgb.score(X_test,y_test))

    # TODO: build model_lab to predict y from X by converting to LAB colour first.
    
    #from lecture ML: classification
    model_lab = make_pipeline(
            FunctionTransformer(rgbtolab, validate = True),
            GaussianNB()
    )
    #model_lab.fit(X_train, y_train)
    
    model_kneighbour_rgb = KNeighborsClassifier(n_neighbors = 9)
    #model_kneighbour_rgb.fit(X_train, y_train);
    
    model_kneighbour_lab = make_pipeline(
        FunctionTransformer(rgbtolab, validate = True),
        KNeighborsClassifier(n_neighbors = 9)        
    )
    #model_kneighbour_lab.fit(X_train, y_train)
    
    model_svc_rgb = SVC(kernel='linear', C=1)
    #model_svc_rgb.fit(X_train, y_train)
    
    model_svc_lab = make_pipeline(
        FunctionTransformer(rgbtolab, validate = True),
        SVC(kernel='linear', C=1)        
    )
    #model_svc_lab.fit(X_train, y_train)


    # train each model and output image of predictions
    models = [model_rgb, model_lab, model_kneighbour_rgb, model_kneighbour_lab, model_svc_rgb, model_svc_lab]
    for i, m in enumerate(models):  # yes, you can leave this loop in if you want.
        m.fit(X_train, y_train)
        plot_predictions(m)
        plt.savefig('predictions-%i.png' % (i,))

    print(OUTPUT_TEMPLATE.format(
        bayes_rgb=model_rgb.score(X_test, y_test),
        bayes_convert=model_lab.score(X_test, y_test),
        knn_rgb=model_kneighbour_rgb.score(X_test, y_test),
        knn_convert=model_kneighbour_lab.score(X_test, y_test),
        rf_rgb=model_svc_rgb.score(X_test, y_test),
        rf_convert=model_svc_lab.score(X_test, y_test),
    ))


if __name__ == '__main__':
    main()
