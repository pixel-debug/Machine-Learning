import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def plot_decision_boundary(model,X,y,normalize=False):
    #codigo obtido de: https://markd87.github.io/articles/ml.html
    padding=0.15
    res=0.01

    x_axis_list = X.to_numpy()[:,0] if type(X) == pd.DataFrame else X[:,0]
    y_axis_list = X.to_numpy()[:,1] if type(X) == pd.DataFrame else X[:,1]

    #max and min values of x and y of the dataset
    x_min,x_max=x_axis_list.min(), x_axis_list.max()
    y_min,y_max=y_axis_list.min(), y_axis_list.max()

    #normaliza os valores de X e Y (se necessario)
    if(normalize):
        x_axis_list /= x_max
        y_axis_list /= y_max

    #range of x's and y's
    x_range=x_max-x_min
    y_range=y_max-y_min

    #add padding to the ranges
    x_min -= x_range * padding
    y_min -= y_range * padding
    x_max += x_range * padding
    y_max += y_range * padding

    #create a meshgrid of points with the above ranges
    xx,yy=np.meshgrid(np.arange(x_min,x_max,res),np.arange(y_min,y_max,res))

    #use model to predict class at each point on the grid
    #ravel turns the 2d arrays into vectors
    #c_ concatenates the vectors to create one long vector on which to perform prediction
    #finally the vector of prediction is reshaped to the original data shape.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #plot the contours on the grid
    plt.figure(figsize=(8,6))
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    #plot the original data and labels
    plt.scatter(x_axis_list, y_axis_list, s=35, c=y, cmap=plt.cm.Spectral)
