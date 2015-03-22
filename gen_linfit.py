import numpy as _nm
from scipy import linalg as _linalg

def fit(xdata,ydata,yerror,Y):

    # form the A matrix, from Garcia eqn 5.25
    A = _nm.zeros((len(ydata),len(Y)))
    for j in range(len(Y)):
        A[:,j] = Y[j](xdata) / yerror

    # Garcia, below eqn 5.27
    b = ydata / yerror

    C = _linalg.inv( _nm.dot(_nm.transpose(A) , A) )

    params = _nm.dot( _nm.dot(C , _nm.transpose(A)) , b)
    params_cov = C

    return params,params_cov

def calc_model(xdata,params,Y):
	
    for i in range(len(Y)):
        if i == 0:
            ymodel = params[i]*Y[i](xdata)
        else:
            ymodel = ymodel + params[i]*Y[i](xdata)

    return ymodel
