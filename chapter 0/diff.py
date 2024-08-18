import numpy as np
import scipy as scipy

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

x1 = 1
x2 = 2
x3 = 0
y = softmax([x1, x2, x3])
dx = 0.001
dy = softmax([x1 + dx, x2, x3]) - softmax([x1, x2, x3])
dy_dx1_numerical = dy/dx

s1, s2, s3 = y
dy_dx3_formula = [s1*(1 - s1), -s1*s2, -s1*s3]

x = 0.1
dx = 0.001
y = np.sin(np.log(x))
dy_dx_numerical = (np.sin(np.log(x+dx)) - np.sin(np.log(x)))/dx
dy_dx = np.cos(np.log(x))/x

print('done')