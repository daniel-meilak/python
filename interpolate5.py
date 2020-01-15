from scipy.interpolate import interp1d
from scipy.optimize import least_squares, fmin
import matplotlib.pyplot as plt
import numpy as np

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
# define an interpolation function
def interpolate(x,y,xnew):
    function=interp1d(x,y,kind='cubic')
    return function(xnew)

# residual fuction to collapse curves
def residual(param,x,y,r):
    beta = param[0]
    nu   = param[1]
    Tc   = param[2]
    x    = (x - Tc)/Tc
    r    = r.astype(int)
    p_b  = np.zeros(x.shape[0])

    # for each data set p
    for p in range(0,y.shape[0],1):

        # for each data set j from p+1 to max_sets
        for j in range(p+1,y.shape[0],1):

            # rescale temperature x_p and magnetization y_p
            x_p  = (r[j]/r[p])**(1/nu)*x
            y_p  = (r[j]/r[p])**(beta/nu)*y[p]

            # create an interpolated function of data set p
            curve_p = interp1d(x_p,y_p,kind='cubic')

            # rescale temperature x_j and magnetization y_j
            x_j = x
            y_j = y[j]

            # create an interpolated function of data set j
            curve_j = interp1d(x_j,y_j,kind='cubic')

            # find common range
            xmin = np.amax([np.amin(x_p),np.amin(x_j)])
            xmax = np.amin([np.amax(x_p),np.amax(x_j)])

            if (np.amin(x_p) <= np.amin(x_j)):
                p_b = p_b + (y_j - curve_p(x_j))**2
            else:
                p_b = p_b + (y_p - curve_j(x_p))**2

    return p_b
#-------------------------------------------------------------------------------

# set up list for points of all plots
# vector of plots to use (from 0-15)
cols = np.array([6,8,10,12,14],dtype=int)
num_plots = cols.size

y=np.zeros(shape=(num_plots,102))
x=np.arange(0,1010,10,dtype=float)

# magentization.txt contains coumnts of temp v magnetization with a header
# in row 1 which contains the grain size for each data set.
filename = 'magnetization_norm.txt'
iter = 0
for i in np.nditer(cols):
    y[iter] = np.loadtxt(filename,dtype=float,usecols=(i,))
    iter = iter + 1

# create vector of diameters from row 1
r = y[:,0]

# delete row 1 from x
y = np.delete(y,0,1)

# normalise magnetization values if required
normalise = 0 # 1 True, 0 False
if normalise:
    for i in range(0,y.shape[0]):
        max = y[i][0]
        y[i] = y[i]/max

# remove values from beginning or end of x and y
skiprows_first = 30
skiprows_last  = 10

y = np.delete(y, range(0,skiprows_first), 1)
y = np.delete(y, range(y.shape[1]-skiprows_last,y.shape[1]), 1)
x = np.delete(x, range(0,skiprows_first), 0)
x = np.delete(x, range(x.shape[0]-skiprows_last,x.shape[0]), 0)

# optimize values of beta and nu by using least squares fit
beta      = 0.525
nu        = 1.0
Tc        = 856.67
param   = np.array([beta,nu,Tc])
optimal = least_squares(residual,param,args=(x,y,r),verbose=2,bounds=([0.1,0.5,800.0],[0.9,1.25,870.0]))
#optimal = fmin(residual, param,args=(x,y,r),maxiter=10000, maxfun=10000, disp=1)
print('Optimized parameters:')
print('Beta=',optimal.x[0])
print('Nu=',optimal.x[1])
print('Tc=',optimal.x[2])
# print('Beta= ',optimal[0])
# print('Nu= ',optimal[1])
# print('Tc= ',optimal[2])

# set up parameters for ploting condensed fit
xmin = np.zeros(shape=(8))
