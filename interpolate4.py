from scipy.interpolate import interp1d
from scipy.optimize import least_squares
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
    t    = (x - Tc)/Tc
    t_s  = np.zeros(shape=(101,))
    summ  = np.zeros(shape=(101,))
    N    = 0
    r    = r.astype(int)

    # for each data set p
    for p in range(0,y.shape[0],1):
        # create an interpolated function of data set p
        curve = interp1d(t,y[p],kind='cubic')

        # for each data set j from p+1 to max_sets
        for j in range(p+1,y.shape[0],1):

            # loop over all temperature points i
            for i in range(0,101,1):
                # find ti in data set j, rescaled to data set p
                t_s[i] = ((r[j]/r[p])**(1/nu))*t[i]

                # temperature points of rescaled t_p must lie in range t_j
                if t[0] <= t_s[i] <= t[100]:

                    term1 = (r[j]**(beta/nu))*y[j][i]
                    term2 = (r[p]**(beta/nu))*curve(t_s[i])
                    summ[i] += abs(term1-term2)
                    N = N + 1

    return summ/N
#-------------------------------------------------------------------------------

# set up list for points of all plots
# vector of plots to use (from 0-15)
cols = np.array([10,14],dtype=int)
num_plots = cols.size

y=np.zeros(shape=(num_plots,102))
x=np.arange(0,1010,10,dtype=float)

# magentization.txt contains coumnts of temp v magnetization with a header
# in row 1 which contains the grain size for each data set.
filename = 'magnetization.txt'
iter = 0
for i in np.nditer(cols):
    y[iter] = np.loadtxt(filename,dtype=float,usecols=(i,))
    iter = iter + 1

# create vector of diameters from row 1
r = y[:,0]

# delete row 1 from x
y = np.delete(y,0,1)

# optimize values of beta and nu by using least squares fit
beta      = 0.525
nu        = 1.0
Tc        = 856.67
param   = np.array([beta,nu,Tc])
optimal = least_squares(residual,param,args=(x,y,r),verbose=2,bounds=([0.0,0.0,845.0],[np.inf,1.0,865.0])) #
print('Optimized parameters:')
print('Beta=',optimal.x[0])
print('Nu=',optimal.x[1])
print('Tc=',optimal.x[2])

# set up parameters for ploting condensed fit
xmin = np.zeros(shape=(8))
