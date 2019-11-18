###############################################################################
#
# This implementation is using Python 2.7
#
#
###############################################################################

from numpy import *
import pylab as py
import scipy.interpolate as scii
import scipy.optimize as scio

##########################  class Fitpar ######################################
class Fitpar:
   """
   - FSS for M(T)-data:

     M(t, L)\sim L^{-\beta/\nu}\tilde{\cal M}_\pm(|t|L^{1/\nu}).
   """

   def __init__(self, par_array):
      self.length_par_array = len(par_array)
      self.update_fitparam(par_array)

   def update_fitparam(self, par_array):
      self.par_array = par_array
      self.Tc = par_array[0]
      self.nu = par_array[1]
      self.beta = par_array[2]

   def set_fitparam_error(self, par_errpl, par_errmi):
      self.Tc_errpl = par_errpl[0]
      self.Tc_errmi = par_errmi[0]
      self.nu_errpl = par_errpl[1]
      self.nu_errmi = par_errmi[1]
      self.beta_errpl = par_errpl[2]
      self.beta_errmi = par_errmi[2]

   def set_pb_optim(self, pb_optim):
      self.pb_optim = pb_optim

   def scaling_function(self, b):
      Tc = self.Tc
      nu = self.nu
      beta = self.beta

      t = (b.T-Tc)/b.T
      x = (b.size)**(1.0/nu)*(t)
      y = (b.size)**(beta/nu)*b.M

      return [y, x]

   def scaling_function_range(self, b, x_min, x_max):
      [y_prime, x_prime] = self.scaling_function(b)
      x = x_prime[(x_prime>=x_min)&(x_prime<=x_max)]
      y = y_prime[(x_prime>=x_min)&(x_prime<=x_max)]
      return [y, x]

   def print_values(self):
      print 'Tc: ', self.Tc, \
            '+/-', '(', self.Tc_errpl, '/', self.Tc_errmi, ')'
      print 'nu: ', self.nu, \
            '+/-', '(', self.nu_errpl, '/', self.nu_errmi, ')'
      print 'beta: ', self.beta, \
            '+/-', '(', self.beta_errpl, '/', self.beta_errmi, ')'
      print 'pb_optim: ', self.pb_optim

######################  end class Fitpar ######################################

##########################  class Bundle ######################################
class Bundle:
   """
   To handle the input data format for the FSS.
   """
   def __init__(self, filename, size_indx, size):
      """
      filename - this is the file name
      size - size_array[size_indx]
      """
      self.filename = filename
      indx = size_indx
      self.size = size

   def read_data(self):
      """
      Read in the M vs. T data, such that the lowest index is the largest value.
      """

      #skiprows_first = 30
      #skiprows_last = 20

      skiprows_first = 30
      skiprows_last = 1

      self.mat = loadtxt(filename, skiprows=skiprows_first)
      ln = len(self.mat[:,0])-skiprows_last
      self.T = zeros((ln))
      self.M = zeros((ln))
      self.T[0:ln] = self.mat[0:-skiprows_last,0]
      self.M[0:ln] = self.mat[0:-skiprows_last,1]

      # py.figure(size_indx)
      # py.plot(self.T, self.M, 'o')
      # py.xlabel('$T$ [K]', fontsize = 20)
      # py.ylabel('$M/M_S$', fontsize = 20)
      # py.show()

###############################################################################
##############              Functions            ##############################
###############################################################################

def do_interpolates(fitpars, size_array): # takes object of the class Fitpar
   """
   Returns size_array-long vector of objects of a class interp1d from scipy.interpolate, which based on the reference values x,y corresponding to simulated (scaled) M vs. T creates an approximate function f, which allows to find interpolated values y_new = f(x_new). The class contains a method __call__ to do the job.
   """

   inter_dat = []
   for size_indx in range(len(size_array)):
      b = bundles[size_indx]
      [y, x] = fitpars.scaling_function(b)
      inter_dat.append(scii.interp1d(x,y))
   return inter_dat

def xinterp_range(b, bI, fitpars):
   """
   Returns the x values (i.e. scaled T) common to b and bI extracted from bI.
   """

   [y, x] = fitpars.scaling_function(b)
   [yI, xI] = fitpars.scaling_function(bI)
   x_common_min = max(min(x), min(xI))
   x_common_max = min(max(x), max(xI))
   boolean = (x>=x_common_min) & (x<=x_common_max)
   x_common = x[(x>=x_common_min) & (x<=x_common_max)]
   y_common = y[(x>=x_common_min) & (x<=x_common_max)]
   return [y_common, x_common]

def xinterp_range_common_to_all(fitpars, size_array):
   """
   Returns x-range (i.e. scaled T) common to all available b
   """
   x_min, x_max = -1e10, 1e10
   for size_indx in range(len(size_array)):
      b = bundles[size_indx]
      [y, x] = fitpars.scaling_function(b)
      if x_min < min(x): x_min = min(x) # looking for the largest minimum
      if x_max > max(x): x_max = max(x) # looking for the smallest maximum
   return [x_min, x_max]

def error_func(inter_dat, fitpars, size_array):
   """
   Evaluate the objective function based on the actual values of critical exponetns. This can be done using 'full_range' mode using "xinterp_range" or 'common_range' using "xinterp_range_common_to_all".
   """

#   fit_mode = 'common_range'
   fit_mode = 'full_range'
   #
   if fit_mode == 'full_range':
      pb2 = 0.0 #sum of residuals
      for size_indx1 in range(len(size_array)): # no-interpolated data index
         b1 = bundles[size_indx1]
         for size_indx2 in range(len(size_array)): # interpolated data index
            if (size_indx1 != size_indx2): # exclude b1 = b2
               b2 = bundles[size_indx2]
               yx = xinterp_range(b1,b2,fitpars)
               if len(yx[1]) > 1:
                   pb2 = pb2 + sum((yx[0] - \
                         inter_dat[size_indx2](yx[1]))**2) # Eq. 2
               else:
                   pb2 = -100.0 # set unrealistic large number
      return pb2

   elif fit_mode == 'common_range':
      [x_min, x_max] = xinterp_range_common_to_all(fitpars, size_array)
      pb2 = 0.0 #sum of residuals
      for size_indx1 in range(len(size_array)): # no-interpolated data index
         b1 = bundles[size_indx1]
         [y1, x1] = fitpars.scaling_function_range(b1, x_min, x_max)
         for size_indx2 in range(len(size_array)): # interpolated data index
            if (size_indx1 != size_indx2): # exclude b1 = b2
               pb2 = pb2 + sum((y1 - inter_dat[size_indx2](x1))**2) # Eq. 2
      return pb2

def min_func(par_array, fitpars, size_array):
   """
   - this is the actual fit function, constructed based on the actual parameter array and "do_interpolates" and "error_func"
   - fitpars is an instance of the class Fitpar
   """

   fitpars.update_fitparam(par_array)
   inter_dat = do_interpolates(fitpars, size_array)
   pb = error_func(inter_dat, fitpars, size_array)
   return pb

def fit_data(fitpars, size_array):
   """
   fit!
   """

   p0 = fitpars.par_array
   p1 = scio.fmin(min_func, p0,(fitpars, size_array),
                  maxiter=10000, maxfun=10000, disp=0)
   fitpars.update_fitparam(p1)
   return p1

def get_errorbars(fitpars_optim, eta, size_array):
   """
   Errorbars on fit parameters
   - pb is the value of the objective function
   """

   # optimum parameters
   inter_dat_optim = do_interpolates(fitpars_optim, size_array)
   p_optim = fitpars_optim.par_array
   pb_optim = error_func(inter_dat_optim, fitpars_optim, size_array)

   # calculate error
   sigma_plus, sigma_minus = [], []
   p = [k for k in p_optim]

   fitpars_prime = Fitpar(p)
   for i in range(len(p)):
      p[i] = p[i]+eta*p[i]
      fitpars_prime.update_fitparam(p)
      interp_data = do_interpolates(fitpars_prime, size_array)
      pb = error_func(interp_data, fitpars_prime, size_array)
      p = [k for k in p_optim]
      sigma_plus.append(eta*p_optim[i]/sqrt(abs(2.0*log(pb/pb_optim))))

      p[i] = p[i]-eta*p[i]
      fitpars_prime.update_fitparam(p)
      interp_data = do_interpolates(fitpars_prime, size_array)
      pb = error_func(interp_data, fitpars_prime, size_array)
      p = [k for k in p_optim]
      sigma_minus.append(eta*p_optim[i]/sqrt(abs(2.0*log(pb/pb_optim))))

   fitpars_optim.set_fitparam_error(sigma_plus, sigma_minus)
   fitpars_optim.set_pb_optim(pb_optim)
   fitpars_optim.print_values()

################################################################################
#########                       plot stuff                 #####################
################################################################################

def plot_collapse(fitpars, size_array):
   """
   Plots collapses based on the best critical exponents
   """
   for size_indx in range(len(size_array)):
      b = bundles[size_indx]
      [y, x] = fitpars.scaling_function(b)
      #for i in range(len(x)):
      #   print size_indx, '\t', x[i], '\t', y[i]
      py.plot(x,y,'o')
      #py.hold(True)

   py.xlabel('$L^{1/\\nu}(T-T_C)/T_C$', fontsize = 20)
   py.ylabel('$L^{\\beta/\\nu}(M/M_S)$', fontsize = 20)
   #py.title('Short-range FePt Hamiltonian')
   #py.axis([-18, 11, -0.1, 1.7])


################################################################################
################################################################################
#########              Setup for fitting the actual data          ##############
################################################################################
################################################################################

# avalable values
size_array = array([2, 3, 4, 5, 6, 7, 8, 9])

# read the input data for scaling
#py.figure(1)

bundles = []
size_indx = 0
for size in size_array:
   filename = './'+str(size)+'nm'
   bundles.append(Bundle(filename, size_indx, size))
   bundles[size_indx].read_data()
   size_indx += 1

# Initialize
#Tc0_vec = [600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0]
#nu0_vec = [0.60, 0.80, 0.90]
#beta0_vec = [0.1, 0.15, 0.2, 0.25, 0.3]

Tc0_vec = [850.0]
nu0_vec = [1.0]
beta0_vec = [0.55]

for i1 in Tc0_vec:
    for i2 in nu0_vec:
        for i3 in beta0_vec:
            Tc0, nu0, beta0 = i1, i2, i3
            print Tc0, nu0, beta0
            p0 = []
            p0.append(Tc0)
            p0.append(nu0)
            p0.append(beta0)

            fitpars = Fitpar(p0)

            # fit!
            p1 = fit_data(fitpars, size_array)
            fitpars.update_fitparam(p1)
            get_errorbars(fitpars, 0.01, size_array)

            print
            print

py.figure(2)
plot_collapse(fitpars, size_array)

#py.figure(3)
#size_array = [2, 3, 4, 5, 6, 7, 8, 9]
#tceff1 = [p1[0]*(1.0-size_array[i]**(-1.0/p1[1])) for i in range(len(size_array))]
#tceff2 = [p1[0]*(1.0+size_array[i]**(-1.0/p1[1]))**(-1.0) for i in range(len(size_array))]

#tceff3 = [672.0, 692.0, 701.0, 708.0, 714.0, 718.0, 720.0, 722.0]


#for i in range(len(tceff)): print size_array[i], '\t', tceff[i]
#py.plot(size_array, tceff1,'ko', size_array, tceff2, 'ro', size_array, tceff3, 'bo')
#py.plot(size_array, tceff2, 'ro', size_array, tceff3, 'bo')
#py.axis([1, 10, 200, 800])
#py.xlabel('size [nm]', fontsize = 20)
#py.ylabel('$T_C^{eff}$ [K]', fontsize = 20)

py.show()
