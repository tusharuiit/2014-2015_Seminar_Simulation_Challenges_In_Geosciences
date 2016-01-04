# This allows the user to run the acoustic wave equation simulation with
# temporal (in time) accuracy of 2nd, 4th, 6th, 8th ... and so on, order in 2D. The order of
# accuracy in space is 2
# The algorithm uses the Dablain Trick.

import numpy as np
import matplotlib.pyplot as plt
import math
# Show the plots in the Notebook.
#plt.switch_backend("nbagg")

# ---------------------------------------------------------
# Simple finite difference solver
#
# Acoustic wave equation  p_tt = c^2 p_xx + src
# 2-D regular grid
# ---------------------------------------------------------

nx = 200      # grid points in x
nz = 200      # grid points in z
# The simulation is run from 10 to (nt - 11). The range needs to be adjusted accordingly
#
nt = 750      # number of time steps
dx = 10.0     # grid increment in x
dt = 0.001    # Time step
c0 = 3000.0   # velocity (can be an array)
isx = nx / 2  # source index x
isz = nz / 2  # source index z
ist = 100     # shifting of source time function
f0 = 100.0    # dominant frequency of source (Hz)
isnap = 1    # snapshot frequency
T = 1.0 / f0  # dominant period
# nop has to be 3, since spatial accuracy has to be of order 2
nop = 3       # length of operator
# N is the order in time, that you can choose
N = 2
orderInTime = N

# Model type, available are "homogeneous", "fault_zone",
# "surface_low_velocity_zone", "random", "topography",
# "slab"
model_type = "homogeneous"

# Receiver locations
irx = np.array([60, 80, 100, 120, 140])
irz = np.array([5, 5, 5, 5, 5])
seis = np.zeros((len(irx), nt))

# Initialize pressure at different time steps and the second
# derivatives in each direction
p = np.zeros((nz, nx))
pold = np.zeros((nz, nx))
pnew = np.zeros((nz, nx))
pxx = np.zeros((nz, nx))
pzz = np.zeros((nz, nx))
pxxfurther = np.zeros((nz, nx))
pzzfurther = np.zeros((nz, nx))
#For the source term

# Initialize velocity model
c = np.zeros((nz, nx))

if model_type == "homogeneous":
    c += c0
elif model_type == "fault_zone":
    c += c0
    c[:, nx / 2 - 5: nx / 2 + 5] *= 0.8    
elif model_type == "surface_low_velocity_zone":
    c += c0
    c[1:10,:] *= 0.8
elif model_type == "random":
    pert = 0.4
    r = 2.0 * (np.random.rand(nz, nx) - 0.5) * pert
    c += c0 * (1 + r)   
elif model_type == "topography":
    c += c0
    c[0 : 10, 10 : 50] = 0                         
    c[0 : 10, 105 : 115] = 0                       
    c[0 : 30, 145 : 170] = 0
    c[10 : 40, 20 : 40]  = 0
    c[0 : 15, 50 : 105] *= 0.8    
elif model_type == "slab":
    c += c0
    c[110 : 125, 0 : 125] = 1.4 * c0
    for i in range(110, 180):
        c[i , i-5 : i + 15 ] = 1.4 * c0
else:
    raise NotImplementedError
    
cmax = c.max()

# Source time function Gaussian, nt + 1 as we lose the last one by diff
src = np.empty(nt + 1)
for it in xrange(nt):
    src[it] = np.exp(-1.0 / T ** 2 * ((it - ist) * dt) ** 2)
# Take the first derivative
src = np.diff(src) / dt
src[nt - 1] = 0
# This is where we deviated from the original code.
srcoriginal = src
src = src/(dt ** 2)

recordpmax = np.empty(nt)

v = max([np.abs(srcoriginal.min()), np.abs(srcoriginal.max())])
# Initialize animated plot
image = plt.imshow(pnew, interpolation='nearest', animated=True,
                   vmin=-v, vmax=+v, cmap=plt.cm.RdBu)

# Plot the receivers
for x, z in zip(irx, irz):
    plt.text(x, z, '+')

plt.text(isx, isz, 'o')
plt.colorbar()
plt.xlabel('ix')
plt.ylabel('iz')


plt.ion()
plt.show()


# required for seismograms
ir = np.arange(len(irx))

# Output Courant criterion
print "Courant Criterion eps :"
print cmax*dt/dx



#calculate 2nd derivative in time for the source function.
def printTimeDerivativeTerm_For_Source ( srcDerivativeArray1 , dt):

    lengthOfArray1 = len(srcDerivativeArray1)

    if(lengthOfArray1 == 1):
        return srcDerivativeArray1, None
    else:
        srcDerivativeArray2 = np.empty(lengthOfArray1 - 2)
        lengthOfArray2 = len(srcDerivativeArray2)

        for i in xrange(0, lengthOfArray2):
            indexOfDecidingArray = i + 1
            srcDerivativeArray2[i] = (srcDerivativeArray1[indexOfDecidingArray+1] + srcDerivativeArray1[indexOfDecidingArray-1] - 2 * srcDerivativeArray1[indexOfDecidingArray] )/(dt ** 2)
        return srcDerivativeArray1, srcDerivativeArray2


#calculate partial derivatives for 2nd order accuracy in space, be careful around the boundaries
def printSecondDerivative_ZZ_OfMatrix_3point(p, pzz):
    for i in xrange(1, nx - 1):
        pzz[:, i] = p[:, i + 1] - 2 * p[:, i] + p[:, i - 1]
    return pzz

def printSecondDerivative_XX_OfMatrix_3point(p, pxx):
    for j in xrange(1, nz - 1):
        pxx[j, :] = p[j - 1, :] - 2 * p[j, :] + p[j + 1, :]
    return pxx



# Time extrapolation
for it in range(10 , nt - 11):

    pxxzzcollector = np.zeros((nz, nx))
    srcTimeDerivativeCollector = 0

    numberOfPointsOnLeftAndRight = N/2 - 1
    srcTimeDerivative = src[(it - numberOfPointsOnLeftAndRight) : (it + numberOfPointsOnLeftAndRight + 1)]

    if nop==3:
        pzz = p
        pxx = p
        
        for orderLoop in xrange(N/2):
            
#calculate partial derivatives in SPACE, be careful around the boundaries
            pzzfurther = printSecondDerivative_ZZ_OfMatrix_3point(pzz, pzzfurther)
            pxxfurther = printSecondDerivative_XX_OfMatrix_3point(pxx, pxxfurther)
            pzz = pzzfurther
            pxx = pxxfurther
            pzz /= dx ** 2
            pxx /= dx ** 2
            pzz = c**2 * pzz
            pxx = c**2 * pxx

            term1 = dt ** (2 * (orderLoop) + 2)
            term1 = term1 / (math.factorial(2 * (orderLoop + 1)))
            term1 = term1 * 2

            pxxzzcollector = pxxzzcollector + (term1 * (pzz + pxx))

#The time part in TIME
            srcTimeDerivative, srcDerivativeArray2 = printTimeDerivativeTerm_For_Source ( srcTimeDerivative , dt)
            indexToBeExtracted = N/2 - 1 - orderLoop
            term2 = 1.0 / (math.factorial(2 * (orderLoop + 1)))
            term2 = term2 * 2
            term2 = term2 * dt ** (2 * (orderLoop) + 2)
            srcTimeDerivativeCollector = srcTimeDerivativeCollector + term2 * srcTimeDerivative[indexToBeExtracted]
            srcTimeDerivative = srcDerivativeArray2
                    
            
# Time extrapolation
    pnew = 2 * p - pold + pxxzzcollector

# Add source term at isx, isz
    pnew[isz, isx] = pnew[isz, isx] + srcTimeDerivativeCollector

# Plot every isnap-th iteration
    if it % isnap == 0:    # you can change the speed of the plot by increasing the plotting interval
        plt.title("Max P: %.10f" % p.max())
        recordpmax[it] = p.max()
        image.set_data(pnew)
        plt.draw()
#        plt.savefig("Enter the name of the file to save your plots.")

    pold, p = p, pnew

    # Save seismograms
    seis[ir, it] = p[irz[ir], irx[ir]]

#np.savetxt("Enter the name of the location here to save the pmax values", recordpmax)
#
# Plot the source time function and the seismograms
#

plt.ioff()
plt.figure(figsize=(12, 12))

plt.subplot(221)
time = np.arange(nt) * dt
plt.plot(time, srcoriginal)
plt.title('Source time function')
plt.xlabel('Time (s) ')
plt.ylabel('Source amplitude ')

plt.subplot(222)
ymax = seis.ravel().max()
for ir in range(len(seis)):
    plt.plot(time, seis[ir, :] + ymax * ir)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

plt.subplot(223)
ymax = seis.ravel().max()
for ir in range(len(seis)):
    plt.plot(time, seis[ir, :] + ymax * ir)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

plt.subplot(224)
# The velocity model is influenced by the Earth model above
plt.title('Velocity Model')
plt.imshow(c)
plt.xlabel('ix')
plt.ylabel('iz')
plt.colorbar()
#plt.savefig("Enter the name of the location here to save the seismograms")

plt.show()
