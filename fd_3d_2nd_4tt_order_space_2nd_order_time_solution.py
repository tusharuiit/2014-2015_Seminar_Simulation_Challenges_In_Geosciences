# ---------------------------------------------------------
# Simple finite difference solver
#
# Acoustic wave equation  p_tt = c^2 p_xx + src
# 3-D regular grid
# ---------------------------------------------------------
# This is a configuration step for the exercise. Please run it before the simulation code!

# This ipython notebook allows the user to run the acoustic wave equation simulation with
# spatial accuracy of 2nd, 4th, 6th, and 8th order in 3D.
# In a set of 100*100*100 grid, it visualizes, [:,50,:] plane

import numpy as np
import matplotlib.pyplot as plt

nx = 100      # grid points in x
nz = 100      # grid points in z
ny = 100      # grid points in y

nt = 7500      # number of time steps
dx = 5.0     # grid increment in x
dt = 0.0001    # Time step
c0 = 3000.0   # velocity (can be an array)
isx = nx / 2  # source index x
isz = nz / 2  # source index z
isy = ny / 2  # source index y
ist = 1000     # shifting of source time function
f0 = 100.0    # dominant frequency of source (Hz)
isnap = 10    # snapshot frequency
T = 1.0 / f0  # dominant period
# By modifying nop, you can change the accuracy
# nop = 3,5, are for 2nd, 4th order accuracy
nop = 3       # length of operator

# Model type, available are "homogeneous", "fault_zone",
# "surface_low_velocity_zone", "random", "topography",
# "slab"
model_type = "homogeneous"

# Receiver locations
irx = np.array([15, 20, 25, 30, 35])
irz = np.array([3, 3, 3, 3, 3])
iry = np.array([3, 3, 3, 3, 3])

seis = np.zeros((len(irx), nt))

# Initialize pressure at different time steps and the second
# derivatives in each direction
p = np.zeros((nz, nx, ny))
pold = np.zeros((nz, nx, ny))
pnew = np.zeros((nz, nx, ny))
pxx = np.zeros((nz, nx, ny))
pzz = np.zeros((nz, nx, ny))
pyy = np.zeros((nz, nx, ny))

# Initialize velocity model
c = np.zeros((nz, nx, ny))

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
    c[27 : 32, 0 : 32, 0 : 32] = 1.4 * c0
    for i in range(27, 45):
        c[i , i-1 : i + 4 , i-1 : i + 4] = 1.4 * c0
else:
    raise NotImplementedError
    
cmax = c.max()

# Source time function Gaussian, nt + 1 as we loose the last one by diff
src = np.empty(nt + 1)
for it in xrange(nt):
    src[it] = np.exp(-1.0 / T ** 2 * ((it - ist) * dt) ** 2)
# Take the first derivative
src = np.diff(src) / dt
src[nt - 1] = 0
recordpmax = np.empty(nt + 1)


v = max([np.abs(src.min()), np.abs(src.max())])
# Initialize animated plot
image = plt.imshow(pnew[:,50,:], interpolation='nearest', animated=True,
                   vmin=-(v/2), vmax=+(v/2), cmap=plt.cm.RdBu)

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


# Time extrapolation
for it in range(100, nt - 110):
    if nop==3:
        # calculate partial derivatives, be careful around the boundaries
        for i in xrange(1, nx - 1):
            pzz[:, :, i] = p[:, :, i + 1] - 2 * p[:, :, i] + p[:, :, i - 1]
        for j in xrange(1, nz - 1):
            pxx[:, j, :] = p[:, j + 1, :] - 2 * p[:, j, :] + p[:, j - 1, :]
        for k in xrange(1, ny - 1):
            pyy[k, :, :] = p[k - 1, :, :] - 2 * p[k, :, :] + p[k + 1, :, :]

    if nop==5:
        # calculate partial derivatives, be careful around the boundaries
        for i in xrange(2, nx - 2):
            pzz[:, :, i] = -1./12*p[:,:,i+2]+4./3*p[:,:,i+1]-5./2*p[:,:,i]+4./3*p[:,:,i-1]-1./12*p[:,:,i-2]
        for j in xrange(2, nz - 2):
            pxx[:, j, :] = -1./12*p[:,j+2,:]+4./3*p[:,j+1,:]-5./2*p[:,j,:]+4./3*p[:,j-1,:]-1./12*p[:,j-2,:]
        for k in xrange(2, ny - 2):
            pyy[k, :, :] = -1./12*p[k+2,:,:]+4./3*p[k+1,:,:]-5./2*p[k,:,:]+4./3*p[k-1,:,:]-1./12*p[k-2,:,:]
                    
    pxx /= dx ** 2
    pzz /= dx ** 2
    pyy /= dx ** 2

    # Time extrapolation
    pnew = 2 * p - pold + dt ** 2 * c ** 2 * (pxx + pzz + pyy)
    # Add source term at isx, isz
    pnew[isz, isx, isy] = pnew[isz, isx, isy] + src[it]
    
    # Plot every isnap-th iteration
    if it % isnap == 0:    # you can change the speed of the plot by increasing the plotting interval
        recordpmax[it] = pnew.max()
        plt.title("Max P: %.10f" % p.max())
        image.set_data(pnew[:, 50, :])
        plt.draw()
#        plt.savefig("Enter the name of the file to save your plots.")

    pold, p = p, pnew

# Save seismograms Hello World
    seis[ir, it] = p[irz[ir], irx[ir], iry[ir]]

#np.savetxt("Enter the name of the location here to save the pmax values", recordpmax)
# Plot the source time function and the seismograms
#

plt.ioff()
plt.figure(figsize=(12, 12))

plt.subplot(221)
time = np.arange(nt) * dt
plt.plot(time, src)
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
plt.imshow(c[:, 25, :])
plt.xlabel('ix')
plt.ylabel('iz')
plt.colorbar()
#plt.savefig("Enter the name of the folder to save the seismograms here")
plt.show()

