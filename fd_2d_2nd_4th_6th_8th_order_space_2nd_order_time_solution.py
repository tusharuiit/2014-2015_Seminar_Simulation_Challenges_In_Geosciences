# ---------------------------------------------------------
# Simple finite difference solver
#
# Acoustic wave equation  p_tt = c^2 p_xx + src
# 2-D regular grid
# ---------------------------------------------------------

# This allows the user to run the acoustic wave equation simulation with
# spatial accuracy of 2nd, 4th, 6th, and 8th order in 2D. The order of
# accuracy in time is 2

import numpy as np
import matplotlib.pyplot as plt

nx = 200      # grid points in x
nz = 200      # grid points in z
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
# By modifying nop, you can change the accuracy
# nop = 3,5,7,9 are for 2nd, 4th, 6th and 8th order accuracy
nop = 3       # length of operator

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

# Source time function Gaussian, nt + 1 as we loose the last one by diff
src = np.empty(nt + 1)
for it in xrange(nt):
    src[it] = np.exp(-1.0 / T ** 2 * ((it - ist) * dt) ** 2)
# Take the first derivative
src = np.diff(src) / dt
src[nt - 1] = 0

recordpmax = np.empty(nt)


v = max([np.abs(src.min()), np.abs(src.max())])
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


# Time extrapolation
for it in range(10 , nt - 11):
    if nop==3:
        # calculate partial derivatives, be careful around the boundaries
        for i in xrange(1, nx - 1):
            pzz[:, i] = p[:, i + 1] - 2 * p[:, i] + p[:, i - 1]
        for j in xrange(1, nz - 1):
            pxx[j, :] = p[j - 1, :] - 2 * p[j, :] + p[j + 1, :]

    if nop==5:
        # calculate partial derivatives, be careful around the boundaries
        for i in xrange(2, nx - 2):
            pzz[:, i] = -1./12*p[:,i+2]+4./3*p[:,i+1]-5./2*p[:,i]+4./3*p[:,i-1]-1./12*p[:,i-2]
        for j in xrange(2, nz - 2):
            pxx[j, :] = -1./12*p[j+2,:]+4./3*p[j+1,:]-5./2*p[j,:]+4./3*p[j-1,:]-1./12*p[j-2,:]
                    
    if nop==7:
        # calculate partial derivatives, be careful around the boundaries
        for i in xrange(3, nx - 3):
            pzz[:, i] = 1./90*p[:,i+3]-3./20*p[:,i+2]+3./2*p[:,i+1]-49./18*p[:,i]+3./2*p[:,i-1]-3./20*p[:,i-2]+1./90*p[:,i-3]
        for j in xrange(3, nz - 3):
            pxx[j, :] = 1./90*p[j+3,:]-3./20*p[j+2,:]+3./2*p[j+1,:]-49./18*p[j,:]+3./2*p[j-1,:]-3./20*p[j-2,:]+1./90*p[j-3,:]
    if nop==9:
        # calculate partial derivatives, be careful around the boundaries
        for i in xrange(4, nx - 4):
            pzz[:, i] = -1./560*p[:,i+4]+8./315*p[:,i+3]-1./5*p[:,i+2]+8./5*p[:,i+1]-205./72*p[:,i]+8./5*p[:,i-1]-1./5*p[:,i-2]+8./315*p[:,i-3]-1./560*p[:,i-4]
        for j in xrange(4, nz - 4):
            pxx[j, :] = -1./560*p[j+4,:]+8./315*p[j+3,:]-1./5*p[j+2,:]+8./5*p[j+1,:]-205./72*p[j,:]+8./5*p[j-1,:]-1./5*p[j-2,:]+8./315*p[j-3,:]-1./560*p[j-4,:]
            
    pxx /= dx ** 2
    pzz /= dx ** 2

    # Time extrapolation
    pnew = 2 * p - pold + dt ** 2 * c ** 2 * (pxx + pzz)
    # Add source term at isx, isz
    pnew[isz, isx] = pnew[isz, isx] + src[it]

    # Plot every isnap-th iteration
    if it % isnap == 0:    # you can change the speed of the plot by increasing the plotting interval
        
        plt.title("Max P: %.10f" % p.max())
        recordpmax[it] = p.max()
        image.set_data(pnew)
        plt.draw()
#        plt.savefig("Enter the name of the location here to save the plots")
    pold, p = p, pnew

    # Save seismograms
    seis[ir, it] = p[irz[ir], irx[ir]]
    
#np.savetxt("#Enter the name of the location to save the pmax values")
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
plt.imshow(c)
plt.xlabel('ix')
plt.ylabel('iz')
plt.colorbar()

#plt.savefig("Enter the name of the folder to save the seismograms here")
plt.show()

