# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 21:06:57 2022

@author: defaultuser0
"""


###############################################################
#                                                             #
#Isaias Munoz                                                 # 
#Ece351                                                       # 
#Lab 11                                                       #
#11/9/22                                                      #
#                                                             #
###############################################################   
import math
import numpy
import scipy.signal
import time
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.fftpack
import control as con

from matplotlib import patches

#scipy.signal.residue() can also be called
#rout gives u the roots so ur A and B and C term after the partial expansion
#pout gives us the poles where they cross in this case (0) and (s+4) and (s+6)
#kout dont know what it gives u 
a=[2,-40]
b=[1,-10,16]


rout, pout,kout= sig.residue(a,b)


print (' this is the Root solutions to the long DEQ')
print(rout)
print (' this is the Poles solutions to the long DEQ')
print(pout)
# for i in range (len(apout)):
#     print( apout[i])
print (' this is the K (residue of a given term) solutions to the long DEQ')
print(kout)





steps = 0.1
# T = 8 #whatver value

# t = np.arange(0, 2 , T)
# T = 8 #whatver value
# w = (2*np.pi)/T
# fs=100
# T=1/fs
t = np.arange(0, 2+steps , steps)













def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import patches
    # get a figure/plot
    ax = plt.subplot(111)
    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
    color='black', ls='dashed')
    ax.add_patch(uc)
    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1
    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    # Plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'o', ms=10,label='Zeros')
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0)
    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'x', ms=10,label='Poles')
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend()
    # set the ticks
    # r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    return z, p, k
    
az=[2,-40,0]
bz=[1,-10,16]
Z,P,K=zplane(az, bz)
print (' this is the Root solutions to the long DEQ in z plane')
print(Z)
print (' this is the Poles solutions to the long DEQ in z plane')
print(P)
# for i in range (len(apout)):
#     print( apout[i])
print (' this is the K (residue of a given term) solutions to the long DEQ in z plane')
print(K)


# Use scipy.signal.freqz() to plot the magnitude and phase responses of H(z). Note:
# You must set whole = True within the scipy.signal.freqz() command. (See function
# documentation for details).
wfreq, mag = sig.freqz(az, bz, whole = True)

#wfreq= radians
wfreqnew=wfreq/np.pi
magnitude=20*np.log10(abs(mag))
plt . figure ( figsize = (6 , 2) )
plt . plot (wfreqnew,  magnitude )
plt . ylabel ('Y output dB')
plt . xlabel ('Hz')
plt . title ('magnitude of transfer function task 4 ')
plt . grid ()


angle=np.angle(mag)*180/(np.pi)
# magnitude=20*np.log10(abs(angle))
plt . figure ( figsize = (6 , 2) )
plt . plot (wfreqnew, angle )
plt . ylabel ('phase otput')
plt . xlabel ('Hz')
plt . title ('phase of transfer function task 5 ')
plt . grid ()














