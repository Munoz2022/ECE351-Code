# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 08:44:08 2022

@author: defaultuser0
"""


###############################################################
#                                                             #
#Isaias Munoz                                                 # 
#Ece351                                                       # 
#Lab 10                                                    #
#11/3/22                                                      #
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

# import cmath
#import matplotlib.pyplot.semilogx() as plt

#Task1 plotting the dB mag of H(S)=1/RC*S/s^2+1/rC*s+1/LC


steps = .5
w = np.arange(1e3, 1e6 + steps, steps)

r=1000
l=.027
c=.0000001
#trying to make a function to add all omegas but didnt work out 
# def magfunc(w,r,c,l):
#     y = np.zeros(len(w))
#     for i in range(len(w)):
#         y+=20*np.log10(  (w/r*c)/  np.sqrt( ( 1 / (l * c) - w**2)**2 +   (w/(r*c))**2 )        )
#     return y   
#magn=magfunc(w,r,l,c)  
#just wrote the simple equation to plot 
ymag= w/(r*c)/  np.sqrt( ( 1 / (l * c) - w**2)**2 +   (w/(r*c))**2 )        

ymagnit=20*np.log10(ymag) 

yphase=90-np.arctan( w/(r*c) / (1/(l*c) -(w**2) ) )  * (180/np.pi) 

for i in range(len(yphase)):
    if  (yphase[i] > 90):
        yphase[i] = yphase[i] - 180
    # else:
    #     yphase[i] = yphase[i]*(180/np.pi)

plt . figure ( figsize = (12 , 8) )
plt . semilogx (w, ymagnit )
plt . ylabel ('Y output dB')
plt . xlabel ('w rads/second')
plt . title ('magnitude of transfer function task 1 ')
plt . grid ()


plt . figure ( figsize = (12 , 8) )
plt . semilogx (w, yphase )
plt . ylabel ('Y output degrees')
plt . xlabel ('w rads/second')
plt . title ('phase angle of transfer function task 1')
plt . grid ()




#plotting using scipy.signal.bode


numer=[0,1/(r*c),0]
denom=[1,1/(r*c), 1/(l*c) ]


# ymagnit2=20*np.log10(denom) 

warray, marray,parray=sig.bode((numer,denom),w)


plt . figure ( figsize = (12 , 8) )
plt . semilogx (warray, marray )
plt . ylabel ('Y output dB')
plt . xlabel ('w rads/second')
plt . title ('magnitude of transfer function using .signal.bode task 2')
plt . grid ()


plt . figure ( figsize = (12 , 8) )
plt . semilogx (warray, parray )
plt . ylabel ('Y output degrees')
plt . xlabel ('w rads/second')
plt . title ('phase angle using .signal.bode task 2 ')
plt . grid ()


plt.figure(figsize=(11,7))
sys=con.TransferFunction(numer,denom) 
_ = con.bode(sys, w, dB= True, Hz = True, deg=True, Plot=True)





freq=100000 *np.pi
newsteps = 1/freq
t1 = np.arange(0, 0.01 + newsteps, newsteps)
funcx = np.cos(2*np.pi*100*t1) + np.cos(2*np.pi*3024*t1) + np.sin(2*np.pi*50000*t1)
plt . figure ( figsize = (10 , 6) )
plt . plot (t1, funcx )
plt . ylabel ('Y output')
plt . xlabel ('seconds')
plt . title ('Graphing x(t)')
plt . grid ()
# warray, marray,parray=sig.bode((numer,denom),w)
thing1,thing2=sig.bilinear(numer,denom,freq)
#spits
final1=sig.lfilter(thing1,thing2,funcx)
plt . figure ( figsize = (11 , 7) )
plt . plot (t1, final1)
plt . ylabel ('Y output')
plt . xlabel ('seconds')
plt . title ('Part 2 filtering the given function x(t) ')
plt . grid ()















