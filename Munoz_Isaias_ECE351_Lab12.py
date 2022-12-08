# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:30:36 2022

@author: defaultuser0
"""



###############################################################
#                                                             #
#Isaias Munoz                                                 # 
#Ece351                                                       # 
#Lab 12 Final Project                                                      #
#11/17/22                                                      #
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


# the other packages you import will go here
import pandas as pd




# # Identify the noise magnitudes and corresponding frequencies due to the low frequency vibration and 
# switching amplifier. This information will help characterize the main noise sources.
# # Also, identify the magnitudes and corresponding frequencies of the position measurement
# # information. To begin, see the example code listing in the Appendix.
# load input signal
df = pd.read_csv('NoisySignal.csv')
t = df['0']. values
sensor_sig = df['1']. values
#this plots the signal coming in only plot 1 noisy
plt.figure(figsize = (10, 7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noisy Input Signal plot 1')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()
# your code starts here , good luck
#need to convert this signal first i think
fs = 1/(t[1]-t[0]) #or the 100khz this just calculates it for me
#because it encapsulates all possibilities

def myfft(x,fs):
    N = len(x) # find the length of the signal
    X_fft = scipy.fftpack.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) # shift zero frequency components
# to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
# signal , (fs is the sampling frequency and
# needs to be defined previously in your code

    X_mag = np.abs(X_fft_shifted)/N  # compute the magnitudes of the signal
    X_phi=np.angle(X_fft_shifted)
    for i in range (len(X_mag)):
        if np.abs(X_mag[i])<1e-10:

            X_phi[i]=0
        # else: 
            # X_phi = np.angle(X_fft_shifted) # compute the phases of the signal
# ----- End of user defined function ----- #
    return freq,X_mag,X_phi #Need to assign these to a value 
freq1,X_mag1,X_phi1=myfft(sensor_sig,fs)#labeling the returns so i can acces them when i go to plot them
#passing noisy signal to do a fft


def make_stem(ax, x, y, color='k', style='solid', label='', linewidths=2.5, **kwargs):
    ax.axhline(x[0], x[-1], 0, color='r')
    ax.vlines(x, 0, y, color=color, linestyles=style, label=label, linewidths=linewidths)
    ax.set_ylim([1.05*y.min(), 1.05*y.max()])
# FFT plots of unfiltered signal
# fig, (ax,ax1,ax2,ax3,ax4) = plt.subplots(figsize=(10,7))
# this plots the unfiler fft of the signal showing us the frequencies at different values only 
##
#
#plot 2a
# make_stem(ax, freq1, X_mag1)
# plt.stem(freq1,X_mag1)
# plt.grid()
# plt.ylabel('|X(f)|')
# plt.xlabel('Frequency [Hz]')
# plt.title('Noisy Input Signal going through the fft showing all frequencies plt 2a')
# plt.show()
#Hardest part figuring how to 
#make these graphsss!!!!!!!!!
#this shows the entire band unfiltered mag and phase
#-400k to 400k rad/s
#using
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq1 , X_mag1 )
# make_stem ( ax ,freq1 , X_phi1 )
plt . title ('Showing the all the band magnitude')
plt.grid()
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('magnittude')
plt . show ()
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq1 , X_phi1 )
plt . title ('Showing the all the band phase')
plt.grid()
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('phase')
plt . show ()



#repeating the above to show specific intervals
#showing 0 to 1800 magnitude and phase
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq1 , X_mag1 )
plt . title ('Showing the  band magnitude 0-1600')
plt.grid()
plt.xlim(0, 1600)
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('magnittude')
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq1 , X_phi1 )
plt . title ('Showing the  the band phase0-1600')
plt.grid()
plt.xlim(0, 1600)
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('phase')
plt . show ()


#repeating the above to show specific intervals
#showing 1800 to 2000 magnitude and phase
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq1 , X_mag1 )
plt . title ('Showing the  band magnitude 1600-2200')
plt.grid()
plt.xlim(1600, 2200)
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('magnittude')
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq1 , X_phi1 )
plt . title ('Showing the the band phase1600-2200')
plt.grid()
plt.xlim(1600, 2200)
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('phase')
plt . show ()



#
#repeating the above to show specific intervals
#showing 2000 to 100000 magnitude and phase
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq1 , X_mag1 )
plt . title ('Showing the  band magnitude 2200-100000')
plt.grid()
plt.xlim(2200, 100000)
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('magnittude')
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq1 , X_phi1 )
plt . title ('Showing the all the band phase 2200-100000')
plt.grid()
plt.xlim(2200, 100000)
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('phase')
plt . show ()





#

# plt.stem(freq1,X_phi1)
# plt.grid()
# plt.ylabel('|X(f)|')
# plt.xlabel('Frequency [Hz]')
# plt.title('Noisy Input Signal going through the fft showing all frequencies plt 3a')
# plt.show()



# make_stem(ax1, freq1, X_mag1)
# plt.stem(freq1,X_phi1)
# plt.grid()
# plt.ylabel('|X(f)|')
# plt.xlabel('Frequency [Hz]')
# plt.title('Noisy Input Signal going through the fft showingspecific  frequencies plt 2b 0 to ')
# plt.xlim(0, 20000)
# plt.show()



#now implementing the values i obtain n paper

#Using series cause it was easier for my case

wc1=1600*2*np.pi
wc2=2200*2*np.pi
#if i choose R=100 cause 100 seems cool
R=100
#B=bandwitch the difference between the corner freq
B=wc2-wc1
wo=np.sqrt(wc1*wc2)
L=R/B
C=1/((wo**2)*L)
#calculating by hand is this values but im sure
#iall need to shift them to adjust bode plot
# C = 88.64 * 1e-6
# L = 79.4 * 1e-3
# R = 100
print('These are my values for RLC series circuit')
print("R:", R)
print("L:", L)
print("C:", C)
num = [(R/L), 0]
den = [1, (R/L), (1/(L*C))]
#plotting the bode as is
step_size = 100
w = np.arange(1, (1e6 )+step_size, step_size)
#rreferencing lab 10 in plotting bode
#did not meet specifications I crashed the plane
#all i need is play with my rlc to filter thats all
plt.figure(figsize=(11,7))
sys=con.TransferFunction(num,den) 
_ = con.bode(sys, w, dB= True, Hz = True, deg=True, Plot=True)
plt.title('showing the whole bode filtered ')
#passing the functions
thing1,thing2=sig.bilinear(num,den,fs)
final1=sig.lfilter(thing1,thing2,sensor_sig)
plt . figure ( figsize = (11 , 7) )
plt . plot (t, final1)
plt . ylabel ('Y output')
plt . xlabel ('seconds')
plt . title (' filtering the given function x(t) ')
plt . grid ()
#lab 10 code used to filter this noisy signal
# freq=100000 *np.pi
# newsteps = 1/freq
# t1 = np.arange(0, 0.01 + newsteps, newsteps)
# funcx = np.cos(2*np.pi*100*t1) + np.cos(2*np.pi*3024*t1) + np.sin(2*np.pi*50000*t1)
# plt . figure ( figsize = (10 , 6) )
# plt . plot (t1, funcx )
# plt . ylabel ('Y output')
# plt . xlabel ('seconds')
# plt . title ('Graphing x(t)')
# plt . grid ()
# # warray, marray,parray=sig.bode((numer,denom),w)
# thing1,thing2=sig.bilinear(numer,denom,freq)
# #spits
# final1=sig.lfilter(thing1,thing2,funcx)
# plt . figure ( figsize = (11 , 7) )
# plt . plot (t1, final1)
# plt . ylabel ('Y output')
# plt . xlabel ('seconds')
# plt . title ('Part 2 filtering the given function x(t) ')
# plt . grid ()


#time to generate the bode plots

step_size = 1
w = np.arange(1e3, 1e6+step_size, step_size)

#rreferencing lab 10 in plotting
# plt.figure(figsize=(11,7))
# sys=con.TransferFunction(num,den) 
# _ = con.bode(sys, w, dB= True, Hz = True, deg=True, Plot=True)

plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , np.arange(1, 1600 +step_size, step_size)*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )
plt.title('0 to 1600 to ')


plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , np.arange(1600, 2200 +step_size, step_size)*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )
plt.title('1600 to 2200 to ')

plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , np.arange(2200, 1e6 +step_size, step_size)*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )
plt.title('2200 to 1e6 to ')


##now doing the fft of filter signal
#should see a difference if chose correct rlc 
#parameters

#repeating the graphs of the fft before the filter
freq2,X_mag2,X_phi2=myfft(final1,fs)#labeling the returns so i can acces them when i go to plot them
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq2 , X_mag2 )
# make_stem ( ax ,freq1 , X_phi1 )

plt . title ('Showing the all the band magnitude filtered')
plt.grid()
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('magnittude')
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq2 , X_phi2 )
plt . title ('Showing the all the band phase filtered')
plt.grid()
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('phase')
plt . show ()
#repeating the above to show specific intervals
#showing 0 to 1800 magnitude and phase
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq2 , X_mag2 )
plt . title ('Showing the  band magnitude 0-1600 filtered')
plt.grid()
plt.xlim(0, 1600)
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('magnittude')
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq2 , X_phi2)
plt . title ('Showing the  the band phase0-1600 filtered')
plt.grid()
plt.xlim(0, 1600)
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('phase')
plt . show ()


#repeating the above to show specific intervals
#showing 1800 to 2000 magnitude and phase
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq2, X_mag2 )
plt . title ('Showing the  band magnitude 1600-2200 filtered')
plt.grid()
plt.xlim(1600, 2200)
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('magnittude')
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq2 , X_phi2 )
plt . title ('Showing the the band phase1600-2200 filtered')
plt.grid()
plt.xlim(1600, 2200)
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('phase')
plt . show ()



#
#repeating the above to show specific intervals
#showing 2000 to 100000 magnitude and phase
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq2 , X_mag2 )
plt . title ('Showing the  band magnitude 2200-100000 filtered')
plt.grid()
plt.xlim(2200, 100000)
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('magnittude')
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax ,freq2 , X_phi2)
plt . title ('Showing the all the band phase 2200-100000 filtered')
plt.grid()
plt.xlim(2200, 100000)
plt . xlabel ('Frequency [Hz]')
plt . ylabel ('phase')
plt . show ()




