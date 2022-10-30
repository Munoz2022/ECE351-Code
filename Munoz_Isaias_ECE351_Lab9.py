# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 08:39:18 2022

@author: defaultuser0
"""

###############################################################
#                                                             #
#Isaias Munoz                                                 # 
#Ece351                                                       # 
#Lab 9                                                    #
#10/27/22                                                      #
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
# import cmath
steps = 0.01
# T = 8 #whatver value

# t = np.arange(0, 2 , T)
# T = 8 #whatver value
# w = (2*np.pi)/T
fs=100
T=1/fs
t = np.arange(0, 2 , T)
x1=np.cos(np.pi*2*t)
x2=5*np.sin(np.pi*2*t)
x3=2*np.cos(np.pi*4*t-2)+np.sin((12*np.pi*t)+3)**2

def myfft(x,fs):
    
   
    N = len(x) # find the length of the signal
    X_fft = scipy.fftpack.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) # shift zero frequency components
# to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
# signal , (fs is the sampling frequency and
# needs to be defined previously in your code


    X_mag = np.abs(X_fft_shifted)/N # compute the magnitudes of the signal
    X_phi = np.angle(X_fft_shifted) # compute the phases of the signal
# ----- End of user defined function ----- #
    return freq,X_mag,X_phi #Need to assign these to a value 
##Starting plots for task 1,2,3
freq1,X_mag1,X_phi1=myfft(x1,fs)#labeling the returns so i can acces them when i go to plot them
freq2,X_mag2,X_phi2=myfft(x2, fs)
freq3,X_mag3,X_phi3=myfft(x3, fs)


#performing the fourier of the three functions task 1 and 2 and 3
# fft1=myfft(x1,fs)
# fft2=myfft(x2,fs)
# fft3=myfft(x3,fs)


#Performing task 1 for the first signal
plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,x1)
plt.grid()
plt.ylabel('Y output')
plt.title('Task 1')


plt.subplot(3,2,3)
plt.stem(freq1,X_mag1)
plt.grid()
plt.ylabel('magnitutde ')

plt.subplot(3,2,4)
plt.stem(freq1,X_mag1)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('magnitutde ')


plt.subplot(3,2,5)
plt.stem(freq1,X_phi1)
plt.grid()
plt.ylabel('phase')

plt.subplot(3,2,6)
plt.stem(freq1,X_phi1)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('phase')

#Performing task 2 for the first signal
plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,x2)
plt.grid()
plt.ylabel('Y output')
plt.title('Task 2')


plt.subplot(3,2,3)
plt.stem(freq2,X_mag2)
plt.grid()
plt.ylabel('magnitutde ')

plt.subplot(3,2,4)
plt.stem(freq2,X_mag2)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('magnitutde ')


plt.subplot(3,2,5)
plt.stem(freq2,X_phi2)
plt.grid()
plt.ylabel('phase')

plt.subplot(3,2,6)
plt.stem(freq2,X_phi2)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('phase')



#Performing task 2 for the first signal
plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,x3)
plt.grid()
plt.ylabel('Y output')
plt.title('Task 3')


plt.subplot(3,2,3)
plt.stem(freq3,X_mag3)
plt.grid()
plt.ylabel('magnitutde ')

plt.subplot(3,2,4)
plt.stem(freq3,X_mag3)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('magnitutde ')


plt.subplot(3,2,5)
plt.stem(freq3,X_phi3)
plt.grid()
plt.ylabel('phase')

plt.subplot(3,2,6)
plt.stem(freq3,X_phi3)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('phase')



def stepfunc ( t ) :  #creating step function
    y = np.zeros ( t.shape  ) #initializing y as all zeroes
    for i in range (len ( t ) ): #startingforloop
        if  t [ i ]<0:  
            y[i]=0
        else: 
            y[i]=1                  
    return y 



def myfft2(x,fs):
    
   
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
        if X_mag[i]<1e-10:

            X_phi[i]=0
        # else: 
            # X_phi = np.angle(X_fft_shifted) # compute the phases of the signal
# ----- End of user defined function ----- #
    return freq,X_mag,X_phi #Need to assign these to a value 
#task 2 doing it
freq1new,X_mag1new,X_phi1new=myfft2(x1,fs)#labeling the returns so i can acces them when i go to plot them
freq2new,X_mag2new,X_phi2new=myfft2(x2, fs)
freq3new,X_mag3new,X_phi3new=myfft2(x3, fs)


#Performing task 1 for the first signal making it clear
plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,x1)
plt.grid()
plt.ylabel('Y output')
plt.title('Task 4')


plt.subplot(3,2,3)
plt.stem(freq1new,X_mag1new)
plt.grid()
plt.ylabel('magnitutde ')

plt.subplot(3,2,4)
plt.stem(freq1new,X_mag1new)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('magnitutde ')


plt.subplot(3,2,5)
plt.stem(freq1new,X_phi1new)
plt.grid()
plt.ylabel('phase')

plt.subplot(3,2,6)
plt.stem(freq1new,X_phi1new)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('phase')

#Performing task 2 for the first signal making it clear
plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,x2)
plt.grid()
plt.ylabel('Y output')
plt.title('Task 4')


plt.subplot(3,2,3)
plt.stem(freq2new,X_mag2new)
plt.grid()
plt.ylabel('magnitutde ')

plt.subplot(3,2,4)
plt.stem(freq2new,X_mag2new)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('magnitutde ')


plt.subplot(3,2,5)
plt.stem(freq2new,X_phi2new)
plt.grid()
plt.ylabel('phase')

plt.subplot(3,2,6)
plt.stem(freq2new,X_phi2new)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('phase')



#Performing task 3 for the first signal making it clear
plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,x3)
plt.grid()
plt.ylabel('Y output')
plt.title('Task 4')


plt.subplot(3,2,3)
plt.stem(freq3new,X_mag3new)
plt.grid()
plt.ylabel('magnitutde ')

plt.subplot(3,2,4)
plt.stem(freq3new,X_mag3new)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('magnitutde ')


plt.subplot(3,2,5)
plt.stem(freq3new,X_phi3new)
plt.grid()
plt.ylabel('phase')

plt.subplot(3,2,6)
plt.stem(freq3new,X_phi3new)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('phase')










T5 = 8 #whatver value
w5 = (2*np.pi)/T5
ak = 0  #this is ze
k=15
##Doing task 5 now 
timerange = np.arange(0,16  , T)

def fourierfunc(k,t):    
    #k=n or number of iterations
    y=0
    for i in range(1,k+1):
        # y+=bkfunc(i)*np.sin((2*np.pi*i*t)/T)
        # y+=(2/(i*np.pi))*(1-np.cos(i*np.pi))*np.sin((i*w*t)
         y +=  (2/(i * np.pi)) * (1 - np.cos(i * np.pi)) * (np.sin(i * w5 * timerange))       
   
    return y


task5func=fourierfunc(k,T5)
freqtask5,X_magtask5,X_phitask5=myfft2(task5func,fs)#labeling the returns so i can acces them when i go to plot them

plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(timerange,task5func)
plt.grid()
plt.ylabel('Y output')
plt.title('Task 5')


plt.subplot(3,2,3)
plt.stem(freqtask5,X_magtask5)
plt.grid()
plt.ylabel('magnitutde ')

plt.subplot(3,2,4)
plt.stem(freqtask5,X_magtask5)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('magnitutde ')


plt.subplot(3,2,5)
plt.stem(freqtask5,X_phitask5)
plt.grid()
plt.ylabel('phase')

plt.subplot(3,2,6)
plt.stem(freqtask5,X_phitask5)
plt.xlim([-2,2])
plt.grid()
plt.ylabel('phase')







# plt.stem(freq , X_mag) # you will need to use stem to get these plots to be
# plt.stem(freq , X_phi) # correct , remember to label all plots appropriately
