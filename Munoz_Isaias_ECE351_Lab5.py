# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 08:39:20 2022

@author: defaultuser0
"""

###############################################################
#                                                             #
#Isaias Munoz                                                 # 
#Ece351                                                       # 
#Lab 5                                                      #
#9/25/22                                                       #
#                                                             #
###############################################################   
import math
import numpy
import scipy.signal
import time
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


#######################################################################
#Part 1 graphin the hand solved time domain impulse as a function
steps = 1e-5 # Define step size
t = np . arange (0 , 1.2e-3 + steps , steps )
# q = np . arange (-1 , 14 + steps , steps )

def stepfunc ( t ) :  #creating step function
    y = np.zeros ( t.shape  ) #initializing y as all zeroes
    for i in range (len ( t ) ): #startingforloop
        if  t [ i ]<0:  
            y[i]=0
        else: 
            y[i]=1                  
    return y 


##defining the rlc circuit paramters to imput into my hand made function
#no need for RLC values since I calcualted them by hand
R=1000
L=.027
C=100e-9
w=18580
a=-5000
magn=1.924*10**8
pg=105.1*(np.pi/180) #convert to degrees

def handimpfunc(t,magn,w,a,pg):
# print(magn)

    y=((magn/w)*np.exp(a*t)*np.sin(w*t+pg))*stepfunc(t)

    return y
# print(pg)
handfunc=handimpfunc(t, magn, w, a, pg)

num = [0, L, 0] #Creates a matrix for the numerator
den = [R*L*C, L, R] #Creates a matrix for the denominator

tout , yout = sig.impulse ((num , den), T = t) #makes the impulse response h(t)

zout , mout = sig.step((num,den),T=t) #gives a step response

# H(S)=(1/R*C)/(s^2+(s/R*C)+1/((L*C)^2))

#Using final value theorem H(s)*u(s) in the laplace domain 
# ftheoremfun=

print(zout,mout)

plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,handfunc)
plt.grid()
plt.ylabel('h1(t) Output')
plt.title('Ploting impulse by hand and using scipy.signal.impulse()')

plt.subplot(3,1,2)
plt.plot(tout,yout)
plt.grid()
plt.ylabel('scipy.signal,impulse()')

plt.figure(figsize=(10,7))
plt.plot(zout,mout)
plt.grid()
plt.ylabel('output of step response')
plt.title('Step response of transfer function')













