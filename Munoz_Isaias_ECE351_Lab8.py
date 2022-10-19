# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 08:12:37 2022

@author: defaultuser0
"""


###############################################################
#                                                             #
#Isaias Munoz                                                 # 
#Ece351                                                       # 
#Lab 8                                                       #
#10/13/22                                                      #
#                                                             #
###############################################################   
import math
import numpy
import scipy.signal
import time
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
# import cmath

steps = 0.01
t = np.arange(0, 20 + steps, steps)
T = 8 #whatver value
w = (2*np.pi)/T
ak = 0  #this is zeroe because it is odd function
# Part 1 Tasks
#plotting the step response found by hand

# steps = .001 # Define step size
# t = np . arange (0 , 2 + steps , steps )
#k=2
def bkfunc(k): #found from 
    
    y=(2/(k*np.pi))*(1-np.cos(k*np.pi))
    return y


#trying to do for loop butits noooooooot workrringng
    # for i in range (k):
        
    #     z= print('This is bk term ' bkfunc(k))
    # return z


#Printing the values to the third term of bk and 2nd term of ak
print("first term ak(0): ", ak)
print("second term ak(1): ", ak)
print("first term bk(1): ", bkfunc(1))
print("second term bk(2): ", bkfunc(2))
print("third term bk(3): ", bkfunc(3))
    
    
# ##plotting the fouriers task 2
#    for i in np.arange(1, n + 1):
#         y = y + (2/(i * np.pi)) * (1 - np.cos(i * np.pi)) * (np.sin(i * w * t))       
#     return y

def fourierfunc(k,t):    
    #k=n or number of iterations
    y=0
    for i in range(1,k+1):
        # y+=bkfunc(i)*np.sin((2*np.pi*i*t)/T)
        # y+=(2/(i*np.pi))*(1-np.cos(i*np.pi))*np.sin((i*w*t)
         y +=  (2/(i * np.pi)) * (1 - np.cos(i * np.pi)) * (np.sin(i * w * t))       
   
    return y
    
 # y=0
 # for i in range(1,1):
 #     # y+=bkfunc(i)*np.sin((2*np.pi*i*t)/T)
 #     y+=(2/(i*np.pi))*(1-np.cos(i*np.pi))*np.sin((2*np.pi*i*t)/8)
fouri1=fourierfunc(1,t)
fouri3=fourierfunc(3, t)
fouri5=fourierfunc(15, t)
    
fouri50=fourierfunc(50, t)
fouri150=fourierfunc(150, t)
fouri1500=fourierfunc(1500, t)
    
plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,fouri1)
plt.grid()
plt.ylabel('y output at k=1')
plt.title('first 3 graphs k=1,3,15')

plt.subplot(3,1,2)
plt.plot(t,fouri3)
plt.grid()
plt.ylabel('y output at k=3')

plt.subplot(3,1,3)
plt.plot(t,fouri5)
plt.grid()
plt.ylabel('y output at k=15')

##########second plots

plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,fouri50)
plt.grid()
plt.ylabel('y output at k=50')
plt.title('next 3 graphs k=50,150,1500')

plt.subplot(3,1,2)
plt.plot(t,fouri150)
plt.grid()
plt.ylabel('y output at k=150')

plt.subplot(3,1,3)
plt.plot(t,fouri1500)
plt.grid()
plt.ylabel('y output at k=1500')



    
    
    
    
    
    
    
    
    
    
    
    
    