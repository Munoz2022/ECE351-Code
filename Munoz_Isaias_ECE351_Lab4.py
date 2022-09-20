# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 08:51:55 2022

@author: defaultuser0
"""


###############################################################
#                                                             #
#Isaias Munoz                                                 # 
#Ece351                                                       # 
#Lab 4                                                       #
#9/15/22                                                       #
#                                                             #
###############################################################   
import math
import numpy
import scipy.signal
import time
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


############################################################################
#Part 1 graphin the h(t) 
steps = 1e-2 # Define step size
t = np . arange (-10 , 10 + steps , steps )
# q = np . arange (-1 , 14 + steps , steps )

def stepfunc ( t ) :  #creating step function
    y = np.zeros ( t.shape  ) #initializing y as all zeroes
    for i in range (len ( t ) ): #startingforloop
        if  t [ i ]<0:  
            y[i]=0
        else: 
            y[i]=1                  
    return y 

def rampfunc ( t ) : 
    y = np.zeros ( t . shape ) 
    for i in range (len ( t ) ): 
        if  t [ i ]<0:
            y[i]=0
        else: 
            y[i]=t[i]                   
    return y 
####Better to make a function 

#w=2pif 
w=2*math.pi*.25

h1t=(np.exp(-t))*(stepfunc(t)-stepfunc(t-3))
h2t=stepfunc(t-2)-stepfunc(t-6)
h3t=np.cos(w*t)*stepfunc(t)


plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,h1t)
plt.grid()
plt.ylabel('h1(t) Output')
plt.title('Ploting h(t) vs time')

plt.subplot(3,1,2)
plt.plot(t,h2t)
plt.grid()
plt.ylabel('Ploting h2(t) vs time')

plt.subplot(3,1,3)
plt.plot(t,h3t)
plt.grid()
plt.ylabel('Ploting h3(t) vs time')
#############################################################################



#########################################################################
#part 2 covolving h(t) with a step function as its forcing function
#using convolcing function created in lab3
def my_conv(f1,f2):
    
    Nf1=len(f1)
    Nf2=len(f2)   #Just defining a length  being passed into function
    
    x1extended=np.append(f1,np.zeros((1,Nf2-1))) #Combining both  and extendning
    x2extended=np.append(f2,np.zeros((1,Nf1-1))) #Combining botha nd extedning

    result=np.zeros(x1extended.shape) #iniitalizing new result array using one the new arrays above

    for i in range(Nf2+Nf1-2): #combin lengths of f1 nad f2
        result[i]=0#Is this initliazing the resultant array?
        for j in range(Nf1):
            #if(len(Nf1) and len(Nf2) > len(x1extended)):
            if(i-j+1>0):
                try:
                    result[i]+=x1extended[j]*x2extended[i-j+1]
                except:
                    print(i,j)
    return result    
forcingfunc=stepfunc(t)

h1c = my_conv(h1t, forcingfunc)*steps
h2c = my_conv(h2t, forcingfunc) *steps
h3c = my_conv(h3t, forcingfunc)*steps

#not sure why times 2???
#the time axis should not matter where to starts
#my time axis needs to match with my new length of the convolution created which is doubled
#original is from -10 to 10 so a length of 20
#now after the convolution i double it and it reads -20 to 20 meaning length of 40
#now my starting point of my time array is still at -10 to 10 which is 20 therefore i
#have to double that bad boy as well or do it like below
tconvo = np . arange (2*t[0] , 2*t[len(t)-1]+steps , steps )
plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(tconvo,h1c)
plt.grid()
plt.ylabel('Output')
plt.title('h1, h2, and h3 Step Response')
plt.subplot(3,1,2)
plt.plot(tconvo,h2c)
plt.grid()
plt.ylabel('Output')
plt.subplot(3,1,3)
plt.plot(tconvo,h3c)
plt.grid()
plt.ylabel('Output')



# thand = np . arange (-10 , 10 + steps , steps )

#plotting the by hand convolution

handh1= .5*(1-np.exp(-2*t))*stepfunc(t)-.5*(1-np.exp(-2*(t-3)))*stepfunc(t-3)
handh2=(t-2)*stepfunc(t-2)-(t-6)*stepfunc(t-6)
handh3=(1/w)*np.sin(w*t)*stepfunc(t)



plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,handh1)
plt.grid()
plt.ylabel('Output')
plt.title('h1, h2, and h3 Step Response by hand')

plt.subplot(3,1,2)
plt.plot(t,handh2)
plt.grid()
plt.ylabel('Output')


plt.subplot(3,1,3)
plt.plot(t,handh3)
plt.grid()
plt.ylabel('Output')











#since im convolving the step response since its  a step func its just an intgral
# of h(t)*u(t)=integral of h(t) in paper at least


# t2 = np . arange (-10 , 10 + steps , steps )







