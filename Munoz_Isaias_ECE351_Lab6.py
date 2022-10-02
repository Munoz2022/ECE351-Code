# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 08:57:54 2022

@author: defaultuser0
"""


###############################################################
#                                                             #
#Isaias Munoz                                                 # 
#Ece351                                                       # 
#Lab 6                                                        #
#9/29/22                                                      #
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




#######################################################################
# Part 1 Tasks
#plotting the step response found by hand

steps = .001 # Define step size
t = np . arange (0 , 2 + steps , steps )


##neded my step func since it is used in my stepresponse
def stepfunc ( t ) :  #creating step function
    y = np.zeros ( t.shape  ) #initializing y as all zeroes
    for i in range (len ( t ) ): #startingforloop
        if  t [ i ]<0:  
            y[i]=0
        else: 
            y[i]=1                  
    return y 
#For a minute i thought it wasnt working but i had extra brackets no need
# to create a new time range
# t1 = np . arange (t[0] , t[len(t)-1]+steps , steps )
# ystep = np.zeros ( t.shape  ) #initializing y as all zeroe
def stepsponse(t):
    y=(.5 - .5*np.exp(-4*t) + np.exp(-6*t))*stepfunc(t)
    return y
# ystep=ystep=(.5 + np.exp(-6*t) - .5*np.exp(-4*t1))*stepfunc(t1)
ystep=stepsponse(t)

plt . figure ( figsize = (12 , 8) )
plt . plot (t , ystep )
plt . ylabel ('Y output')
plt . xlabel ('t convolution')
plt . title ('Step function by hand')
plt . grid ()



#task 2
#plotting H(S) found in prelab
#H(S)=s^2+6s+12/s^2+10s+24

# scipy.signal.step() takes two things essentially the transfer func denom and 
#numera and then the time range 


num = [1, 6, 12]
 #Creates a matrix for the numerator coeffecients of transfer func numera
den = [1, 10, 24] 
#Creates a matrix for the denominator coeffecients of transfer func denom

zout , mout = scipy.signal.step((num,den),T=t) #gives a step response
#scipy.signal.step can be shorten to sig.step() like in lab 3 

print(zout,mout)
plt . figure ( figsize = (12 , 8) )
plt . plot (zout , mout )
plt . ylabel ('Y output')
plt . xlabel ('t range')
plt . title ('Step function using scipy.signal.step ')
plt . grid ()


#task3

#Y(S)=H(S)X(S) where X(S) is step func=1/s
print ('this is the solution to the the first DEQ')
a = [1, 6, 12]
b = [1, 10, 24,0]



rout, pout,kout= sig.residue(a,b)
#scipy.signal.residue() can also be called
#rout gives u the roots so ur A and B and C term after the partial expansion
#pout gives us the poles where they cross in this case (0) and (s+4) and (s+6)
#kout dont know what it gives u lol
print (rout,pout,kout)

#Make and print the partial fraction decomp


#end of part 1 tasks
############################################################################




############################################################################
#Beginnig of Part 2 task 1

a1=[25250]
b1=[1,18,218,2036,9085,25250,0]
arout, apout,akout= sig.residue(a1,b1)

print (' this is the Root solutions to the long DEQ')
print(arout)
print (' this is the Poles solutions to the long DEQ')
print(apout)
# for i in range (len(apout)):
#     print( apout[i])
print (' this is the K (residue of a given term) solutions to the long DEQ')
print(akout)
#plottingnew time range
# steps = .001 # Define step size
tnew = np . arange (0 , 4.5 + steps , steps )

def cosmethod(arout,apout,tnew):
    y=0 

    for i in range (len ( arout ) ): #startingforloop
       # y+=2*abs(arout[i])*np.exp(np.real(apout[i]*tnew))*np.cos(np.imag(apout[i])*tnew+np.angle(arout[i]))*stepfunc(tnew))
        y += (2 * abs(arout[i]) * np.exp(np.real(apout[i]) * tnew)* np.cos(np.imag(apout[i]) * tnew + np.angle(arout[i]))) * stepfunc(tnew)
    return y 
# print(abs(3+4j))

ystepcosmet=cosmethod(arout,apout,tnew)

plt . figure ( figsize = (12 , 8) )
plt . plot (tnew , ystepcosmet )
plt . ylabel ('Y output')
plt . xlabel ('new time range')
plt . title ('Step function by cosine method')
plt . grid ()
#from previous a1 and b1
    #a1=[25250]
# b1=[1,18,218,2036,9085,25250,0]



newb1 = [1, 18, 218, 2036, 9085, 25250]

lout, nout=scipy.signal.step((a1,newb1),T=tnew)

# num = [1, 6, 12]
#  #Creates a matrix for the numerator coeffecients of transfer func numera
# den = [1, 10, 24] 
# #Creates a matrix for the denominator coeffecients of transfer func denom

# zout , mout = scipy.signal.step((num,den),T=t) #gives a step response
# #scipy.signal.step can be shorten to sig.step() like in lab 3 

# print(zout,mout)
plt . figure ( figsize = (12 , 8) )
plt . plot (lout, nout )
plt . ylabel ('Y output')
plt . xlabel ('t range')
plt . title ('Verify the cosine method using step function using scipy.signal.step')
plt . grid ()








