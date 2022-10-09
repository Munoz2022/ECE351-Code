# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 09:08:52 2022

@author: defaultuser0
"""

###############################################################
#                                                             #
#Isaias Munoz                                                 # 
#Ece351                                                       # 
#Lab 7                                                        #
#10/06/22                                                      #
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

#------------------------------------------------------------------
#part 1
#g=(s+9)/(s-8)(s+2)(S+4)

#a=(s+4)/(s+1)(s+3)

#b=(s+2)+(s+14)

#first function G finding poles,roots,and k

num1=[1,9]
# den1=[1,-6,-16]
den1=[1,-2,-40,-64]

# zeroe1,pole1,k1=sig.tf2zpk(num1,den1)
zeroe1,pole1,k1=sig.tf2zpk(num1,den1)

print('this is zeroes for G(s)', zeroe1)
print('this is poles for G(s)', pole1)

#second function a finding poles,roots,and k
num2=[1,4]
den2=[1,4,3]
zeroe2,pole2,k2=sig.tf2zpk(num2,den2)
print('this is zeroes for A(s)', zeroe2)
print('this is poles for A(s)', pole2)

#Third function a finding poles,roots,and k

bfunc=[1,26,168]

rootsofb=np.roots(bfunc)

print('this is roots for B(s)', rootsofb)

steps = .001 # Define step size
t = np . arange (0 , 7 + steps , steps )

#array  of open looop after simplifyinh
numopen=[1,9]
denotopen = [1, -2, -40, -64]

this1, this2=scipy.signal.step((numopen,denotopen),T=t)

plt . figure ( figsize = (12 , 8) )
plt . plot (this1, this2 )
plt . ylabel ('Y output')
plt . xlabel ('t range')
plt . title ('open loop ')
plt . grid ()
#---------------------------------------------------------------------------
#part2



#H(s)=Y(s)/X(S)=GA/1+GB


#g=(s+9)/(s-8)(s+2)(S+4)
#num1/den1
#a=(s+4)/(s+1)(s+3)
#num2/den2
#b=(s+2)+(s+14)
#b

numerator=sig.convolve(num1,num2)
print('total numerator',numerator)

denominator1=sig.convolve(den2,den1)
print('denominator 1',denominator1)

barray=[1,26,168]
denominator2=sig.convolve(num1,sig.convolve(barray,den2))
print('denominator2',denominator2)
totaldenom=denominator1+denominator2
print('total denominator',denominator2)

this3, this4=scipy.signal.step((numerator,totaldenom),T=t)
# denominator2=sig.convolve(num1,barray)
# denominator2=sig.convolve(num1,barray)
# print('numerator',numerator2)

#h=(num1/num2)*(num2/den2)/1+(num1/den1)*b
plt . figure ( figsize = (12 , 8) )
plt . plot (this3, this4 )
plt . ylabel ('Y output')
plt . xlabel ('t range')
plt . title ('close loop ')
plt . grid ()
#------------

#-----------------------------------------------------------------------------


