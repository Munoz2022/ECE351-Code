# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 08:12:30 2022

@author: defaultuser0
"""


###############################################################
#                                                             #
#Isaias Munoz                                                 # 
#Ece351                                                       # 
#Lab 3                                                        #
#9/1/22                                                       #
#                                                             #
###############################################################   
import math
#import numpy
import scipy.signal
import time
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt



steps = 1e-2 # Define step size
t = np . arange (0 , 20 + steps , steps )
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
fig1=stepfunc(t-2)-stepfunc(t-9)
fig2=(np.exp(-t))
fig3=rampfunc(t-2)*(stepfunc(t-2)-stepfunc(t-3))+rampfunc(4-t)*(stepfunc(t-3)-stepfunc(t-4))

plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t,fig1)
plt.grid()
plt.ylabel('f1 Output')
plt.title('Ploting f1,f2,f3 using unit and ramp functions')

plt.subplot(3,1,2)
plt.plot(t,fig2)
plt.grid()
plt.ylabel('f2 Output')

plt.subplot(3,1,3)
plt.plot(t,fig3)
plt.grid()
plt.ylabel('f3 Output')




##############################################################################



###########################################################################


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
   

#calling cnvultions f1 anf2

tconvo = np . arange (0 , 2*t[len(t)-1] , steps )

convof1andf2=my_conv(fig1,fig2)
convof2andf3=my_conv(fig2,fig3)
convof1andf3=my_conv(fig1,fig3)
plt . figure ( figsize = (10 , 7) )
plt . plot (tconvo , convof1andf2 )
plt . grid ()
plt . ylabel ('Y output')
plt . xlabel ('t convolution')
plt . title ('Convolution of f1 and f2')

plt . figure ( figsize = (10 , 7) )
plt . plot (tconvo , convof2andf3 )
plt . grid ()
plt . ylabel ('Y output')
plt . xlabel ('t convolution')
plt . title ('Convolution of f2 and f3')


plt . figure ( figsize = (10 , 7) )
plt . plot (tconvo , convof1andf3 )
plt . grid ()
plt . ylabel ('Y output')
plt . xlabel ('t convolution')
plt . title ('Convolution f1 and f3')
#verifying library convultion


confirmation1=scipy.signal.convolve(fig1,fig2)
confirmation2=scipy.signal.convolve(fig2,fig3)
confirmation3=scipy.signal.convolve(fig1,fig3)

plt . figure ( figsize = (10 , 7) )
plt . plot (tconvo , confirmation1 )
plt . grid ()
plt . ylabel ('Y output')
plt . xlabel ('t convolution')
plt . title ('Convolution of f1 and f2 using library function')

plt . figure ( figsize = (10 , 7) )
plt . plot (tconvo , confirmation2 )
plt . grid ()
plt . ylabel ('Y output')
plt . xlabel ('t convolution')
plt . title ('Convolution of f2 and f3 using library function')

plt . figure ( figsize = (10 , 7) )
plt . plot (tconvo , confirmation3 )
plt . grid ()
plt . ylabel ('Y output')
plt . xlabel ('t convolution')
plt . title ('Convolution of f1 and f3 using library function')













############################################################################

























