# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 21:36:39 2022

@author: defaultuser0
"""

###############################################################
#                                                             #
#Isaias Munoz                                                 # 
#Ece351                                                       # 
#Lab 2                                                        #
#9/1/22                                                       #
#                                                             #
###############################################################   
import math
import numpy
import scipy.signal
import time
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt



steps = .2e-2 # Define step size
#creating and plotting cosine(t) 
###############################################################################
t = np . arange (0 , 10 + steps , steps )

def func1 ( t ) : # The only variable sent to the function is t
    y = np.zeros ( t . shape ) # initialze y(t) as an array of zeros
    for i in range (len ( t ) ) : # run the loop once for each index of t
         if i < ( len( t ) + 1) : #works both ways with an if or withouth an if
            y [ i ] = np . cos ( t [ i ])
    return y # send back the output stored in an array


y = func1 ( t ) # call the function we just created


#plotting the function I just created
plt . figure ( figsize = (10 , 7) )
plt . plot (t , y )
plt . grid ()
plt . ylabel ('cos(t)' )
plt . xlabel ('time')
plt . title ('Plotting cosine task 1')
################################################################################





#creating a stepfunction and ramp function and using it to plot fig2
###############################################################################
# >>> np.arange(start=1, stop=10, step=3)
# array([1, 4, 7])

t = np . arange (-5 , 10 + steps , steps )
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

tryone=stepfunc(t)
try2=rampfunc(t)

plt . figure ( figsize = (10 , 7) )
plt . plot (t , tryone )
plt . grid ()
plt . ylabel ('Y output')
plt . xlabel ('t')
plt . title ('plotting step func to make sure it works')

plt . figure ( figsize = (10 , 7) )
plt . plot (t , try2 )
plt . grid ()
plt . ylabel ('Y output')
plt . xlabel ('t')
plt . title ('plotting ramp func to make sure it works')

fig2=rampfunc(t)-rampfunc(t-3)+5*stepfunc(t-3)-2*stepfunc(t-6)-2*rampfunc(t-6)


plt . figure ( figsize = (10 , 7) )
plt . plot (t , fig2 )
plt . grid ()
plt . ylabel ('Y output')
plt . xlabel ('t')
plt . title ('plotting FIgure 2 using step and ramp functions')

##############################################################################









##############################################################################
#Apply a time reversal and plot the result.
def timereversalfunc(temp): #Much easier when making a function!!!!!!! took forever

    timereverse=np.zeros(temp.shape)

    for i in range (0,len(temp)-1):
        
        timereverse[i]=temp[(len(temp)-1)-i] #imputting values into array reversed
                    
    return timereverse

reversalz=timereversalfunc(fig2)  #calling timereversal and passing it original fig

plt . figure ( figsize = (10 , 7) )
plt . plot (t , reversalz)
plt . grid ()
plt . ylabel ('Y output' )
plt . xlabel ('time')
plt . title ('Plotting Fig 2 with timereversal')


    
#############################################################################




##############################################################################
#  apply f(t-4) and f(-t-4)
#this is a long way of shifitiing the t axis
#Apply time-shift operations f (t −4) and f (−t −4) and plot the results.
# steps = .2e-2 # Define step size
# q = np . arange (-1 , 14 + steps , steps ) #this also works by just chaning
# your range of numbers because your chaing your t shifting no need
# to write a functuon well maybe, not quiet a function is better tbh
#other way doesnt work because steeps to 14 way to long

def functminus4(temp2,shifter):

        temp2= temp2 +shifter#shifting your time scale instaed of the actual figrue
     # temp2.rotate(4)           
        return temp2

tminus4=functminus4(t,4)
# fig2shift=rampfunc(q)-rampfunc(q-3)+5*stepfunc(q-3)-2*stepfunc(q-6)-2*rampfunc(q-6)

plt . figure ( figsize = (10 , 7) )
plt . plot (tminus4 , fig2)
plt . grid ()
plt . ylabel ('Y output' )
plt . xlabel ('time shifted')
plt . title ('Plotting Fig 2 with f(t-4)')


# negtminusneg4=functminus4(-t,-4)#doesnt work cause your reverse both

plt . figure ( figsize = (10 , 7) )
plt . plot (tminus4 , reversalz)
plt . grid ()
plt . ylabel ('Y output' )
plt . xlabel ('time shifted')
plt . title ('Plotting Fig 2 with f(-t-4)')

# plt . figure ( figsize = (10 , 7) )
# plt . plot (q, fig2shift)
# plt . grid ()
# plt . ylabel ('Y output' )
# plt . xlabel ('time')
# plt . title ('Plotting Fig 2 with timereversal')


##############################################################################


#############################################################################

# Apply time scale operations f (t/2) and f (2t) and plot the results.

# def scalefunc(temp3,scale):
#     # temp3/=scale
#     # return temp3

#     for i in range(0,len(temp3)-1):
#         temp3[i]=temp3[i]*scale

#     return temp3
t = np . arange (-5 , 10 + steps , steps )
#much simpler thank making a function since you can divide the timescale and
#then simply graph your original function
tdivby2=t/2
ttimes2=t*2

# funcscale=scalefunc(t,2)


plt . figure ( figsize = (10 , 7) )
plt . plot (tdivby2 , fig2)
plt . grid ()
plt . ylabel ('Y output' )
plt . xlabel ('time divided by 2')
plt . title ('Plotting Fig 2 with f(t/2)')


plt . figure ( figsize = (10 , 7) )
plt . plot (ttimes2 , fig2)
plt . grid ()
plt . ylabel ('Y output' )
plt . xlabel ('time multiplied by 2')
plt . title ('Plotting Fig 2 with f(2t)')

############################################################################




############################################################################


#Calculate and plot the diritive of func2Plot
derivfig2 = np.diff(fig2)
xsize = np.arange(-5, 10, steps)
plt.figure(figsize=(10, 7))
plt.plot(xsize, derivfig2)
plt.ylabel('Y Output')
plt.xlabel('Time')
plt.title("Derivative of fig2")
plt.grid()




###########################################################################


