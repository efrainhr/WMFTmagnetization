'''
  ================================================================
  : class : ‘magnetization‘ -- WMFT and Kuz’min EoS magnetization
  ================================================================

  This module calculates the magnetization for a variety of alloys
  as prescribed by the WMFT and the equation of state developed
  by Kuz’min.

  Developed by Efrain Hernandez-Rivera (2019--2020)
  US Army Research Laboratory
  --
  THIS SOFTWARE IS MADE AVAILABLE ON AN "AS IS" BASIS
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, NEITHER
  EXPRESSED OR IMPLIED
'''

import numpy as np
from sympy import *
import matplotlib.pylab as plt

#atomic masses
Me = {'Ni':58.6934, 'Co':58.933195, 'Fe': 55.845, 'Gd':64}

#constants
k = 1.38e-23 # J/K
N0 = 8.49e28 # #Fe/m3
muB = 9.274e-24 # J/T

Na = 6.02214076e23
fa = 4*pi*1e-7 # funky magnetic conversion

#from Guo and Enomoto, Mat Trans JIM 41.8 911-916 (2000)
# note: a=100/Tc0*dTc/dx, b=1/mFe*dm/dx
coefs={'V': {'dTdx':7.5, 'a':0.72, 'dmdx':-2.68,'b':-1.22},\
       'Cr':{'dTdx':-1.5,'a':-0.14,'dmdx':-2.29,'b':-1.04},\
       'Mn':{'dTdx':-15.,'a':-1.44,'dmdx':-2.11,'b':-0.96},\
       'Co':{'dTdx': 12.,'a':1.15, 'dmdx':1.11, 'b':0.5},\
       'Ni':{'dTdx':-3.6,'a':-0.34,'dmdx':1.3, 'b':0.59},\
       'Mo':{'dTdx':0., 'a':0., 'dmdx':-2.11,'b':-0.96},\
       'Si':{'dTdx':-3.5,'a':-0.34,'dmdx':-2.29,'b':-1.04},\
       'labs':['dTc/dX','a','dm/dx','b']}

#from Kuz’min, PRB 7 184431 (2008)
# note: M0 (emu/g) Tc (K), p, kappa, a0 (MOe)
# : 1 [emu] = 1.078283e20[uB]
# : 1 [emu/g] * 1.078283e20 [uB/emu] * Ma [g/mol] / Na [mol/atom]
kuz ={'Gd':{'M0':266, 'Tc':293, 'p':1.50, 'k':0.35, 'a0':0.9},\
      'Ni':{'M0':58, 'Tc':631, 'p':0.28, 'k':0.47, 'a0':1.85},\
      'Fe':{'M0':222, 'Tc':1044,'p':0.25, 'k':0.18, 'a0':3.33},\
      'Co':{'M0':164, 'Tc':1390,'p':0.25, 'k':0.43, 'a0':3.70}}

class magnetization(object):
    '''
    Class to calculate magnetization curves as a function of
    temperature

    Inputs:
        a, b : Coeficents for calculation of the Curie
        temperature and magnetic moment, respectively. Values for
        several alloying elements (V, Cr, Mn, Co, Ni, Mo, Si) can be
        called as follows:

        >>> E = ’Ni’
        >>> a, b = coefs[E][’a’], coefs[E][’b’]

        [Guo and Enomoto, Mat Trans JIM 41.8 911-916 (2000)]

        Tc : Curie temperature (K)

        H : Magnetic field intensity (T)

        j : Angular momentum quantum number, should be a
        multiple of 1/2 (optional, default=1)

        maxT : Maximum temperature (K) to which magnetization is
        determined (optional, default=1100)

        dT : Discretization step for the temperature array (
        optional, default=0.1)

        m0 : Magnetic moment of pure iron (mu_B) (optional,
        default=2.2)
    '''

    def __init__(self,a,b,Tc,H,j=1.,maxT=1100,dT=0.1,m0=2.22):
        self.a = a
        self.b = b
        self.j = j
        self.m = muB*m0 # [Bohr magneton]
        self.Tc = Tc # [K]
        self.H = H # [T]
        self.maxT = maxT # maximum T to analyze [K]
        self.dT = dT # temperature discretization

        self.alpha, self.T = symbols('alpha T')

        self.Bj = (self.j+0.5)/self.j*coth(self.alpha*(self.j+0.5)/self.j)\
                  - 0.5/self.j*coth(0.5*self.alpha/self.j)
        self.dBj = diff(self.Bj,self.alpha)

        self.alphaSol = []

    def solveAlpha(self,a0=1000,verify=False):
        '''
        Solve WMFT for pure system, obtaining alpha parameters.
        Function will loop from 1 to Tm in 1 K increments.

        Inputs:
            a0 : Initial guess for alpha (optional, default=1000)

        Outputs:
            alphaSol : Array of alpha values between 1 K and Tm
        '''

        j, m, alpha, T, Tc = self.j, self.m, self.alpha, self.T, self.Tc
        eq = (j+1)/3./j*T/Tc*alpha - (j+1)/3./j*m*self.H/Tc/k
        dT = np.arange(1,int(self.maxT)+1,self.dT)

        for t in dT:
            self.alphaSol.append(nsolve((eq.subs(T,t) \
                    - self.Bj),(alpha),(a0), verify=verify))
            a0 = self.alphaSol[-1]

    def calcMag(self,X,i=0):
        '''
        Calculate temperature dependent magnetization for given
        alloying concentration

        Inputs:
            X : Concentration of alloying element

        Outputs:
            M : Array of magnetization values between 1 K and maxT
        '''
        a, b, j, alpha = self.a, self.b, self.j, self.alphaSol[i]
        m = self.m#*(1+b*X)
        Tc = self.Tc#*(1.+a*X)
        T = Tc*(1+a*X)*float(i)/Tc

        Bj = self.Bj.subs( self.alpha,alpha)
        dBj = self.dBj.subs(self.alpha,alpha)

        A = (k*T*a*alpha + (b-a)*m*self.H)/(k*T - 3.*k*self.Tc*j*dBj/(j+1.))
        M = fa*m*N0*(Bj + (dBj*A + Bj*b)*X)
        return M

    def landau(self,sig,m0=2.22,X=0.,p=0.25,k=0.18,a0=3.3):
        '''
        Calculate magnetization of pure Fe using Kuz’min application of Landau
        [Kuz’min, Phys Rev B 77, 184431 (2008)]

        Inputs:
            sig : Array for the reduced magnetization (M/M0)

            p, k, a0 : Fitting parameters for pure Fe as
            determined by Kuz’min (optional, default= 0.25, 0.18, 3.3)

        Outputs:
            [T, m] : Array of temperature and magnetization
            values between 1 K and Tc
        '''
        H = self.H/1e2 #field given in T
        u = 0.5*(k*sig**2. + (1.-k)*sig**4. - (H/(a0*(sig+1e-1))))
        t = ((1 - 2*u + p*p*u*u)**0.5 - p*u)**(2/3.)

        return np.array([t*self.Tc*(1+self.a*X), sig*m0*(1+self.b*X)])

if __name__=='__main__':
    E='Ni'; dT = 0.5; maxT = 2000
    T = np.arange(1,maxT+1,dT); n = T.size
    m0 = kuz[E]['M0']*1.078283e20*Me[E]/Na
    Tc = kuz[E]['Tc']
    wmft = magnetization(0,0,Tc,0,maxT=maxT, dT=dT, j=0.5, m0=m0)
    wmft.solveAlpha()
    M=np.array([wmft.calcMag(0,i=i) for i in range(n)])

    data = wmft.landau(np.linspace(1e-5,1,100001),p=kuz[E]['p'],k=kuz[E]['k'],a0=kuz[E]['a0'],m0=m0)

    plt.plot(data[0],data[1],label=r"Kuz’min",lw=2)
    plt.plot(T,M,'--',label=r'WMFT ($j=1/2$)',lw=2)

    wmft = magnetization(0,0,Tc,0,maxT=maxT, dT=dT, j=1., m0=m0)
    wmft.solveAlpha()
    M=np.array([wmft.calcMag(0,i=i) for i in range(n)])

    plt.plot(T,M,'--',label=r'WMFT ($j=1$)',lw=2)

    plt.xlim(0,700)
    plt.ylim(0,0.7)

    plt.axvline(Tc,lw=1,ls=':',c='k')
    plt.grid()

    plt.xlabel('Temperature (K)',fontsize=16)
    plt.ylabel(r'Magnetization ($\mu_B$)',fontsize=16)

    plt.legend()
    plt.savefig('Ni-magnetization.png')

