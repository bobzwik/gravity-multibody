# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: GNU GPLv3
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi
from math import sqrt
from scipy.integrate import ode
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import timeit
import time

class Body:
    
    def __init__(self, mass = 100000, state = np.zeros(6)):
        self.mass = mass
        self.state = state

    def updateHist(self):
        if hasattr(self, 'stateHistory'):
            self.stateHistory = np.vstack((self.stateHistory, self.state))
        else:
            self.stateHistory = self.state

    @property
    def mass(self):
        return self._mass
    
    @mass.setter
    def mass(self, newmass):
        self._mass = newmass
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, newstate):
        if type(newstate) == tuple:
            self.state = newstate[0]
            self.updateHist()
        else:
            if not type(newstate).__module__ == np.__name__:
                newstate = np.array(newstate)
            if len(newstate.shape) == 1:
                match newstate.shape[0]:
                    case 3:
                        newstate = np.append(newstate, [0,0,0])
                    case 6:
                        pass
                    case _:
                        raise TypeError('state must be a list, tuple, array of 3 or 6 items.')
            else:
                raise TypeError('Need 1-D array for state.')
            self._state = newstate
        
    
    


class SystemOfBodies:

    def __init__(self, bodies):
        self.bodies = bodies
        self.n = len(bodies)
        self.masses = np.array([body.mass for body in bodies])
        self.states = np.reshape([body.state for body in bodies], self.n*6)


        self.integrator = ode(self.state_dot).set_integrator('dopri5', first_step='0.0001', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.states, 0)
    
    # @property
    # def masses(self):
    #     return self._masses
    
    # @masses.setter
    # def masses(self, newMasses):
    #     self._masses = newMasses

    # @property
    # def states(self):
    #     return self._states
    
    # @states.setter
    # def states(self, newStates):
    #     self._states = newStates

    def distances(self, currentStates):
        n = self.n
        states = currentStates.reshape((n,6))
        rx = np.empty((n,n))
        ry = np.empty((n,n))
        rz = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                if j==i:
                    rx[i][j] = 0
                    ry[i][j] = 0
                    rz[i][j] = 0
                else:
                    rx[i][j] = states[j][0]-states[i][0]
                    ry[i][j] = states[j][1]-states[i][1]
                    rz[i][j] = states[j][2]-states[i][2]
        r = np.sqrt(np.power(rx, 2) + np.power(ry, 2) + np.power(rz, 2))    # r[i][j]  = sqrt(rx[i][j]**2 + ry[i][j]**2 + rz[i][j]**2)
        return rx, ry, rz, r
    
    def forces(self, rx, ry, rz, r):
        masses = self.masses
        n = self.n
        G = 6.674e-11       # (m3)(kg−1)(s−2)
        fx = np.empty((n,n))
        fy = np.empty((n,n))
        fz = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                if j==i:
                    fx[i][j] = 0
                    fy[i][j] = 0
                    fz[i][j] = 0
                else:
                    fx[i][j] = G*masses[i]*masses[j]*rx[i][j]/(r[i][j])**3      # (G*m1*m2/r**2)*rx/r
                    fy[i][j] = G*masses[i]*masses[j]*ry[i][j]/(r[i][j])**3
                    fz[i][j] = G*masses[i]*masses[j]*rz[i][j]/(r[i][j])**3
        f = np.sqrt(np.power(fx, 2) + np.power(fy, 2) + np.power(fz, 2))    # f[i][j]  = sqrt(fx[i][j]**2 + fy[i][j]**2 + fz[i][j]**2)
        return fx, fy, fz, f

    def state_dot(self, t, states):
        masses = self.masses
        n = self.n
        rx, ry, rz, r = self.distances(states)

        G = 6.674e-11       # (m3)(kg−1)(s−2)
        ax = np.empty((n,n))
        ay = np.empty((n,n))
        az = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                if j==i:
                    ax[i][j] = 0
                    ay[i][j] = 0
                    az[i][j] = 0
                else:
                    ax[i][j] = G*masses[j]*rx[i][j]/(r[i][j])**3      # ((G*m1*m2/r**2)*rx/r)/m1
                    ay[i][j] = G*masses[j]*ry[i][j]/(r[i][j])**3
                    az[i][j] = G*masses[j]*rz[i][j]/(r[i][j])**3
        sdot = np.empty((6*n))
        for i in range(n):
            sdot[i*6:(i+1)*6] = np.array([states[3+i*6], states[4+i*6], states[5+i*6], np.sum(ax[i]), np.sum(ay[i]), np.sum(az[i])])
        return sdot
    
    def update(self, t, Ts):

        self.states = self.integrator.integrate(t+Ts)

        rx, ry, rz, r = self.distances(self.states)
        fx, fy, fz, f = self.forces(rx, ry, rz, r)

        bodyStates = self.states.reshape((self.n,6))
        for body, i in zip(self.bodies, range(self.n)):
            body.state = (bodyStates[i], Ts)

        output = np.array([rx, ry, rz, r, fx, fy, fz, f])
        return output



start_time = time.time()
bodies = []
for i in range(3):
    bodies.append(Body())

bodies[0].state = [0,0,0]
bodies[1].state = [2,3,4]
bodies[2].state = [-1,2,-3]
system = SystemOfBodies(bodies)

t  = 0
Ts = 1
Tf = 10000

SimData = np.array([system.update(0, 0)])

while t < Tf:
    output = system.update(t, Ts)
    SimData = np.concatenate((SimData, np.array([output])), axis=0)
    t += Ts

end_time = time.time()
print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for body in bodies:
    x = body.stateHistory[:,0]
    y = body.stateHistory[:,1]
    z = body.stateHistory[:,2]
    ax.plot(x, y, z)
plt.show()