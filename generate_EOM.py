# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

"""
Using PyDy and Sympy, this script generates the equations for the state derivatives 
of a quadcopter in a 3-dimensional space. The states in this particular script are:

x, y, z           : Position of the drone's center of mass in the inertial frame, 
                    expressed in the inertial frame
xdot, ydot, zdot  : Velocity of the drone's center of mass in the inertial frame, 
                    expressed in the inertial frame
q0, q1, q2, q3    : Orientation of the drone in the inertial frame using quaternions
p, q, r           : Angular velocity of the drone in the inertial frame,
                    expressed in the drone's frame

Important note    : This script uses a frd body orientation (front-right-down) and 
                    a NED world orientation (North-East-Down). The drone's altitude is -z.

Other note        : In the resulting state derivatives, there are still simplifications
                    that can be made that SymPy cannot simplify (factoring).
"""

from sympy import symbols, Matrix, div
from sympy.physics.mechanics import *
from sympy.core.backend import sqrt

# Reference frames and Points
# ---------------------------
N = ReferenceFrame('N')  # Inertial Frame

No = Point('No')
B1 = Point('B1')  # Body 1
B2 = Point('B2')  # Body 2
B3 = Point('B3')  # Body 3
B4 = Point('B4')

# Variables
# ---------------------------
# x, y and z are the body's coordinates in the inertial frame, expressed with the inertial frame
# vx, vy and vz are the body's velocities in the inertial frame, expressed with the inertial frame

x1, y1, z1, vx1, vy1, vz1 = dynamicsymbols('x1 y1 z1 vx1 vy1 vz1')
x2, y2, z2, vx2, vy2, vz2 = dynamicsymbols('x2 y2 z2 vx2 vy2 vz2')
x3, y3, z3, vx3, vy3, vz3 = dynamicsymbols('x3 y3 z3 vx3 vy3 vz3')
x4, y4, z4, vx4, vy4, vz4 = dynamicsymbols('x4 y4 z4 vx4 vy4 vz4')

# First derivatives of the variables
x1d, y1d, z1d, vx1d, vy1d, vz1d = dynamicsymbols('x1 y1 z1 vx1 vy1 vz1', 1)
x2d, y2d, z2d, vx2d, vy2d, vz2d = dynamicsymbols('x2 y2 z2 vx2 vy2 vz2', 1)
x3d, y3d, z3d, vx3d, vy3d, vz3d = dynamicsymbols('x3 y3 z3 vx3 vy3 vz3', 1)

# Constants
# ---------------------------
m1, m2, m3, m4, G = symbols('m1 m2 m3 m4 G')

# Origin
# ---------------------------
No.set_vel(N, 0)

# Translation
# ---------------------------
B1.set_pos(No, x1*N.x + y1*N.y + z1*N.z)
B1.set_vel(N, B1.pos_from(No).dt(N))

B2.set_pos(No, x2*N.x + y2*N.y + z2*N.z)
B2.set_vel(N, B2.pos_from(No).dt(N)) 

B3.set_pos(No, x3*N.x + y3*N.y + z3*N.z)
B3.set_vel(N, B3.pos_from(No).dt(N))

B4.set_pos(No, x4*N.x + y4*N.y + z4*N.z)
B4.set_vel(N, B4.pos_from(No).dt(N))

# Create Bodies
# ---------------------------
Body1 = Particle('Body1', B1, m1)
Body2 = Particle('Body2', B2, m2)
Body3 = Particle('Body3', B3, m3)
BodyList = [Body1]

# Forces
# ---------------------------

F12 = (B1, G*m1*m2/(B2.pos_from(B1).magnitude()**2)*B2.pos_from(B1).normalize())
F13 = (B1, G*m1*m3/(B3.pos_from(B1).magnitude()**2)*B3.pos_from(B1).normalize())
F14 = (B1, G*m1*m4/(B4.pos_from(B1).magnitude()**2)*B4.pos_from(B1).normalize())

F21 = (B2, G*m2*m1/(B1.pos_from(B2).magnitude()**2)*B1.pos_from(B2).normalize())
F23 = (B2, G*m2*m3/(B3.pos_from(B2).magnitude()**2)*B3.pos_from(B2).normalize())

F31 = (B3, G*m3*m1/(B1.pos_from(B3).magnitude()**2)*B1.pos_from(B3).normalize())
F32 = (B3, G*m3*m2/(B2.pos_from(B3).magnitude()**2)*B2.pos_from(B3).normalize())

ForceList = [F12, F13, F14]

# Kinematic Differential Equations
# ---------------------------
kd = [vx1 - x1d, vy1 - y1d, vz1 - z1d]

# Kane's Method
# ---------------------------
KM = KanesMethod(N, q_ind=[x1, y1, z1], u_ind=[vx1, vy1, vz1], kd_eqs=kd)
(fr, frstar) = KM.kanes_equations(BodyList, ForceList)

# Equations of Motion
# ---------------------------
MM = KM.mass_matrix_full
kdd = KM.kindiffdict()
rhs = KM.forcing_full
MM = MM.subs(kdd)
rhs = rhs.subs(kdd)

MM.simplify()
print()
print('Mass Matrix')
print('-----------')
mprint(MM)

# rhs.simplify()
print('Right Hand Side')
print('---------------')
mprint(rhs)
print()

# So, MM*x = rhs, where x is the State Derivatives
# Solve for x
stateDot = MM.inv()*rhs
print('State Derivatives')
print('-----------------------------------')
mprint(stateDot)
print()

print(G*m1*m2/(B2.pos_from(B1).magnitude()**2)*B2.pos_from(B1).normalize())

print(G*m1*m2/(B2.pos_from(B1).magnitude()**2))
print(G*m1*m2/(B2.pos_from(B1).magnitude())**2)
print(B2.pos_from(B1).normalize())
# -G*m2*(x1 - x2)/ ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**(3/2)

# -G*m2*(y1 - y2)/ ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**(3/2)

