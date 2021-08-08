import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


st.title("Heat Equation Implementation :grin:")

solve_method = st.sidebar.selectbox(
    'Select your method',
    ('Forward Difference', 'Implicit', 'Theta'))
st.sidebar.header("Parameters")

dt = st.sidebar.slider('Insert the dt:', min_value = 0.001, max_value = 0.01, step = 0.001)
dx = st.sidebar.slider('Insert the dx:', min_value = 0.01, max_value = 0.5, step = 0.01)
L = st.sidebar.slider('Insert the length L of the bar:', min_value = 1., max_value = 10., step = 0.5)
I = st.sidebar.slider('Insert the f(0,t):', min_value = 1., max_value = 10., step = 0.5)
F = st.sidebar.slider('Insert the f(L,t):', min_value = 1., max_value = 10., step = 0.5)
A = st.sidebar.slider('Insert the A such f(x,0) = A:', min_value = 1., max_value = 10., step = 0.5)
T = st.sidebar.slider('Insert the total time:', min_value = 1., max_value = 3., step = 0.2)
alpha = st.sidebar.slider('Insert alpha:', min_value = 1., max_value = 3., step = 0.2)


def heat_equation(alpha,I,F,f,dx,dt):
  """
  Solving the heat equation:
    u_t = (alpha)^2*u_xx
    u(0,t) = I     u(1,t) = F
    u(x,0) = f(x)
  Using finite difference. Above dx is the mesh length of the bar size and dt
  is the mesh size in the time
  """
  leng = 1
  Nt = round(leng/dx)
  u = np.zeros((50,Nt+1))
  ts = np.linspace(0,leng,Nt+1)
  for i in range(Nt+1):
    u[0][i] = f(ts[i])
  for i in range(50):
    u[i][0], u[i][-1] = I, F
  for i in range(1,50):
    for j in range(1,Nt):
      u[i][j] = u[i-1][j] + alpha**2 * (dt/(dx**2)) * (u[i-1][j-1] + u[i-1][j+1] - 2*u[i-1][j])
  return u

def imp_heat_equation(alpha,I,F,f,dx,dt):
  """
  Solving the heat equation:
    u_t = (alpha)^2*u_xx
    u(0,t) = I     u(1,t) = F
    u(x,0) = f(x)
  Using finite difference. Above dx is the mesh length of the bar size and dt
  is the mesh size in the time
  """
  leng = L
  Nt = round(leng/dx)
  u = np.zeros((50,Nt+1))
  ts = np.linspace(0,leng,Nt+1)
  for i in range(Nt+1):
    u[0][i] = f(ts[i])
  for i in range(50):
    u[i][0], u[i][-1] = I, F
  aux_mat = np.zeros((Nt+1,Nt+1))
  my_lambda = alpha**2 * (dt/(dx**2))
  for i in range(1,Nt):
    aux_mat[i][i-1], aux_mat[i][i], aux_mat[i,i+1] = -1*my_lambda, 1 + 2*my_lambda, -1*my_lambda
  aux_mat[0][0], aux_mat[0][1] = 1, 0
  aux_mat[-1][-2], aux_mat[-1][-1] = 0, 1
  aux_mat = np.linalg.inv(aux_mat)
  for i in range(1,50):
    u[i] = np.dot(aux_mat, np.transpose(u[i-1]))
  return u

st.write("We have the following heat equation and the selected configurations for our problem.")

st.latex(r''' \frac{\partial u}{\partial t} = \alpha^2 \frac{\partial^2 u}{\partial x^2 }''')
st.latex(r''' f(0,t) = ''' + str(I) + r''', \qquad f(x,0) = ''' + str(A) + r''', \qquad f(L,t) = ''' + str(F))
st.latex(r'''\alpha = ''' + str(alpha) + r''', \qquad L =  ''' + str(L))
st.latex(r''' dt = ''' + str(dt) + r''', \qquad dx = ''' + str(dx))



class Dummy():
    def dummy_forward(self , data=None):
        fig, ax = plt.subplots() #solved by add this line 
        ax = sns.heatmap(heat_equation(alpha,I,F,lambda x : A, dx, dt))
        return fig
    def dummy_implicit(self, data=None):
        fig, ax = plt.subplots() #solved by add this line 
        ax = sns.heatmap(imp_heat_equation(alpha,I,F,lambda x : A, dx, dt))
        return fig

eco = Dummy()
if solve_method == 'Forward Difference':
    st.write("Here we have a heatmap with the y-axis for the time, the x-axis for the position of the bar and the color for the temperature. \n Here we have the Forward Difference simulaiton.")
    st.pyplot(eco.dummy_forward())
    st.write("We used the following forward difference scheme.")
    st.latex(r'''\lambda = \alpha^2 \frac{\Delta t}{(\Delta x)^2} ''')
    st.latex(r''' u_{i,n+1} = (1-2\lambda)u_{i,n} + \lambda (u_{i,n-1}+u_{i,n+1})''')
elif solve_method == 'Implicit':
    st.write("We used the following forward difference scheme.")
    st.latex(r'''\lambda = \alpha^2 \frac{\Delta t}{(\Delta x)^2}, \qquad a = 1 + 2\lambda, \qquad b = -\lambda ''')
    st.latex(r''' \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & \cdots & 0 & 0 \\
        b & a & b & 0 & 0 & \cdots & 0 & 0 \\ 
        0 & b & a & b & 0 & \cdots & 0 & 0 \\
        \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & \cdots & 0 & 1 \\\end{bmatrix}^{-1} \mathbf{u}_n = \mathbf{u}_{n+1}''')
    st.write("And here is our simulation.")
    st.pyplot(eco.dummy_implicit())


