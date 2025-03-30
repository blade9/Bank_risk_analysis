# -*- coding: utf-8 -*-
"""

@author: MarkWright

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


def VasicekNextRate(r, kappa, theta, sigma, dt=1/252):
  # Implements above closed form solution
  val1 = np.exp(-1*kappa*dt)
  val2 = (sigma**2)*(1-val1**2) / (2*kappa)
  out = r*val1 + theta*(1-val1) + (np.sqrt(val2))*np.random.normal()
  return out


# Vasicek simulation short rate
def VasicekSim(N, r0, kappa, theta, sigma, dt = 1/252):
  short_r = [0]*N # Create array to store rates
  short_r[0] = r0 # Initialize rates at $r_0$
  for i in range(1,N):
    short_r[i]=VasicekNextRate(short_r[i-1],kappa, theta, sigma, dt)
  return short_r


# Vasicek multi-simulation
def VasicekMultiSim(M, N, r0, kappa, theta, sigma, dt = 1/252):
  sim_arr = np.ndarray((N, M))
  for i in range(0,M):
    sim_arr[:, i] = VasicekSim(N, r0, kappa, theta, sigma, dt)
  return sim_arr

trsy = pd.read_excel(r'Treasury Curve Data.xlsx', sheet_name='Clean Data')
trsy_shifted = trsy.shift(periods=1)
trsy_changes = trsy.iloc[1:, 9]- trsy_shifted.iloc[1:, 9]

# Maximum Likelihood Estimation to calibrate parameters
def VasicekCalibration(rates, dt=1/252):
  n = len(rates)
  Ax = sum(rates[0:(n-1)])
  Ay = sum(rates[1:n])
  Axx = np.dot(rates[0:(n-1)], rates[0:(n-1)])
  Axy = np.dot(rates[0:(n-1)], rates[1:n])
  Ayy = np.dot(rates[1:n], rates[1:n])
  theta = (Ay * Axx - Ax * Axy) / (n * (Axx - Axy) - (Ax**2 - Ax*Ay))
  kappa = -np.log((Axy - theta * Ax - theta * Ay + n * theta**2) / (Axx - 2*theta*Ax + n*theta**2)) / dt
  a = np.exp(-kappa * dt)
  sigmah2 = (Ayy - 2*a*Axy + a**2 * Axx - 2*theta*(1-a)*(Ay - a*Ax) + n*theta**2 * (1-a)**2) / n
  sigma = np.sqrt(sigmah2 * 2 * kappa / (1 - a**2))
  r0 = rates[n-1]
  return [kappa, theta, sigma, r0]


params = VasicekCalibration(trsy.iloc[:, 9])
kappa = params[0]
theta = params[1]
sigma = params[2]
r0 = params[3]
years = 1
N = years * 252
t = np.arange(0, N) / 252
test_sim = VasicekSim(N, r0, kappa, theta, sigma, 1/252)
plt.figure(figsize=(10,5))
plt.plot(t, test_sim, color='r')
plt.show()



# M projectories of the swap rates
M = 1000
rates_arr = VasicekMultiSim(M, N, r0, kappa, theta, sigma)
plt.figure(figsize=(10,5))
#avg  = np.average(rates_arr[:756,:])
#avgCurve = rates_arr[:]
plt.plot(t,rates_arr)
plt.hlines(y=theta, xmin = -100, xmax=100, zorder=10, linestyles = 'dashed', label='Theta')
plt.annotate('Theta', xy=(1.0, theta))
plt.xlim(-0.05, 1.05)
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.ylabel("Rate")
plt.xlabel("Time (year)")
plt.title('Long run (mean reversion level) treasury rate, theta = ' + str(round(theta, 2)) + '%')
plt.show()


std_dev = np.std(rates_arr[19,:])



