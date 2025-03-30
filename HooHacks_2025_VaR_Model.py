# -*- coding: utf-8 -*-
"""
@author: MarkWright

https://sarit-maitra.medium.com/pca-monte-carlo-simulation-for-vasicek-interest-rate-model-9522858cc89d

https://towardsdatascience.com/random-walks-with-python-8420981bc4bc

https://stackoverflow.com/questions/33203645/how-to-plot-a-histogram-using-matplotlib-in-python-with-a-list-of-data

https://github.com/bickez/puppy-economics/blob/master/vasicek.R
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns


# Gets the next simulated range change
def VasicekNextRate(r, kappa, theta, sigma, dt=1/252):
  # Implements above closed form solution
  val1 = np.exp(-1*kappa*dt)
  val2 = (sigma**2)*(1-val1**2) / (2*kappa)
  out = r*val1 + theta*(1-val1) + (np.sqrt(val2))*np.random.normal()
  return out


# Vasicek simulation - combines a full path of simulated range changes
# daily changes the next 3 months
def VasicekSim(N, r0, kappa, theta, sigma, dt = 1/252):
  short_r = [0]*N # Create array to store rates
  short_r[0] = r0 # Initialize rates at $r_0$
  for i in range(1,N):
    short_r[i]=VasicekNextRate(short_r[i-1],kappa, theta, sigma, dt)
  return short_r


# Vasicek multi-simulation - runs a higher number of fully simulated rate paths
# Each path is a full 3 month simulated rate change
def VasicekMultiSim(M, N, r0, kappa, theta, sigma, dt = 1/252):
  sim_arr = np.ndarray((N, M))
  for i in range(0,M):
    sim_arr[:, i] = VasicekSim(N, r0, kappa, theta, sigma, dt)
  return sim_arr

# Pulls in treasury data to calibrate the parameters necessary for the Vasicek simulation
trsy = pd.read_excel(r'Treasury Curve Data.xlsx', sheet_name='Clean Data')


# Maximum Likelihood Estimation to calibrate the parameters
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


# Running a total of 1000 simulations (1000 daily treasury rate changes for a term of 3 months)
# 840 for 1 std dev, 975 for 2 std dev
confidence_index = [840, 975]

# Creates a data frame where each column is a point of the US Treasury curve
# 1 Month
# 3 Months
# 6 Months
# 1 Year
# 2 Year
# 3 Year
# 5 Year
# 7 Year
# 10 Year
# 20 Year
# 30 Year
column_headers = ['DGS1MO','DGS3MO','DGS6MO','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30']
simulated_trsy_column_headers = ['DGS1MO','DGS3MO','DGS6MO','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30','Month','Confidence']
simulated_trsy = pd.DataFrame(columns=simulated_trsy_column_headers)



# The model simulates each point of the US Treasury curve separately. 
# The inner for loop generates daily changes in each point of the curve for the next year 
# and stores the end of each month in its own container.
# The data is input into the model sequentially so the on

# For 1 std dev (67% confidence) and # 2 std dev (96% confidence)
for index in confidence_index:
    
    # Lists to hold each simulated value 1 month, 2 month, and 3 months out
    month_one_curve_list = []
    month_two_curve_list = []
    month_three_curve_list = []
        
    # For each point in the treasury curve
    for i in range(1, len(trsy.T)):
        
        # Set the parameters for the Vasicek Interest Rate Model
        params = VasicekCalibration(trsy.iloc[:, i])
        kappa = params[0]
        theta = params[1]
        sigma = params[2]
        r0 = params[3]
        years = 1
        N = years * 252
        t = np.arange(0, N) / 252
    
        # M simulations of the point in the treasury curve
        M = 1000
        rates_arr = VasicekMultiSim(M, N, r0, kappa, theta, sigma) # run the simulation and save to array
        
        
        month_one_sim = rates_arr[20,:] # simulated rates 1 month out - 20 business days for 1 month
        month_two_sim = rates_arr[40,:] # simulated rates 2 months out - 40 business days for 2 months
        month_three_sim= rates_arr[60,:] # simulated rates 3 months out - 60 business days for 3 months
        
        
        # sorts each days 1000 simulated rates smallest to largest
        month_one_sorted = np.sort(month_one_sim).tolist()
        month_two_sorted = np.sort(month_two_sim)
        month_three_sorted = np.sort(month_three_sim)
        
        # get the 840th rate for 1st std dev (67th percentile), or 975 for 2nd std dev (96th percentile)
        month_one_curve_list.append(month_one_sorted[index])
        month_two_curve_list.append(month_two_sorted[index])
        month_three_curve_list.append(month_three_sorted[index])
    
    
    # Append the worst case simulated rate one month out, at the 67th percentile or 96th percentile confidence level,
    # into a container data frame for month 1
    month_one_curve = pd.DataFrame(month_one_curve_list, index=column_headers).T
    month_one_curve['Month'] = 'Month One Curve'
    month_one_curve['Confidence'] = index

    # Append the worst case simulated rate one month out, at the 67th percentile or 96th percentile confidence level,
    # into a container data frame for month 2
    month_two_curve = pd.DataFrame(month_two_curve_list, index=column_headers).T
    month_two_curve['Month'] = 'Month Two Curve'
    month_two_curve['Confidence'] = index

    # Append the worst case simulated rate one month out, at the 67th percentile or 96th percentile confidence level,
    # into a container data frame for month 3
    month_three_curve = pd.DataFrame(month_three_curve_list, index=column_headers).T
    month_three_curve['Month'] = 'Month Three Curve'
    month_three_curve['Confidence'] = index

    # Append the simulated points of the curve into a master container data frame
    # the points of each curve are put into the model in ascending order so the output in this
    # container will also be in ascending order - the order we want
    simulated_trsy = simulated_trsy._append(month_one_curve)
    simulated_trsy = simulated_trsy._append(month_two_curve)
    simulated_trsy = simulated_trsy._append(month_three_curve)


simulated_trsy = simulated_trsy.reset_index(drop=True)

##################################################################

from scipy.interpolate import CubicSpline

# Uses cubic spline interpolation to turn the 11 point treasury curve into a 360 point treasury curve
# where there is a rate each month for the next 30 years

# Interpolates the current treasury curve
treasury_curve = pd.read_excel(r'Treasury Curve Data.xlsx', sheet_name = 'Current Curve')
f = CubicSpline(treasury_curve['Term'], treasury_curve['Rate'])
x = np.linspace(1, 360, 360)
y = f(x)
treasury_curve_interp = pd.DataFrame(y, columns=['Current Curve'])



# Interpolates each simulated treasury curve 
# 1MO, 2MO, and 3MO's out at the worst case 67th and 96th percentile confidence levels
terms = pd.Series(data=[1, 3, 6, 12, 24, 36, 60, 84, 120, 240, 360])
collected_curves = {}
for i in range(len(simulated_trsy)):
    
    simulated_curve = simulated_trsy.iloc[i, :11]
    
    f = CubicSpline(terms, simulated_curve)
    x = np.linspace(1, 360, 360)
    y = f(x)

    collected_curves[i] = y.tolist()


# A collection of all the final interpolated curves
final = pd.DataFrame(collected_curves)


# we want just the current curve, the worst case simulated curve at the 67th percentile 3 months out
# and the worst case simulated curve at the 96th percentile 3 months out
curves_for_plot = treasury_curve_interp.copy(deep=True)
curves_for_plot['Worst Case 67% CI'] = final.loc[:, 2]
curves_for_plot['Worst Case 96% CI'] = final.loc[:, 5]

# Plots the current treasury curve and the worst case simulated treasury curves 3 months out at the 
# 67th and 96th confidence levels
sns.lineplot(data=curves_for_plot, palette=['C0', '#f97306', 'r'], dashes=[(2, 0), (2, 2), (2, 2)])
plt.ylim(-0.05, 7.05)
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.xticks(np.arange(0, 361, 60))
plt.ylabel("Percent")
plt.xlabel("Term (months)")
plt.title('Simulated US Treasury Curve 3 Months from Now')
plt.show()



