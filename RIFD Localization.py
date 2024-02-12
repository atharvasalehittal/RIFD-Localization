#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install matplotlib


# In[3]:


import matplotlib.pyplot as plt

# Distance values
distances = [0, 0.5, 1, 1.5, 2]

# RSSI values
rssi_values = [-54.18421053, -54.075, -53.86363636, -54.29166667, -55.17241379]

# Plotting the graph
plt.plot(distances, rssi_values, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Distance (meters)')
plt.ylabel('RSSI')
plt.title('RSSI vs Distance')

# Display the graph
plt.show()


# In[9]:


import numpy as np
from scipy.stats import linregress

# Convert distances to logarithmic scale
log_distances = np.log10(distances)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(log_distances, rssi_values)

# Path loss exponent (n) is the absolute value of the slope
n = abs(slope)

# Print the result
print("Path Loss Exponent (n):", n)


# In[1]:


import math

# Given values
Pt = 17  # dBm
RSSI = -54.291  # dBm
f = 915000000.45  # Hz
c = 3e8  # speed of light in m/s
X = 1.5813

# Calculate distance using Friis transmission equation
d = 10**((Pt - RSSI - 20 * math.log10(f) + 20 * math.log10(c / (4 * math.pi)) + X) / 20)

# Print the result
print("Distance (meters):", d)


# In[1]:


import matplotlib.pyplot as plt

# Data
distance = [0.2, 0.4, 0.8, 1.2, 1.6, 2.0]
rssi = [-61.45512821, -62.1754386, -62.7612, -63.2, -63.38793103, -65.48148148]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(distance, rssi, marker='o', linestyle='-', color='b')

# Adding labels and title
plt.xlabel('Distance')
plt.ylabel('RSSI')
plt.title('RSSI vs. Distance')

# Show the plot
plt.grid(True)
plt.show()


# In[76]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Path loss equation
def path_loss(distance, f, n):
    return 20 * np.log10(f) + 10 * n * np.log10(distance) - 147.55

# Data
distance = np.array([0.2, 0.4, 0.8, 1.2, 1.6, 2.0])
rssi = np.array([-61.45512821, -62.1754386, -62.7612, -63.2, -63.38793103, -65.48148148])
frequency = np.array([909532051.3, 904592105.3, 907046610.2, 906480769.2, 907051724.1, 909037037.00])

# Initial guess for the path loss exponent
initial_guess = 2.0  # Initial guess for the path loss exponent (n)

# Perform the curve fit, keeping frequency fixed
params, covariance = curve_fit(lambda d, n: path_loss(d, frequency, n), distance, rssi, p0=initial_guess)

# Extract the estimated path loss exponent (n)
n_est = params[0]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(distance, rssi, marker='o', linestyle='-', color='b', label='Data')
plt.plot(distance, path_loss(distance, frequency, n_est), 'r-', label='Fit')

# Adding labels and title
plt.xlabel('Distance')
plt.ylabel('RSSI')
plt.title('RSSI vs. Distance with Path Loss Fit (Fixed Frequency)')

# Show the plot
plt.legend()
plt.grid(True)
plt.show()

# Print the estimated path loss exponent
print(f"Estimated Path Loss Exponent (n): {n_est}")


# In[28]:


import math

# Given values
Pt = 29  # dBm
RSSI = -63.2 # dBm
f = 0.01612799  # Hz
c = 3e8  # speed of light in m/s

# Path Loss Model Parameters
n = -0.3165239783  # You need to specify the path loss exponent (adjust as needed)

# Calculate distance using the modified path loss model equation
d = 10**((Pt - RSSI + 27.55 - 20 * math.log10(f)) / (10 * n))

# Print the result
print("Distance (meters):", d)


# In[27]:


import numpy as np
from scipy.optimize import curve_fit

# Given data
distance = np.array([0.2, 0.4, 0.8, 1.2, 1.6, 2.0])
rssi = np.array([-61.45512821, -62.1754386, -62.7612, -63.2, -63.38793103, -65.48148148])
frequency = np.array([909532051.3, 904592105.3, 907046610.2, 906480769.2, 907051724.1, 909037037.0])

# Path loss equation
def path_loss(x, n):
    distance, frequency = x
    return 20 * np.log10(frequency) + 10 * n * np.log10(distance) - 27.55

# Initial guess for the path loss exponent (n)
initial_guess = 2.0

# Perform the non-linear regression
params, covariance = curve_fit(path_loss, (distance, frequency), rssi, p0=initial_guess)

# Extract the optimized path loss exponent (n)
optimized_n = params[0]

# Print the result
print("Optimized Path Loss Exponent (n):", optimized_n)


# In[49]:


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Given data
distance = np.array([0.2, 0.4, 0.8, 1.2, 1.6, 2.0])
rssi = np.array([-61.45512821, -62.1754386, -62.7612, -63.2, -63.38793103, -65.48148148])
frequency = np.array([909.5320513, 904.5921053, 907.0466102, 906.4807692, 907.0517241, 909.0370370])

# Path loss equation
def path_loss(x, n):
    distance, frequency = x
    return 20 * np.log10(frequency) + 10 * n * np.log10(distance) - 27.55

# Combine distance and frequency into a single array
combined_data = np.vstack((distance, frequency))

# Initial guess for the path loss exponent (n)
initial_guess = 2.0

# Perform the non-linear regression
params, covariance = curve_fit(path_loss, combined_data, rssi, p0=initial_guess)

# Extract the optimized path loss exponent (n)
optimized_n = params[0]

# Print the result
print("Optimized Path Loss Exponent (n):", optimized_n)

# Plotting the data and fitted curve
plt.scatter(distance, rssi, label='Original Data')
plt.xlabel('Distance (m)')
plt.ylabel('RSSI')
plt.title('Path Loss Regression')
plt.grid(True)

# Generate points for the fitted curve
fit_curve = path_loss(combined_data, optimized_n)

# Plot the fitted curve
plt.plot(distance, fit_curve, label=f'Fitted Curve (n={optimized_n:.2f})', color='red')
plt.legend()
plt.show()


# In[45]:


import numpy as np

# Given values
L = -63.2    # Replace with your RSSI value
f = 906480769.2  # Replace with your frequency value
n =  16.17  # Replace with your optimized path loss exponent value

# Calculate distance (d)
d = 10 ** ((L + 27.55 - 20 * np.log10(f)) / (10 * n))

# Print the result
print("Calculated Distance (d):", d)


# In[51]:


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Provided data
distance = np.array([0.2, 0.4, 0.8, 1.2, 1.6, 2.0])
rssi = np.array([-61.45512821, -62.1754386, -62.7612, -63.2, -63.38793103, -65.48148148])
frequency = np.array([909.5320513, 904.5921053, 907.0466102, 906.4807692, 907.0517241, 909.037037])

# Define the path loss equation
def path_loss_model(distance, frequency, n):
    return 20 * np.log10(frequency) + 10 * n * np.log10(distance) - 27.55

# Use curve_fit to find the optimal parameters (including n)
params, covariance = curve_fit(lambda d, n: path_loss_model(d, frequency, n), distance, rssi, p0=(2.0))

# Extract the path loss exponent (n) from the optimized parameters
n_optimal = params[0]

# Print the result
print("Optimal path loss exponent (n):", n_optimal)

# Plot the original data and the fitted curve
plt.scatter(distance, rssi, label='Original Data')
plt.plot(distance, path_loss_model(distance, frequency, n_optimal), 'r', label='Fitted Curve')
plt.xlabel('Distance (m)')
plt.ylabel('RSSI')
plt.legend()
plt.show()


# In[55]:


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Provided data
distance = np.array([0.2, 0.4, 0.8, 1.2, 1.6, 2.0])
rssi = np.array([-61.45512821, -62.1754386, -62.7612, -63.2, -63.38793103, -65.48148148])
frequency = np.array([909532051.3, 904592105.3, 907046610.2, 906480769.2, 907051724.1, 909037037.00])

# Define the path loss equation
def path_loss_model(distance, n):
    return 20 * np.log10(frequency) + 10 * n * np.log10(distance) - 147.55

# Use curve_fit to find the optimal parameters (including n)
params, covariance = curve_fit(path_loss_model, distance, rssi, p0=(2.0))

# Extract the path loss exponent (n) from the optimized parameters
n_optimal = params[0]

# Print the result
print("Optimal path loss exponent (n):", n_optimal)

# Plot the original data and the fitted curve
plt.scatter(distance, rssi, label='Original Data')
plt.plot(distance, path_loss_model(distance, n_optimal), 'r', label='Fitted Curve')
plt.xlabel('Distance (m)')
plt.ylabel('RSSI')
plt.legend()
plt.show()


# In[57]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Provided data
distance = np.array([0.2, 0.4, 0.8, 1.2, 1.6, 2.0])
rssi = np.array([-61.45512821, -62.1754386, -62.7612, -63.2, -63.38793103, -65.48148148])

# Define the path loss equation
def path_loss_model(distance, n):
    return -61.45512821 - 10 * n * np.log10(distance)

# Use curve_fit to find the optimal parameters (including n)
params, covariance = curve_fit(path_loss_model, distance, rssi, p0=(2.0))

# Extract the path loss exponent (n) from the optimized parameters
n_optimal = params[0]

# Plot the original data and the fitted curve
plt.scatter(distance, rssi, label='Original Data')
plt.plot(distance, path_loss_model(distance, n_optimal), 'r', label='Fitted Curve')
plt.xlabel('Distance (m)')
plt.ylabel('RSSI')
plt.legend()
plt.show()

# Print the result
print("Optimal path loss exponent (n):", n_optimal)


# In[67]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Provided data
distance = np.array([0.2, 0.4, 0.8, 1.2, 1.6, 2.0])
rssi = np.array([-61.45512821, -62.1754386, -62.7612, -63.2, -63.38793103, -65.48148148])

# Define the path loss equation
def path_loss_model(distance, frequency, n):
    return 20 * np.log10(frequency) + 10 * n * np.log10(distance) - 147.55

# Use curve_fit to find the optimal parameters (including n)
params, covariance = curve_fit(lambda d, n: path_loss_model(d,907290049.5, n), distance, rssi, p0=(2.0))

# Extract the path loss exponent (n) from the optimized parameters
n_optimal = params[0]

# Plot the original data and the fitted curve
plt.scatter(distance, rssi, label='Original Data')
plt.plot(distance, path_loss_model(distance, 906480769.2, n_optimal), 'r', label='Fitted Curve')
plt.xlabel('Distance (m)')
plt.ylabel('RSSI')
plt.legend()
plt.show()

# Print the result
print("Optimal path loss exponent (n):", n_optimal)


# In[ ]:


############Correct Code (Reader-1) #############


# In[1]:


import matplotlib.pyplot as plt

# Readings for left foot
left_foot_readings = [-56, -56.5, -57, -57.5, -58.5, -59, -59.5, -60, -65.5, -64, -66]

# Readings for right foot
right_foot_readings = [-58.5, -59, -59.5, -60.5, -61, -61.5, -62, -63, -63.5, -64.5]

# Readings for third foot
readings_third = [-63, -63.5, -62.5, -65.5, -64, -65, -65.5, -63.5, -66, -67.5]

#Readingfs for 4th foot
readings_fourth = [-67, -67.5,  -65, -65.5]

#Readingfs for 5th foot
readings_fifth = [-68.5, -65, -71]

# Generate x-axis values for each reading (assuming they are at the same point)
x_left = [1] * len(left_foot_readings)
x_right = [2] * len(right_foot_readings)
x_third = [3] * len(readings_third)
x_fourth = [4] * len(readings_fourth)
x_fifth = [5] * len(readings_fifth)

# Plot a scatter plot for the left foot
plt.scatter(x_left, left_foot_readings, marker='o', color='b', label='Left Foot')

# Plot a scatter plot for the right foot
plt.scatter(x_right, right_foot_readings, marker='o', color='r', label='Right Foot')

# Plot a scatter for the third foot
plt.scatter(x_third, readings_third, marker='o', color='g', label='Third Foot')

# Plot a scatter for the fourth foot
plt.scatter(x_fourth, readings_fourth, marker='o', color='y', label='Fourth Foot')

#PLot a scatter for fifth foot
plt.scatter(x_fifth, readings_fifth, marker='o', color='c', label='Fifth Foot')

# Add labels and title
plt.xlabel('Foot')
plt.ylabel('Reading Value')
plt.title('Multiple Readings for Each Foot')

# Set y-axis limits for better visibility
plt.ylim(bottom=-70, top=-50)  # Adjust the values based on your data range

# Add a legend to differentiate between left, right, and third foot
plt.legend()

# Display the graph
plt.show()


# In[26]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to fit (sigmoid function as an example)
def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

# Combine all data points
all_x = np.concatenate([x_left, x_right, x_third, x_fourth, x_fifth])
all_y = np.concatenate([left_foot_readings, right_foot_readings, readings_third, readings_fourth, readings_fifth])

# Initial guess for parameters
initial_guess = [np.mean(all_y), 1, np.mean(all_x)]

# Perform the curve fit with a different optimization method (lm method)
params, covariance = curve_fit(sigmoid, all_x, all_y, p0=initial_guess, method='lm', maxfev=10000)

# Generate points for the fitted curve
x_fit = np.linspace(0.5, 5.5, 500)
y_fit = sigmoid(x_fit, *params)

# Plot the scatter plot
plt.scatter(all_x, all_y, marker='o', color='gray', label='Original Data')

# Plot the fitted curve
plt.plot(x_fit, y_fit, color='black', label='Fitted Curve (Sigmoid)')

# Add labels and title
plt.xlabel('Foot')
plt.ylabel('Reading Value')
plt.title('Non-linear Curve Fit for Multiple Readings')

# Set axis limits for better visibility
plt.xlim(0.5, 5.5)
plt.ylim(bottom=-72, top=-48)

# Add a legend
plt.legend()

# Display the graph
plt.show()


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to fit (sigmoid function as an example)
def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

# Function to compute the inverse of the fitted sigmoid function
def inverse_sigmoid(y, a, b, c):
    return c - (1 / b) * np.log(a / y - 1)

# Readings for left foot
left_foot_readings = [-56, -56.5, -57, -57.5, -58.5, -59, -59.5, -60, -65.5, -64, -66]

# Readings for right foot
right_foot_readings = [-58.5, -59, -59.5, -60.5, -61, -61.5, -62, -63, -63.5, -64.5]

# Readings for third foot
readings_third = [-63, -63.5, -62.5, -65.5, -64, -65, -65.5, -63.5, -66, -67.5]

#Readingfs for 4th foot
readings_fourth = [-67, -67.5,  -65, -65.5]

#Readingfs for 5th foot
readings_fifth = [-68.5, -65, -71]

# Generate x-axis values for each reading (assuming they are at the same point)
x_left = [1] * len(left_foot_readings)
x_right = [2] * len(right_foot_readings)
x_third = [3] * len(readings_third)
x_fourth = [4] * len(readings_fourth)
x_fifth = [5] * len(readings_fifth)

all_x = np.concatenate([x_left, x_right, x_third, x_fourth, x_fifth])
all_y = np.concatenate([left_foot_readings, right_foot_readings, readings_third, readings_fourth, readings_fifth])

# Initial guess for parameters
initial_guess = [np.mean(all_y), 1, np.mean(all_x)]

params, covariance = curve_fit(sigmoid, all_x, all_y, p0=initial_guess, method='lm', maxfev=10000)
                                
# Assume you have the following RSSI value
given_rssi = -64.3636   ###For 1st expriement

# Estimate the distance for the given RSSI value
estimated_distance = inverse_sigmoid(given_rssi, *params)

print(f"Estimated Distance for RSSI {given_rssi}: {estimated_distance:.2f} feet")


# In[ ]:


######## Correct Code (Reader-2) ##############


# In[8]:


import matplotlib.pyplot as plt

# Readings for left foot
left_foot_readings = [-54, -56.5, -57, -62.5, -63, -64, -64.5, -65.5, -66, -67.5]

# Readings for right foot
right_foot_readings = [-58, -59, -59.5, -60, -61, -63.5, -64, -64.5, -65, -65.5, -66]

# Readings for third foot
readings_third = [-58.5, -59, -60, -61, -61.5, -62.5, -63, -63.5, -64, -64.5, -66]

#Readingfs for 4th foot
readings_fourth = [-61.5, -62.5, -63.5, -64, -64.5, -65, -65.5, -66, -68, -67.5]

#Readingfs for 5th foot
readings_fifth = [-61.5, -62, -62.5, -63.5, -64.5, -65, -65.5, -66, -67, -67.5, -69.5]

# Generate x-axis values for each reading (assuming they are at the same point)
x_left = [1] * len(left_foot_readings)
x_right = [2] * len(right_foot_readings)
x_third = [3] * len(readings_third)
x_fourth = [4] * len(readings_fourth)
x_fifth = [5] * len(readings_fifth)

# Plot a scatter plot for the left foot
plt.scatter(x_left, left_foot_readings, marker='o', color='b', label='Left Foot')

# Plot a scatter plot for the right foot
plt.scatter(x_right, right_foot_readings, marker='o', color='r', label='Right Foot')

# Plot a scatter for the third foot
plt.scatter(x_third, readings_third, marker='o', color='g', label='Third Foot')

# Plot a scatter for the fourth foot
plt.scatter(x_fourth, readings_fourth, marker='o', color='y', label='Fourth Foot')

#PLot a scatter for fifth foot
plt.scatter(x_fifth, readings_fifth, marker='o', color='c', label='Fifth Foot')

# Add labels and title
plt.xlabel('Foot')
plt.ylabel('Reading Value')
plt.title('Multiple Readings for Each Foot')

# Set y-axis limits for better visibility
plt.ylim(bottom=-70, top=-50)  # Adjust the values based on your data range

# Add a legend to differentiate between left, right, and third foot
plt.legend()

# Display the graph
plt.show()


# In[42]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to fit (sigmoid function as an example)
def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

# Readings for left foot
left_foot_readings = [-54, -56.5, -57, -62.5, -63, -64, -64.5, -65.5, -66, -67.5]

# Readings for right foot
right_foot_readings = [-58, -59, -59.5, -60, -61, -63.5, -64, -64.5, -65, -65.5, -66]

# Readings for third foot
readings_third = [-58.5, -59, -60, -61, -61.5, -62.5, -63, -63.5, -64, -64.5, -66]

#Readingfs for 4th foot
readings_fourth = [-61.5, -62.5, -63.5, -64, -64.5, -65, -65.5, -66, -68, -67.5]

#Readingfs for 5th foot
readings_fifth = [-61.5, -62, -62.5, -63.5, -64.5, -65, -65.5, -66, -67, -67.5, -69.5]

# Generate x-axis values for each reading (assuming they are at the same point)
x_left = [1] * len(left_foot_readings)
x_right = [2] * len(right_foot_readings)
x_third = [3] * len(readings_third)
x_fourth = [4] * len(readings_fourth)
x_fifth = [5] * len(readings_fifth)

# Combine all data points
all_x = np.concatenate([x_left, x_right, x_third, x_fourth, x_fifth])
all_y = np.concatenate([left_foot_readings, right_foot_readings, readings_third, readings_fourth, readings_fifth])

# Initial guess for parameters
initial_guess = [np.mean(all_y), 1, np.mean(all_x)]

# Perform the curve fit with a different optimization method (lm method)
params, covariance = curve_fit(sigmoid, all_x, all_y, p0=initial_guess, method='lm', maxfev=10000)

# Generate points for the fitted curve
x_fit = np.linspace(0.5, 5.5, 500)
y_fit = sigmoid(x_fit, *params)

# Plot the scatter plot
plt.scatter(all_x, all_y, marker='o', color='gray', label='Original Data')

# Plot the fitted curve
plt.plot(x_fit, y_fit, color='black', label='Fitted Curve (Sigmoid)')

# Add labels and title
plt.xlabel('Foot')
plt.ylabel('Reading Value')
plt.title('Non-linear Curve Fit for Multiple Readings')

# Set axis limits for better visibility
plt.xlim(0.5, 5.5)
plt.ylim(bottom=-70, top=-50)

# Add a legend
plt.legend()

# Display the graph
plt.show()


# In[10]:


import numpy as np
from scipy.optimize import curve_fit

# Function to fit (sigmoid function as an example)
def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

def inverse_sigmoid(y, a, b, c):
    return c - (1 / b) * np.log(a / y - 1)

# Readings for left foot
left_foot_readings = [-54, -56.5, -57, -62.5, -63, -64, -64.5, -65.5, -66, -67.5]

# Readings for right foot
right_foot_readings = [-58, -59, -59.5, -60, -61, -63.5, -64, -64.5, -65, -65.5, -66]

# Readings for third foot
readings_third = [-58.5, -59, -60, -61, -61.5, -62.5, -63, -63.5, -64, -64.5, -66]

#Readingfs for 4th foot
readings_fourth = [-61.5, -62.5, -63.5, -64, -64.5, -65, -65.5, -66, -68, -67.5]

#Readingfs for 5th foot
readings_fifth = [-61.5, -62, -62.5, -63.5, -64.5, -65, -65.5, -66, -67, -67.5, -69.5]

# Generate x-axis values for each reading (assuming they are at the same point)
x_left = [1] * len(left_foot_readings)
x_right = [2] * len(right_foot_readings)
x_third = [3] * len(readings_third)
x_fourth = [4] * len(readings_fourth)
x_fifth = [5] * len(readings_fifth)

# Combine all data points
all_x = np.concatenate([x_left, x_right, x_third, x_fourth, x_fifth])
all_y = np.concatenate([left_foot_readings, right_foot_readings, readings_third, readings_fourth, readings_fifth])

# Initial guess for parameters
initial_guess = [np.mean(all_y), 1, np.mean(all_x)]

# Perform the curve fit with a different optimization method (lm method)
params, covariance = curve_fit(sigmoid, all_x, all_y, p0=initial_guess, method='lm', maxfev=10000)

given_rssi = -64.29   ### Reading for 2nd Reader
# Use the inverse sigmoid function to estimate the distance
estimated_distance = inverse_sigmoid(given_rssi, *params)

print(f"Estimated Distance for RSSI {given_rssi}: {estimated_distance:.2f} feet")


# In[ ]:


##### Correct Code (Reader-3) #######


# In[8]:


import matplotlib.pyplot as plt

# Readings for left foot
left_foot_readings = [-54.5, -55, -55.5, -56, -56.5, -57, -58 -59, -59.5, -60, -60.5, -61, -61.5]

# Readings for right foot
right_foot_readings = [-56, -57, -57.5, -58, -59, -59.5, -60, -60.5, -61, -61.5, -62, -63, -64]

# Readings for third foot
readings_third = [-59, -59.5, -60, -60.5, -61, -61.5, -62, -62.5, -63.5, -64, -65, -66.5, -67, -68, -68.5]

#Readingfs for 4th foot
readings_fourth = [-58, -60, -60.5, -62.5, -63, -63.5, -64, -65, -60, -67, -67.5, -68, -68.5, -69, -69.5]

#Readingfs for 5th foot
readings_fifth = [-62, -66, -66.5, -67, -67.5, -68, -68.5, -69]

# Generate x-axis values for each reading (assuming they are at the same point)
x_left = [1] * len(left_foot_readings)
x_right = [2] * len(right_foot_readings)
x_third = [3] * len(readings_third)
x_fourth = [4] * len(readings_fourth)
x_fifth = [5] * len(readings_fifth)

# Plot a scatter plot for the left foot
plt.scatter(x_left, left_foot_readings, marker='o', color='b', label='Left Foot')

# Plot a scatter plot for the right foot
plt.scatter(x_right, right_foot_readings, marker='o', color='r', label='Right Foot')

# Plot a scatter for the third foot
plt.scatter(x_third, readings_third, marker='o', color='g', label='Third Foot')

# Plot a scatter for the fourth foot
plt.scatter(x_fourth, readings_fourth, marker='o', color='y', label='Fourth Foot')

#PLot a scatter for fifth foot
plt.scatter(x_fifth, readings_fifth, marker='o', color='c', label='Fifth Foot')

# Add labels and title
plt.xlabel('Foot')
plt.ylabel('Reading Value')
plt.title('Multiple Readings for Each Foot')

# Set y-axis limits for better visibility
plt.ylim(bottom=-70, top=-50)  # Adjust the values based on your data range

# Add a legend to differentiate between left, right, and third foot
plt.legend()

# Display the graph
plt.show()


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to fit (sigmoid function as an example)
def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

# Readings for left foot
left_foot_readings = [-54.5, -55, -55.5, -56, -56.5, -57, -58 -59, -59.5, -60, -60.5, -61, -61.5]

# Readings for right foot
right_foot_readings = [-56, -57, -57.5, -58, -59, -59.5, -60, -60.5, -61, -61.5, -62, -63, -64]

# Readings for third foot
readings_third = [-59, -59.5, -60, -60.5, -61, -61.5, -62, -62.5, -63.5, -64, -65, -66.5, -67, -68, -68.5]

#Readingfs for 4th foot
readings_fourth = [-58, -60, -60.5, -62.5, -63, -63.5, -64, -65, -60, -67, -67.5, -68, -68.5, -69, -69.5]

#Readingfs for 5th foot
readings_fifth = [-62, -66, -66.5, -67, -67.5, -68, -68.5, -69]

# Generate x-axis values for each reading (assuming they are at the same point)
x_left = [1] * len(left_foot_readings)
x_right = [2] * len(right_foot_readings)
x_third = [3] * len(readings_third)
x_fourth = [4] * len(readings_fourth)
x_fifth = [5] * len(readings_fifth)

# Combine all data points
all_x = np.concatenate([x_left, x_right, x_third, x_fourth, x_fifth])
all_y = np.concatenate([left_foot_readings, right_foot_readings, readings_third, readings_fourth, readings_fifth])

# Initial guess for parameters
initial_guess = [np.mean(all_y), 1, np.mean(all_x)]

# Perform the curve fit with a sigmoid function
params, covariance = curve_fit(sigmoid, all_x, all_y, p0=initial_guess, maxfev=10000)

# Generate points for the fitted curve
x_fit = np.linspace(0.5, 5.5, 500)
y_fit = sigmoid(x_fit, *params)

# Plot the scatter plot
plt.scatter(all_x, all_y, marker='o', color='gray', label='Original Data')

# Plot the fitted curve
plt.plot(x_fit, y_fit, color='black', label='Fitted Curve (Sigmoid)')

# Add labels and title
plt.xlabel('Foot')
plt.ylabel('Reading Value')
plt.title('Non-linear Curve Fit for Multiple Readings')

# Set axis limits for better visibility
plt.xlim(0.5, 5.5)
plt.ylim(bottom=-72, top=-48)

# Add a legend
plt.legend()

# Display the graph
plt.show()


# In[11]:


import numpy as np
from scipy.optimize import curve_fit

# Function to fit (sigmoid function as an example)
def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

# Inverse of the sigmoid function (logistic function)
def inverse_sigmoid(y, a, b, c):
    return -np.log((a / y) - 1) / b + c

# Readings for left foot
left_foot_readings = [-54.5, -55, -55.5, -56, -56.5, -57, -58 -59, -59.5, -60, -60.5, -61, -61.5]

# Readings for right foot
right_foot_readings = [-56, -57, -57.5, -58, -59, -59.5, -60, -60.5, -61, -61.5, -62, -63, -64]

# Readings for third foot
readings_third = [-59, -59.5, -60, -60.5, -61, -61.5, -62, -62.5, -63.5, -64, -65, -66.5, -67, -68, -68.5]

#Readingfs for 4th foot
readings_fourth = [-58, -60, -60.5, -62.5, -63, -63.5, -64, -65, -60, -67, -67.5, -68, -68.5, -69, -69.5]

#Readingfs for 5th foot
readings_fifth = [-62, -66, -66.5, -67, -67.5, -68, -68.5, -69]

# Generate x-axis values for each reading (assuming they are at the same point)
x_left = [1] * len(left_foot_readings)
x_right = [2] * len(right_foot_readings)
x_third = [3] * len(readings_third)
x_fourth = [4] * len(readings_fourth)
x_fifth = [5] * len(readings_fifth)

all_x = np.concatenate([x_left, x_right, x_third, x_fourth, x_fifth])
all_y = np.concatenate([left_foot_readings, right_foot_readings, readings_third, readings_fourth, readings_fifth])

# Initial guess for parameters
initial_guess = [np.mean(all_y), 1, np.mean(all_x)]

# Perform the curve fit with a sigmoid function
params, covariance = curve_fit(sigmoid, all_x, all_y, p0=initial_guess, maxfev=10000)

example_rssi = -63.19
# Estimate distance using the inverse sigmoid function
estimated_distance = inverse_sigmoid(example_rssi, *params)

print(f"Estimated Distance for RSSI {example_rssi}: {estimated_distance} units")


# In[ ]:


#### Trilateration


# In[13]:


#Cell Phone Trilateration Algorithm - www.101computing.net/cell-phone-trilateration-algorithm/
#import draw

#A function to apply trilateration formulas to return the (x,y) intersection point of three circles
def trackPhone(x1,y1,r1,x2,y2,r2,x3,y3,r3):
  A = 2*x2 - 2*x1
  B = 2*y2 - 2*y1
  C = r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2
  D = 2*x3 - 2*x2
  E = 2*y3 - 2*y2
  F = r2**2 - r3**2 - x2**2 + x3**2 - y2**2 + y3**2
  x = (C*E - F*B) / (E*A - B*D)
  y = (C*D - A*F) / (B*D - A*E)
  return x,y

#Generate and represent data to be used by the trilateration algorithm
#x1,y1,r1,x2,y2,r2,x3,y3,r3 = draw.drawCellTowers()

#Apply trilateration algorithm to locate phone
x,y = trackPhone(0,6.41,3.11,10.33,6.37,4.27,4.75,0,2.907)

#Output phone location / coordinates
print("Cell Phone Location:")
print(x,y)


# In[ ]:




