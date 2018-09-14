#
#............................Linear Regression using one variable..................................
#....................................Author: Mayank Mishra.........................................
#

#import requisites
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


#Calculating the variance from the mean for the vector par
def calc_mean_vector(par):
    mean_par = np.mean(par)
    return (par - mean_par)

#calculating the covarince of x and y 
#covariance is equal to the dot product of variance from the mean of vector x and y  divide by length of the sample
def covariance(x, y):
    n = len(x)
    return sum(calc_mean_vector(x) * calc_mean_vector(y)) / (n-1)


#calculating the variance which is equal to the average of the sum of sqaured variance from the mean
def variance(x):
   return sum(pow(calc_mean_vector(x), 2)) / len(x)


#calculating the slope and intercept for the regression line
#slope is eual to the correlation of two vector times standard deviation of y all divided by standard deviation of x
#which will be derived as covariance of two vector divided by the variance of independent variable
#and intercept is the difference of mean of y and the slope times the mean of x
def calc_slope_intercept(x, y):
    slope =  covariance(x, y) / variance(x)
    intercept = np.mean(y) - slope * np.mean(x)
    return slope,intercept


#Equation of line y = mx+c where y is dependent variable or target variable, 
#m is slope of the line x is the feature variable and c is the intercept from the y axis of the regreesion line 
def equation_line(intercept, slope, x):
    return intercept + (slope * x)


#rsquare value is the difference between 1 and (sum of squared variation in the predicted value and actual value divided by the variation in y)
#more closer to 1 more accuracy in our prediction
def rsquared(ypred, y):
    squared_error = sum(pow(ypred - y, 2))
    square_var = sum(pow(np.mean(y) - y, 2))

    rsq = 1.0 - (squared_error / square_var)
    return rsq
    


#Regression Test
if __name__ == '__main__':
    
    #numple of samples
    num_house = 200

    #1.Preparing our Data set
    np.random.seed(50)
    house_size = np.random.randint(low=1000, high=3000, size=num_house)

    np.random.seed(50)
    house_price = house_size * 50 + np.random.randint(low=1000, high=10000, size=num_house)

    #initial plot for them
    plt.plot(house_size,house_price,'x')
    plt.xlabel("House Size")
    plt.ylabel("House_Price")
    plt.show()


    #Calculating the coefficients
    slope, intercept = calc_slope_intercept(house_size, house_price)

    #making the predictions
    predict_price = equation_line(intercept, slope, house_size)

    #computing the r - squared value
    r_square = rsquared(predict_price, house_price)

    #Printing the parameter on the screen
    print("-----------------------------------------------------------------------------------------------------------")
    print("Value for the slope is {} and intercept is {}".format(slope,intercept))
    print("This method predicted with R square value : {}".format(r_square))
    print("-----------------------------------------------------------------------------------------------------------")
    print("Plotting the regression line")

    plt.scatter(house_size,house_price)
    plt.plot(house_size,predict_price, c='r')
    plt.xlabel("House Size")
    plt.ylabel("House_Price")
    plt.show()