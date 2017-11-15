#
#............................Linear Regression using Gradient Descent Algorithm ...................
#....................................Author: Mayank Mishra.........................................
#

#import requisites
import numpy as np
import matplotlib.pyplot as plt
import math


#Equation of line y = mx+c where y is dependent variable or target variable, 
#m is slope of the line x is the feature variable and c is the intercept from the y axis of the regreesion line 
def hypothesis_func(intercept, slope, x):
    return intercept + slope * x


#rsquare value is the difference between 1 and (sum of squared variation in the predicted value and actual value divided by the variation in y)
#more closer to 1 more accuracy in our prediction
def rsquared(ypred, y):
    squared_error = sum(pow(ypred - y, 2))
    square_var = sum(pow(np.mean(y) - y, 2))

    rsq = 1.0 - (squared_error / square_var)
    return rsq


#cost functionhalf od average of sum of variance from our hypothesis    
def cost_function(x, y, theta0, theta1, m):
    J = sum((hypothesis_func(theta0, theta1, x) - y) ** 2) /2/m
    return J


#our gradient descent algorithm
def gradientDescent(x, y, m, alpha, theta0, theta1, iteration):
    itr = 0
    while itr < iteration:
        hypothesis = hypothesis_func(theta0, theta1, x)
        loss = hypothesis - y
        gradient0 = sum(loss)/m 
        gradient1 = sum(loss*x)/m
        temp0 = theta0 - alpha * gradient0
        temp1 = theta1 - alpha * gradient1
        if temp0 == theta0 and temp1 == theta1:
            break;
        else:
            theta0 = temp0
            theta1 = temp1
        itr += 1
    return theta0, theta1


#need to normalise the data to prevent the underflow
def normalize(array):
    return (array - array.mean()) / array.std()


#Regression Test
if __name__ == '__main__':
    
    #numple of samples
    num_house = 200

    #Preparing our Data set
    np.random.seed(50)
    house_size = np.random.randint(low=1000, high=3000, size=num_house)

    np.random.seed(50)
    house_price = house_size * 50 + np.random.randint(low=1000, high=10000, size=num_house)

    #initial plot for them
    plt.plot(house_size,house_price,'x')
    plt.xlabel("House Size")
    plt.ylabel("House_Price")
    plt.show()

    #test-train strategy
    #define number of training sample = 0.7 = 70%. We can take the first 70% as the data are generated randomly.
    nsample = math.floor(num_house * 0.7)

    #training data
    train_house_size = np.asarray(house_size[:nsample])
    train_house_price = np.asanyarray(house_price[:nsample])

    train_house_size_norm = normalize(train_house_size)
    train_house_price_norm = normalize(train_house_price)

    #test data
    test_house_size = np.array(house_size[nsample:])
    test_house_price = np.array(house_price[nsample:])

    test_house_size_norm = normalize(test_house_size)
    test_house_price_norm = normalize(test_house_price)


    #defining ncessary paramenters such as learning rate, initial value for slope and intercept
    learning_rate = 0.01
    slope = 0
    intercept = 0
    iteration = 1500

    #running gradient descent algorithm to get the value for slope and intercept
    intercept, slope = gradientDescent(train_house_size_norm, train_house_price_norm, nsample, learning_rate, intercept, slope, iteration)

    #computing the predicted price from the learning and calculating the r-square value from it for training data
    train_predicted_prices_norm = hypothesis_func(intercept, slope, train_house_size_norm)
    train_rsa = rsquared(train_predicted_prices_norm, train_house_price_norm)

    #computing the predicted price from the learning and calculating the r-square value from it for test data
    test_predicted_prices_norm = hypothesis_func(intercept, slope, test_house_size_norm)
    test_rsa = rsquared(test_predicted_prices_norm, test_house_price_norm)

    #some basic operation for making plot for our normalized data
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_house_price.mean()
    train_price_std = train_house_price.std()

    #calculating paramaters for the line of regression
    regress_line_x = train_house_size_norm * train_house_size_std + train_house_size_mean
    regress_line_y = train_predicted_prices_norm * train_price_std + train_price_mean

    #printing the value for rsa's and our coefficients
    print("--------------------------------------------------------------------------------------------------------------------------")
    print("Coefficients values are: \nSlope : {} \nIntercept : {}".format(slope, intercept))
    print("The R - squared value for the training data set : {}".format(train_rsa))
    print("The R - squared value for the testing data set : {}".format(test_rsa))
    print("--------------------------------------------------------------------------------------------------------------------------")


    #Kudos let's plot our curve
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size,train_house_price,"go",label = "Training data")
    plt.plot(test_house_size,test_house_price,"mo",label = "Testing data")
    plt.plot(regress_line_x, regress_line_y, label = "Learned Regression")
    plt.legend(loc = "upper left")
    plt.show()