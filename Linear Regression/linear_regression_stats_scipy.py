import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def predict(x, intercept, slope):
    return slope * x + intercept

if __name__ == '__main__':
    num_house = 200
    np.random.seed(50)

    house_size = np.random.randint(low=1000, high=3000, size=num_house)

    np.random.seed(50)
    house_price = house_size*50+np.random.randint(low=1000, high=10000, size=num_house)


    plt.plot(house_size,house_price,'x')
    plt.xlabel("House Size")
    plt.ylabel("House_Price")
    plt.show()

    slope, intercept, r_value, p_value, std_err = stats.linregress(house_size, house_price)
    print(r_value ** 2)

    line = predict(house_size, intercept, slope)
    plt.scatter(house_size,house_price)
    plt.plot(house_size, line, c='r')
    plt.xlabel("House Size")
    plt.ylabel("House_Price")
    plt.show()


