#! /usr/bin/env python3
import matplotlib.pyplot as plt


radius = [(1000209 - 5.0) / 1000209, (1000209 - 10.0) / 1000209, (1000209 - 20.0) / 1000209,
          (1000209 - 50.0) / 1000209, (1000209 - 100.0) / 1000209]

cf = [1.3399, 1.2867, 1.1130, 1.0798, 1.0552]

mllib = [0.7841, 1.0228, 0.8845, 0.8000, 0.9282]

mahout = [1.0742, 1.2656, 1.0882, 0.9685, 1.0546]

plt.plot(radius, cf, marker='o', color='r', label='Optimised CF')
plt.plot(radius, mahout, marker='o', color='g', label='Mahout')
plt.plot(radius, mllib, marker='o', color='b', label='MLlib')
plt.xlabel('Training set/(Test set + Training set)')
plt.ylabel('RMSE')
plt.title('RMSE of Recommendation Solution')
plt.legend()
plt.show()