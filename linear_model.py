import os
import numpy as np
import matplotlib.pyplot as plt


def data_generation():
    data = []
    for i in range(1000):
        x = np.random.uniform(-10, 10)
        eps = np.random.normal(0, 0.1)

        y = 1.477 * x + 0.089 + eps

        data.append([x, y])

    data = np.array(data)
    return data

def mse(b, w, points):

    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        totalError += (y - (w * x - b))**2

    totalError /= len(points)
    return totalError

def step_gradient(w_current, b_current, points, lr):
    
    w_gradient = 0
    b_gradient = 0

    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        
        w_gradient += (2/N) * x * ((w_current * x + b_current) -y)
        b_gradient += (2/N) * ((w_current * x + b_current) - y)

    new_w = w_current - lr * w_gradient 
    new_b = b_current - lr * b_gradient

    return [new_w, new_b]

def gradient_desent(points, init_w, init_b, lr, num_iteration):

    w = init_w
    b = init_b
    losses = []

    for step in range(num_iteration):

        w, b = step_gradient(w, b, points, lr)

        loss = mse(b, w, points)

        if step % 25 == 0:
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    
        losses += [loss]

    return [w, b], losses

def main():

    sample_data = data_generation()

    lr = 0.001
    
    init_w = 0.0
    init_b = 0.0

    num_iteration = 1000

    [w, b], losses = gradient_desent(sample_data, init_w, init_b, lr, num_iteration)

    final_loss = mse(b, w, sample_data)

    plt.plot(losses)
    plt.show()
    print(f"final loss: {final_loss}, w:{w}, b:{b}")


if __name__ == "__main__":
    main()




