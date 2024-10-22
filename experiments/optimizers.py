import numpy as np

# y = 3 * x^2 + 8


def sgd(X, Y, theta, epoch, batch_size, lr):
    for e in range(epoch):
        loss = 0
        delta_theta0 = 0
        delta_theta1 = 0
        for i in range(len(X)):
            loss += (Y[i] - theta[0] * X[i] ** 2 - theta[1]) ** 2
            delta_theta0 += -(
                lr * 2 * (Y[i] - theta[0] * X[i] ** 2 - theta[1]) * X[i] ** 2
            )
            delta_theta1 += lr * 2 * (Y[i] - theta[0] * X[i] ** 2 - theta[1])
            if (i + 1) % batch_size == 0:
                theta[0] -= delta_theta0 / batch_size
                theta[1] -= delta_theta1 / batch_size
                delta_theta0 = 0
                delta_theta1 = 0
        print("loss:", loss / len(X))


def momentum(X, Y, theta, epoch, batch_size, lr, beta=0.9):
    v0, v1 = 0, 0
    for _ in range(epoch):
        for b in range(len(X) // batch_size):
            loss = np.mean(
                (
                    Y[batch_size * b : batch_size * (b + 1)]
                    - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                    - theta[1]
                )
                ** 2
            )
            g0 = np.mean(
                (
                    -2
                    * (
                        Y[batch_size * b : batch_size * (b + 1)]
                        - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                        - theta[1]
                    )
                    * X[batch_size * b : batch_size * (b + 1)] ** 2
                )
            )
            g1 = np.mean(
                2
                * (
                    Y[batch_size * b : batch_size * (b + 1)]
                    - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                    - theta[1]
                )
            )
            v0 = v0 * beta + (1 - beta) * g0
            v1 = v1 * beta + (1 - beta) * g1
            theta[0] -= lr * v0
            theta[1] -= lr * v1
        print(f"loss:{loss / len(X)}, theta:{theta}")


def AdaGrad(X, Y, theta, epoch, batch_size, lr, epsilon=1e-6):
    G = [0] * len(theta)
    for _ in range(epoch):
        for b in range(len(X) // batch_size):
            loss = np.mean(
                (
                    Y[batch_size * b : batch_size * (b + 1)]
                    - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                    - theta[1]
                )
                ** 2
            )
            g0 = np.mean(
                (
                    -2
                    * (
                        Y[batch_size * b : batch_size * (b + 1)]
                        - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                        - theta[1]
                    )
                    * X[batch_size * b : batch_size * (b + 1)] ** 2
                )
            )
            g1 = np.mean(
                2
                * (
                    Y[batch_size * b : batch_size * (b + 1)]
                    - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                    - theta[1]
                )
            )
            G += np.array([g1, g1]) ** 2
            theta[0] -= lr / (np.sqrt(G[0] + epsilon)) * g0
            theta[1] -= lr / (np.sqrt(G[1] + epsilon)) * g1
        print(f"loss:{loss / len(X)}, theta:{theta}")


def RMSProp(X, Y, theta, epoch, batch_size, lr, beta=0.9, epsilon=1e-6):
    G = np.zeros(len(theta))
    v = np.zeros(len(theta))
    for _ in range(epoch):
        for b in range(len(X) // batch_size):
            loss = np.mean(
                (
                    Y[batch_size * b : batch_size * (b + 1)]
                    - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                    - theta[1]
                )
                ** 2
            )
            g0 = np.mean(
                (
                    -2
                    * (
                        Y[batch_size * b : batch_size * (b + 1)]
                        - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                        - theta[1]
                    )
                    * X[batch_size * b : batch_size * (b + 1)] ** 2
                )
            )
            g1 = np.mean(
                2
                * (
                    Y[batch_size * b : batch_size * (b + 1)]
                    - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                    - theta[1]
                )
            )
            v += beta * v + (1 - beta) * np.array([g0, g1]) ** 2
            theta[0] -= lr / (np.sqrt(v[0] + epsilon)) * g0
            theta[1] -= lr / (np.sqrt(v[1] + epsilon)) * g1
        print(f"loss:{loss / len(X)}, theta:{theta}")


def AdaDelta(X, Y, theta, epoch, batch_size, lr, beta=0.9, epsilon=1e-6):
    v = np.zeros(len(theta))
    G = np.zeros(len(theta))
    for _ in range(epoch):
        for b in range(len(X) // batch_size):
            loss = np.mean(
                (
                    Y[batch_size * b : batch_size * (b + 1)]
                    - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                    - theta[1]
                )
                ** 2
            )
            g0 = np.mean(
                (
                    -2
                    * (
                        Y[batch_size * b : batch_size * (b + 1)]
                        - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                        - theta[1]
                    )
                    * X[batch_size * b : batch_size * (b + 1)] ** 2
                )
            )
            g1 = np.mean(
                2
                * (
                    Y[batch_size * b : batch_size * (b + 1)]
                    - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                    - theta[1]
                )
            )
            v += beta * v + (1 - beta) * np.array([g0, g1])
            # current step size
            G += beta * G + (1 - beta) * np.array([g0, g1]) ** 2
            theta -= lr / (np.sqrt(G) + epsilon) * v
        print(f"loss:{loss / len(X)}, theta:{theta}")


def Adam(X, Y, theta, epoch, batch_size, lr, beta=0.9, epsilon=1e-6):
    v = np.zeros(len(theta))
    s = np.zeros(len(theta))
    for _ in range(epoch):
        for b in range(len(X) // batch_size):
            loss = np.mean(
                (
                    Y[batch_size * b : batch_size * (b + 1)]
                    - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                    - theta[1]
                )
                ** 2
            )
            g0 = np.mean(
                (
                    -2
                    * (
                        Y[batch_size * b : batch_size * (b + 1)]
                        - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                        - theta[1]
                    )
                    * X[batch_size * b : batch_size * (b + 1)] ** 2
                )
            )
            g1 = np.mean(
                2
                * (
                    Y[batch_size * b : batch_size * (b + 1)]
                    - theta[0] * X[batch_size * b : batch_size * (b + 1)] ** 2
                    - theta[1]
                )
            )
            v += beta * v + (1 - beta) * np.array([g0, g1]) ** 2
            # current step size
            si = np.sqrt(s + epsilon) / np.sqrt(v + epsilon) * np.array([g0, g1])
            s = beta * s + (1 - beta) * si**2
            theta[0] -= si[0]
            theta[1] -= si[1]
        print(f"loss:{loss / len(X)}, theta:{theta}")


x_array = np.array(
    [
        -8.50682451,
        -9.6530865,
        -9.45531063,
        -4.96858099,
        -5.40637658,
        -9.11626824,
        -2.61223439,
        -0.06889008,
        -8.60705315,
        4.18550108,
        -7.16096243,
        0.39929746,
        3.95656045,
        -8.07763126,
        3.30851174,
        -0.77371308,
        -2.04195773,
        -6.17199323,
        3.29515728,
        -3.41923808,
    ]
)

y_array = np.array(
    [
        225.09818991,
        287.54623668,
        276.20869734,
        82.0603913,
        95.68672312,
        257.31903991,
        28.47130553,
        8.01423753,
        230.24409169,
        60.555258,
        161.83814878,
        8.47831538,
        54.96311183,
        203.74438053,
        40.83874984,
        9.79589578,
        20.50877417,
        122.28050142,
        40.57418455,
        43.07356709,
    ]
)
Adam(x_array, y_array, [3, 2], 50, 10, 1e-4)
