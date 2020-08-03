import numpy as np

# batch size
N = 64
# input/output dimension
D_in, D_out = 1000, 10
# hidden dimension
H = 100

# random input/output
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# randomly initialize weights
w1 = np.random.randn(D_in, H) # 1000 x 100
w2 = np.random.randn(H, D_out) # 100 x 10

learning_rate = 1e-6
for t in range(500):
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    loss = np.square(y_pred-y).sum()
    print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
