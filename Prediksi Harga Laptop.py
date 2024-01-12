# Fungsi sigmoid
def sigmoid(x):
    return 1 / (1 + (2.71828 ** -x))

# Fungsi turunan sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Fungsi feedforward
def feedforward(X, W1, B1, W2, B2):
    Z_in = [
        X[0] * W1[0][0] + X[1] * W1[0][1] + X[2] * W1[0][2] + X[3] * W1[0][3] + B1[0],
        X[0] * W1[1][0] + X[1] * W1[1][1] + X[2] * W1[1][2] + X[3] * W1[1][3] + B1[1],
        X[0] * W1[2][0] + X[1] * W1[2][1] + X[2] * W1[2][2] + X[3] * W1[2][3] + B1[2],
        X[0] * W1[3][0] + X[1] * W1[3][1] + X[2] * W1[3][2] + X[3] * W1[3][3] + B1[3]
    ]
    Z = [sigmoid(z) for z in Z_in]
    Y_in = Z[0] * W2[0] + Z[1] * W2[1] + Z[2] * W2[2] + Z[3] * W2[3] + B2
    Y = sigmoid(Y_in)
    return Y, Z

# Fungsi feedback
def feedback(X, Y, Z, target, W2, learning_rate, W1):
    error = target - Y

    # Perhitungan delta dan perbaikan bobot W2
    delta_Y = error * sigmoid_derivative(Y)
    W2 = [W2[i] + learning_rate * Z[i] * delta_Y for i in range(len(W2))]

    # Perhitungan delta dan perbaikan bobot W1
    delta_Z = [W2[i] * delta_Y * sigmoid_derivative(Z[i]) for i in range(len(Z))]
    W1 = [[W1[j][i] + learning_rate * X[j] * delta_Z[i] for i in range(len(W1[0]))] for j in range(len(W1))]

    return W1, W2

# Data input
X = [4, 5, 1, 3]

# Matriks bobot dan bias
W1 = [
    [0.2, 0.3, 0.1, 0.4],
    [0.5, 0.1, 0.2, 0.3],
    [-0.3, -0.2, -0.1, 0.2],
    [0.4, 0.3, -0.4, -0.1]
]

B1 = [0.1, 0.2, -0.1, 0.3]
W2 = [-0.2, 0.1, 0.3, 0.2]
B2 = 0.2

# Target
target = 1

# Learning rate
learning_rate = 0.01

# Lakukan feedforward
Y, Z = feedforward(X, W1, B1, W2, B2)

# Lakukan feedback
W1, W2 = feedback(X, Y, Z, target, W2, learning_rate, W1)

# Tampilkan hasil
print("Hasil prediksi feedforward:", Y)

# Kesimpulan
if Y > 0.5:
    print("Kesimpulan: Laptop diprediksi mahal")
else:
    print("Kesimpulan: Laptop diprediksi tidak mahal")
