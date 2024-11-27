import numpy as np

'''  
       Veri Seti 
x1     x2     x3      y 
0       0      1      1 
0       1      1      1 
1       0      1      1 
1       1      1      0 
 
1       1      0      0 
0       0      0      0 
'''

# Modelin başlangıç ağırlıklarını (w1, w2, w3, Ø), öğrenme hızı parametresini (ζ) ve eğitimin en fazla kaç tekrar
# yapılacağına ilişkin epoch sınırını(epoch_max) kullanıcıdan alma
print(
    "Ağırlık değerlerini(w1, w2, w3, Ø), öğrenme hızı parametresini(ζ) ve eğitimin en fazla kaç tekrar yapılacağına "
    "ilişkin epoch sınırını(epoch_max) giriniz:")
w1 = float(input("w1 (0-1 aralığında): "))
w2 = float(input("w2 (0-1 aralığında): "))
w3 = float(input("w3 (0-1 aralığında): "))
theta = float(input("Bias (θ) (0-1 aralığında): "))
learning_rate = float(input("Öğrenme hızı (ζ): "))
epoch_max = int(input("Maksimum epoch sayısı: "))

# Ağırlıkların ve bias'ın vektör olarak tanımlanması
weights = np.array([w1, w2, w3])
bias = theta

# Veri setinin ilk 4 örneği ile modelin eğitimi
X_train = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# Eğitim seti için gerçek çıktılar
y_train = np.array([1, 1, 1, 0])

# Model eğitildikten sonra son 2 örnek üzerinde tahmin yapma
X_test = np.array([
    [1, 1, 0],
    [0, 0, 0]
])

# Test seti için gerçek çıktılar
y_test = np.array([0, 0])


# Perceptron öğrenme algoritması
def perceptron_train(X, y, weights, bias, learning_rate, epoch_max):
    epoch = 0
    while epoch < epoch_max:
        error_count = 0  # Hata sayacı
        for i in range(len(X)):
            # Net girdi ve tahmin
            net_input = np.dot(X[i], weights) + bias  # y = x1.w1 + x2.w2 + x3.w3 + θ
            # Perceptron için sınıflar 1 ve 0
            y_prediction = 1 if net_input >= 0 else 0  # Tahmin, net girdi eşik değerini(0) aşarsa 1, aşmazsa 0 olarak belirlenir.

            # Hata kontrolü ve ağırlık güncelleme
            if y_prediction != y[i]:
                error_count += 1
                weights = learning_rate * (y[i] - y_prediction) * X[i] + weights  # W = η.e.i + ΔW(previous)
                bias = learning_rate * (y[i] - y_prediction) + bias               # θ = η.e.1 + Δθ(previous)

        # Eğer hata yoksa öğrenme tamamlandı
        if error_count == 0:
            print(f"Öğrenme, {epoch + 1} epoch sonunda tamamlandı.")
            return weights, bias

        epoch += 1

    print(
        "Epoch sınırına ulaşıldı, ancak öğrenme tamamlanamadı. Programdan çıkış yapılıyor. Farklı parametrelerle tekrar deneyiniz.")
    return weights, bias


# Eğitim işlemi
weights, bias = perceptron_train(X_train, y_train, weights, bias, learning_rate, epoch_max)


# Test verileri üzerinde tahmin yapma
def perceptron_predict(X, weights, bias):
    predictions = []
    for i in range(len(X)):
        net_input = np.dot(X[i], weights) + bias   # y = x1.w1 + x2.w2 + x3.w3 + θ
        y_prediction = 1 if net_input >= 0 else 0  # Sınıf tahmini
        predictions.append(y_prediction)
    return predictions

# Test sonuçlarını karşılaştırma
y_pred_test = perceptron_predict(X_test, weights, bias)
for i in range(len(X_test)):
    print(f"{i + 5}. veri vektöründe tahmin edilen değer {y_pred_test[i]}, "
          f"gerçek değer {y_test[i]} ve hata {y_test[i] - y_pred_test[i]}.")
