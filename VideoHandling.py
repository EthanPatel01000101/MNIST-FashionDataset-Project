import tensorflow as tf
from NeuralNetwork import Network

(trainingX, trainingY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainingX = trainingX / 255.0
testX = testX / 255.0

print(len(trainingX[0].flatten()))

model = Network(28*28, 128, 128, 10, softmax=True)