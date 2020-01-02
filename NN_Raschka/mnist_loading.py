import mnist_raschka_loader


X_train, y_train = mnist_raschka_loader.load_mnist('NN_Raschka/mnist/', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = mnist_raschka_loader.load_mnist('NN_Raschka/mnist/', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

mnist_raschka_loader.viewing_each_character_sample(X_train, y_train)
mnist_raschka_loader.viewing_samples_one_character(X_train, y_train)