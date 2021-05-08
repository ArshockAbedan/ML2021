import matplotlib.pyplot as pl

def plot_reconstruction(autoencoder, inputs, n=10):
    outputs = autoencoder.predict(inputs[:n])

    pl.figure(figsize=(20, 4))
    for i in range(n):
        ax = pl.subplot(2, n, i+1)
        pl.imshow(inputs[i].reshape(28, 28), cmap='gray')
        pl.axis('off')

        ax = pl.subplot(2, n, n + i + 1)
        pl.imshow(outputs[i].reshape(28, 28), cmap='gray')
        pl.axis('off')
