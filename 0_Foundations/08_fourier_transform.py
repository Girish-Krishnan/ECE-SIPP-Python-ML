"""Fourier transform demonstration."""
import numpy as np
import matplotlib.pyplot as plt


def main():
    t = np.linspace(0, 1, 500)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(t), d=t[1] - t[0])

    print('First 10 frequency magnitudes:')
    for f, mag in zip(freqs[:10], np.abs(fft)[:10]):
        print(f"{f:5.2f} Hz: {mag:.3f}")

    # Visualize the signal and its spectrum
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title("Time domain signal")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")

    plt.subplot(2, 1, 2)
    plt.stem(freqs, np.abs(fft), use_line_collection=True)
    plt.title("Frequency spectrum")
    plt.xlabel("frequency [Hz]")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
