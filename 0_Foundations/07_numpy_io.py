"""Saving and loading NumPy arrays."""
import numpy as np


def main():
    arr = np.arange(9).reshape(3, 3)
    print("Original array:\n", arr)

    np.save('array.npy', arr)
    print('Array saved to array.npy')

    loaded = np.load('array.npy')
    print('Loaded array:\n', loaded)

    # Save multiple arrays in a compressed npz
    np.savez_compressed('arrays.npz', first=arr, second=arr * 2)
    data = np.load('arrays.npz')
    print('\nArrays in npz:', list(data.keys()))

    np.savetxt('array.txt', arr, fmt='%d')
    print('Also saved to array.txt')


if __name__ == '__main__':
    main()
