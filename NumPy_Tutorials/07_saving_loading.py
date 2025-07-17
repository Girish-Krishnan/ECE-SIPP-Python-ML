import numpy as np


def main():
    arr = np.arange(9).reshape(3, 3)
    print("Original array:\n", arr)

    np.save('array.npy', arr)
    print('Array saved to array.npy')

    loaded = np.load('array.npy')
    print('Loaded array:\n', loaded)

    np.savetxt('array.txt', arr, fmt='%d')
    print('Also saved to array.txt')


if __name__ == '__main__':
    main()
