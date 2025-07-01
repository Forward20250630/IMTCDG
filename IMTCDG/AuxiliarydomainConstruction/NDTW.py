import numpy as np

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))


def Origin_DTW(m_seq1, m_seq2):
    n = len(m_seq1)
    m = len(m_seq2)

    # Create a distance matrix and initialize the first row and column
    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        dtw_matrix[i][0] = float('inf')
    for j in range(1, m + 1):
        dtw_matrix[0][j] = float('inf')
    dtw_matrix[0][0] = 0

    # Fill in the rest of the matrix using dynamic programming
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean_distance(m_seq1[i - 1], m_seq2[j - 1])
            dtw_matrix[i][j] = cost + min(dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1])

    # The DTW distance is the value in the bottom-right corner of the matrix
    return dtw_matrix[n][m]


def Multivariate_DTW(X, Y):
    """ Multivariate time series / Synchronous Dynamic Time Warping (S-DTW) distance """
    n_timestamps_X, n_dimensions_X = X.shape
    n_timestamps_Y, n_dimensions_Y = Y.shape

    # Check dimensions compatibility
    if n_dimensions_X != n_dimensions_Y:
        raise ValueError("Number of dimensions in X and Y must be the same.")

    # Initialize the DTW matrix
    dtw_matrix = np.zeros((n_timestamps_X + 1, n_timestamps_Y + 1))

    # Initialize the first row and column of the DTW matrix
    for i in range(1, n_timestamps_X + 1):
        dtw_matrix[i, 0] = float('inf')
    for j in range(1, n_timestamps_Y + 1):
        dtw_matrix[0, j] = float('inf')

    # Fill the DTW matrix
    for i in range(1, n_timestamps_X + 1):
        for j in range(1, n_timestamps_Y + 1):
            cost = np.linalg.norm(X[i-1, :] - Y[j-1, :])  # Euclidean distance
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])

    return dtw_matrix[-1, -1] / n_dimensions_X


class Improved_DTW(object):
    def __init__(self, method_type="Origin"):
        self.method_type = method_type

    def Compute(self, X, Y):
        if self.method_type == "Origin":
            DTW_distance = Origin_DTW(X, Y)
        else:
            DTW_distance = Multivariate_DTW(X, Y)

        return DTW_distance

if __name__ == "__main__":
    i_DTW = Improved_DTW(method_type="Multivariate")
    X = np.array([[1, 2, 3], [3, 4, 4], [5, 6, 5], [2, 4, 6]])
    Y = np.array([[2, 3, 4], [4, 5, 6], [6, 7, 7], [8, 9, 10]])
    dtw_distance = i_DTW.Compute(X, Y)
    print("DTW distance:", dtw_distance)