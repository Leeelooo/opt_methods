import numpy as np
import scipy.linalg


def _check_square(A):
    assert len(A.shape) == 2, "only 2d matrices are supported"
    assert A.shape[0] == A.shape[1], "only nxn matrices are supported"


def lu_doolittle(A):
    _check_square(A)

    n = A.shape[0]

    U = np.zeros((n, n), dtype=np.double)
    L = np.eye(n, dtype=np.double)

    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(n):
            U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
            L[(k+1):, k] = (A[(k+1):, k] - L[(k+1):, :] @ U[:, k]) / U[k, k]

        L[~np.isfinite(L)] = 0

    return L, U


def forward_substitution(L, b):
    _check_square(L)
    assert L.shape[0] == b.shape[0]

    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double)

    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y


def back_substitution(U, y):
    _check_square(U)
    assert U.shape[0] == y.shape[0]

    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double)

    x[-1] = y[-1] / U[-1, -1]

    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]

    return x


def lu_solve(A, b):
    assert A.shape[0] == b.shape[0]

    L, U = lu_doolittle(A)

    y = forward_substitution(L, b)

    return back_substitution(U, y)


def lu_inverse(A):
    _check_square(A)

    n = A.shape[0]

    b = np.eye(n, dtype=np.double)
    A_inv = np.zeros_like(A, dtype=np.double)

    L, U = lu_doolittle(A)

    for i in range(n):
        y = forward_substitution(L, b[:, i])
        A_inv[:, i] = back_substitution(U, y)

    return A_inv


LU_FUNC = scipy.linalg.lu


def test_lu(A):
    L, U = lu_doolittle(A)
    A_1 = L @ U
    error_1 = np.linalg.norm(A - A_1)

    P, L, U = LU_FUNC(A)
    error_2 = np.linalg.norm(A - P @ L @ U)

    print(f"error my: {error_1}\terror lib:{error_2}")

    return error_1, error_2


def test_solve(A, b):
    x = lu_solve(A, b)
    error_my = np.linalg.norm(A @ x - b)
    error_lib = np.linalg.norm(A @ scipy.linalg.solve(A, b) - b)

    print(f'error my:{error_my}\terror lib:{error_lib}')


def test_inv(A):
    inv = lu_inverse(A)
    inv_lib = scipy.linalg.inv(A)

    I = np.eye(A.shape[0])

    print(
        f"error my: {np.linalg.norm(A @ inv - I)}\terror lib:{np.linalg.norm(A @ inv_lib - I)}")


def noisy_matrix(n):
    np.random.seed(59005)

    matrix = -np.random.choice(5, size=(n, n)).astype(np.double)
    for i in range(n):
        matrix[i, i] = -(np.sum(matrix[i]) - matrix[i, i]) + 10 ** -n

    return matrix


def hilbert_generator(n): return np.fromfunction(lambda i, j: 1 / (i + j + 1),
                                                 (n, n), dtype=np.float)  # since we are starting from i=0 and j=0


TESTS = [
    ("LU Decomposition", test_lu, (
        (np.array([[1, 0, 3], [0, 3, 1], [0, 0, 6]]),),
        (np.array([[1, 4, 5], [6, 8, 22], [32, 5, 5]]),),
        (noisy_matrix(5),),
        (noisy_matrix(7),),
        (noisy_matrix(9),),
        (noisy_matrix(11),),
        (noisy_matrix(13),),
        (hilbert_generator(5),),
        (hilbert_generator(7),),
        (hilbert_generator(9),),
        (hilbert_generator(11),),
        (hilbert_generator(13),)
    )),
    ("LU Solve", test_solve, (
        (
            (np.array([[1, 4, 5], [6, 8, 22], [32, 5, 5]]),
             np.array([1, 2, 3.]))
        ),
        (
            (noisy_matrix(5), np.dot(noisy_matrix(5), np.arange(1, 6)))
        ),
        (
            (noisy_matrix(7), np.dot(noisy_matrix(7), np.arange(1, 8)))
        ),
        (
            (noisy_matrix(9), np.dot(noisy_matrix(9), np.arange(1, 10)))
        ),
        (
            (noisy_matrix(11), np.dot(noisy_matrix(11), np.arange(1, 12)))
        ),
        (
            (noisy_matrix(13), np.dot(noisy_matrix(13), np.arange(1, 14)))
        ),
        (
            (hilbert_generator(5), np.dot(hilbert_generator(5), np.arange(1, 6)))
        ),
        (
            (hilbert_generator(7), np.dot(hilbert_generator(7), np.arange(1, 8)))
        ),
        (
            (hilbert_generator(9), np.dot(hilbert_generator(9), np.arange(1, 10)))
        ),
        (
            (hilbert_generator(11), np.dot(hilbert_generator(11), np.arange(1, 12)))
        ),
        (
            (hilbert_generator(13), np.dot(hilbert_generator(13), np.arange(1, 14)))
        )
    )),
    ("LU Inverse", test_inv, (
        (np.array([[1, 0, 3], [0, 3, 1], [0, 0, 6]]),),
        (np.array([[1, 4, 5], [6, 8, 22], [32, 5, 5]]),),
        (noisy_matrix(5),),
        (noisy_matrix(7),),
        (noisy_matrix(9),),
        (noisy_matrix(11),),
        (noisy_matrix(13),),
        (hilbert_generator(5),),
        (hilbert_generator(7),),
        (hilbert_generator(9),),
        (hilbert_generator(11),),
        (hilbert_generator(13),)
    ))
]

if __name__ == "__main__":
    for name, func, args in TESTS:
        print(f"TEST {name}")
        for A in args:
            func(*A)
