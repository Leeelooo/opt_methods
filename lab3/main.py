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
        y = forward_substitution(L, b[i, :])

        A_inv[:, i] = back_substitution(U, y)

    return A_inv


LU_FUNC = scipy.linalg.lu


def test_lu(A):
    L, U = lu_doolittle(A)
    A_1 = L @ U
    error_1 = np.linalg.norm(A - A_1)

    P, L, U = LU_FUNC(A)
    error_2 = np.linalg.norm(P @ A - L @ U)

    print(f"error my: {error_1}\terror lib:{error_2}")

    return error_1, error_2


def test_solve(A, b):
    x = lu_solve(A, b)

    valid = np.all(A @ x == b)
    print("PASS" if valid else "ERROR")

    return valid


def test_inv(A):
    inv = lu_inverse(A)
    inv_lib = scipy.linalg.inv(A)

    diff = np.linalg.norm(inv_lib - inv)

    print(f"diff with lib: {diff}")


hilbert_generator = lambda n: np.fromfunction(lambda i, j: 1 / (i + j + 1), (n, n), dtype=np.float) # since we are starting from i=0 and j=0


TESTS = [
    ("LU Decomposition", test_lu, (
        (np.array([[1, 0, 3], [0, 3, 1], [0, 0, 6]])),
        (np.array([[1, 4, 5], [6, 8, 22], [32, 5., 5]])),
        (hilbert_generator(5)),
        (hilbert_generator(7))
    )),
    ("LU Solve", test_solve, (
        (
            (np.array([[1, 4, 5], [6, 8, 22], [32, 5., 5]]),
             np.array([1, 2, 3.]))
        ),
    )),
    ("LU Inverse", test_inv, (
        (np.array([[1, 0, 3], [0, 3, 1], [0, 0, 6]])),
        (np.array([[1, 4, 5], [6, 8, 22], [32, 5., 5]])),
        (hilbert_generator(5)),
        (hilbert_generator(7))
    ))
]

if __name__ == "__main__":
    for name, func, args in TESTS:
        print(f"TEST {name}")
        for A in args:
            if isinstance(A, np.ndarray):
                func(A)
            else:
                func(*A)
