from scipy import sparse as sp
import json
import numpy as np
import time

if __name__ == '__main__':
    n = 1000
    spmat = sp.rand(n, n, density=0.1, format="bsr")
    npmat = np.random.rand(n, n)

    times = 200_000

    u = np.random.rand(n, 1)
    start = time.perf_counter()
    for _ in range(times):
        u = spmat.dot(u)
        u = np.tanh(u)
    end = time.perf_counter()
    duration = end - start
    print(f"Sparse: {duration}")

    u = np.random.rand(n, 1)
    start = time.perf_counter()
    for _ in range(times):
        u = np.matmul(npmat, u)
        u = np.tanh(u)
    end = time.perf_counter()
    duration = end - start
    print(f"Numpy: {duration}")

    # print(mat.toarray())

    # with open("../roentgen/array.js", "w", encoding="utf-8") as f:
    #     data = f"var array = {json.dumps(mat.toarray().tolist())};"
    #     f.write(data)
