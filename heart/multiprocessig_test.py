from multiprocessing.pool import Pool, ThreadPool
import time

def f(x):
    time.sleep(5)
    return x

def multi_f(start):
    with Pool(2) as pool:
        results = pool.map(f, [i for i in range(start, start + 2)])
    return results
    
if __name__ == '__main__':

    start = time.perf_counter() 
    with ThreadPool(2) as pool:
        results = pool.map(multi_f, [0,10])
    end = time.perf_counter()

    print(f"Time: {end - start}")
    print(results)