import numpy as np
from scipy import sparse as sp

def w_local(n, w_min, w_max, spectral_radius, settings):
    local_ratio = settings["local_ratio"]
    
    n_local = round(n * local_ratio)
    n_other = n - n_local
    
    local = []
    for i in range(n_local):
        row = []
        for z in range(n_local):
            if abs(z - i) <= 2:
                row.append(np.random.random() * (w_max - w_min) + w_min)
            else:
                row.append(0)
        local.append(row)

    local = np.array(local)

    other = np.random.random(size=(n_other, n_other)) * (w_max - w_min) + w_min
 
    M1 = np.zeros((n_local, n_other))
    M2 = np.zeros((n_other, n_local))

    w = np.block([
        [local, M1],
        [M2, other]
    ])    

    sr = calc_sr(w)

    w = (w / sr) * spectral_radius

    return sp.bsr_array(w)

def calc_sr(w):
    return np.max(
        np.abs(
            np.linalg.eigvals(
                w
            )
        )
    )

def w_sparse(n, w_min, w_max, spectral_radius, settings):
    w = sp.rand(
            n,
            n, 
            density=settings["density"],
            format="bsr"
        ) *  (w_max - w_min)
    
    w.data = w.data + w_min

    sr = calc_sr(w.toarray())

    w = (w / sr) * spectral_radius

    return w

def get_w(n_reservoir, pars):
    
    method = pars.get("w_method")
    settings = pars.get(f"w_{method}")
    w_min = pars.get("w_min")
    w_max = pars.get("w_max")
    spectral_radius = pars.get("spectral_radius")

    if method == "local":
        w = w_local(n_reservoir, w_min, w_max, spectral_radius, settings)
    
    elif method == "sparse":
        w = w_sparse(n_reservoir, w_min, w_max, spectral_radius, settings)

    return w



if __name__ == "__main__":
    
    from settings import Params
    import os

    pars = Params("./doctor_params.json")
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    w = get_w(pars.get("n_reservior"), pars)
    s = np.array_repr(
            w.toarray()
    ).replace(
        "\n",
        ""
    ).replace(
        "array",
        ""
    ).replace(
        "]",
        "]\n"
    ).replace(
        ")",
        ""
    ).replace(
        "(",
        ""
    ).replace(
        "  ",
        ""
    )

    s = f" {s}"

    print(calc_sr(w.toarray()))
    
    sp.save_npz(os.path.join(os.getcwd(), "trash", "w.npz"), w)
    
    w = sp.load_npz(os.path.join(os.getcwd(), "trash", "w.npz"))

    print(calc_sr(w.toarray()))