import numpy as np
from scipy import sparse as sp



def normal(params, size=None):
    mu, scale = params
    return np.random.normal(mu, scale, size=size)

def get_neighbours(heart_pos, heart_shape, indicies, manhattan=1, passable="top"):
    neighbours = []
    h_i, h_j = heart_pos
    h, w = heart_shape
    
    if passable == "top":
        top_passable = True
        bottom_passable = False
    elif passable == "bottom":
        top_passable = False
        bottom_passable = True
    else:
        raise ValueError(f"Invalid passable parameter got '{passable}' expected 'top' or 'bottom'")

    for di in range(-manhattan, manhattan + 1):
        for dj in range(-manhattan, manhattan + 1):
            distance = abs(di) + abs(dj)
            if distance > manhattan:
                continue
                
            ni = h_i + di
            nj = h_j + dj

            # No periodic boundaries from sides
            if nj >= w or nj < 0:
                continue

            if top_passable and ni < h:
                neighbours.append([indicies[ni, nj], distance])

            if bottom_passable and ni >= 0:
                ni = ni % h
                neighbours.append([indicies[ni, nj], distance])
            
    return neighbours

def heartlike(
    pars,
    heart_pars
    ):
    
    heart_shape = (
        heart_pars.get('gridy'),
        heart_pars.get('gridx')
    )

    heart_weights = pars.get("local_heart_weights")
    n_local = heart_shape[0] * heart_shape[1]
    n_other = pars.get("local_n_other")
    n_reservoir = n_local + n_other
    indicies = np.reshape(np.arange(n_local), heart_shape)
    manhattan = pars.get("local_manhattan")
    detectors = heart_pars.get("detectors")
    w_max = pars.get("w_max")
    w_min = pars.get("w_min")
    spectral_radius = pars.get("spectral_radius")

    # ---------------------- W ------------------------------
    heart = np.zeros((n_local, n_local))
    for i in range(heart_shape[0]):
        for j in range(heart_shape[1]):
            neighbours = get_neighbours(
                [i, j],
                heart_shape,
                indicies,
                manhattan=manhattan,
                passable="top"
            )
            src = indicies[i, j]
            for neighbour, distance in neighbours:
                heart[src, neighbour] = normal(heart_weights) * (1 / (distance + 1))
    
    if n_other > 0:
        other = sp.rand(
            n_other,
            n_other, 
            density=pars.get("local_other_density"),
            format="bsr"
        ) *  (w_max - w_min)
    
        other.data = other.data + w_min
    
        # From other to heart
        M1 = np.zeros((n_local, n_other)) # np.random.random((n_local, n_other)) * (w_max - w_min) + w_min

        # From heart to other
        M2 = np.random.random((n_other, n_local)) * (w_max - w_min) + w_min # np.zeros((n_other, n_local))  

        w = np.block([
            [heart, M1],
            [M2, other.toarray()]
        ])
    else:
        w = heart

    sr = calc_sr(w)

    w = (w / sr) * spectral_radius

    # ---------------------- W_in-----------------------------
    n_detectors = len(detectors)
    n_input = n_detectors * 2 + 1
    w_in = np.zeros((n_reservoir, n_input))
    w_in_weights = pars.get("local_heart_w_in")
    w_in_other_weights = pars.get("w_in")
    n_other_in = n_detectors

    # Setup connection to heart
    for u_index, detector in enumerate(detectors):
        neighbours = get_neighbours(
            detector,
            heart_shape,
            indicies,
            manhattan=manhattan,
            passable="bottom"
        )

        for neighbour, distance in neighbours:
            w_in[neighbour, u_index] = normal(w_in_weights) * (1 / (distance + 1))

    # Setup connection to reservoir      
    for u_offset in range(n_other_in):
        u_index = n_detectors + u_offset
        connections_to_reservoir = normal(w_in_other_weights, size=n_other)
        w_in[n_local:,u_index] = connections_to_reservoir
    
    # Setup bias
    w_in_bias = pars.get("w_in_bias")
    w_in[:,-1] = normal(w_in_bias, size=n_reservoir)

    # ---------------------- W_out----------------------------
    n_output = len(heart_pars.get("injectors"))
    n_readouts = n_input + n_reservoir
    w_out = np.random.normal(
        0,
        0.5,
        (n_output, n_readouts)
    )

    return sp.bsr_array(w_in), sp.bsr_array(w), w_out

def calc_sr(w):
    return np.max(
        np.abs(
            np.linalg.eigvals(
                w
            )
        )
    )

def sparse(pars, heart_pars):
    n = pars.get("n_reservior")
    w_max = pars.get("w_max")
    w_min = pars.get("w_min")
    spectral_radius = pars.get("spectral_radius")

    # ---------------------- W ------------------------------
    w = sp.rand(
            n,
            n, 
            density=pars.get("sparse_density"),
            format="bsr"
        ) *  (w_max - w_min)
    
    w.data = w.data + w_min

    sr = calc_sr(w.toarray())

    w = (w / sr) * spectral_radius
    # ---------------------- W_in -----------------------------

    n_detectors = len(heart_pars.get("detectors"))
    n_input = n_detectors * 2 + 1
    w_in_weights = pars.get("w_in")
    w_in = normal(w_in_weights, size=(n, n_input))

    # ---------------------- W_out ----------------------------
    n_output = len(heart_pars.get("injectors"))
    n_readouts = n_input + n
    w_out = np.random.normal(
        0,
        0.5,
        (n_output, n_readouts)
    )

    return w_in, w, w_out

def get_architecture(pars, heart_pars):
    
    method = pars.get("w_method")

    if method == "local":
        w_in, w, w_out = heartlike(
            pars,
            heart_pars
        )
    
    elif method == "sparse":
        w_in, w, w_out = sparse(
            pars,
            heart_pars
        )
    else:
        raise ValueError(f"Unknown method: '{method}'")

    return w_in, w, w_out


if __name__ == "__main__":
    
    from settings import Params

    # heart_shape = (4, 3)
    # indicies = np.arange(heart_shape[0] * heart_shape[1]).reshape(heart_shape)
    # ns = get_neighbours((1, 0), heart_shape, indicies, manhattan=2)

    heart_pars = Params("./params.json")
    pars = Params("./doctor_params.json")
    heart_pars.override("gridx", 3)
    heart_pars.override("gridy", 4)
    heart_pars.override("detectors", [[0, 0], [1, 0]])

    w_in, w, w_out = heartlike(
        pars,
        heart_pars,
        {
            "heart_weights": [0.9, 0.2],
            "w_in": [0.1, 0.05],
            "n_other": 0,
            "manhattan": 1
        }
    )
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    s = np.array_repr(
            w.toarray()[:,:]
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

    print(s)