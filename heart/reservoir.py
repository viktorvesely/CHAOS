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

    n_local = heart_shape[0] * heart_shape[1]
    n_other = pars.get("local_n_other")
    n_reservoir = n_local + n_other
    indicies = np.reshape(np.arange(n_local), heart_shape)
    manhattan = pars.get("local_manhattan")
    detectors = heart_pars.get("detectors")
    spectral_radius = pars.get("spectral_radius")
    w_heart_heart, w_other_heart, w_heart_other, w_other_other = pars.get("local_w_architecture")
    w_in_cur_heart, w_in_fut_heart, w_in_cur_other, w_in_fut_other = pars.get("local_w_in_architecture")

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
                heart[src, neighbour] = normal(w_heart_heart) * (1 / (distance + 1))
    
    if n_other > 0:
        o_o_mu, o_o_sd = w_other_other
        other = sp.random(
            n_other,
            n_other, 
            density= 1 - pars.get("local_other_density"),
            format="csr"
        ) * o_o_sd
    
        other.data = other.data + o_o_mu
        
        # From other to heart
        M1 = normal(w_other_heart, size=(n_local, n_other)) # np.random.random((n_local, n_other)) * (w_max - w_min) + w_min

        # From heart to other
        M2 = normal(w_heart_other, size=(n_other, n_local)) # np.zeros((n_other, n_local))  

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
    w_in_block_fut_heart = np.zeros((n_local, n_detectors))
    w_in_block_cur_heart = np.zeros((n_local, n_detectors))


    for u_index, detector in enumerate(detectors):
        neighbours = get_neighbours(
            detector,
            heart_shape,
            indicies,
            manhattan=manhattan,
            passable="bottom"
        )


        for neighbour, distance in neighbours:
            # Current to heart
            w_in_block_cur_heart[neighbour, u_index] = normal(w_in_cur_heart) * (1 / (distance + 1))
            # Future to heart
            w_in_block_fut_heart[neighbour, u_index] = normal(w_in_fut_heart) * (1 / (distance + 1))


    w_in_block_fut_other = normal(w_in_fut_other, (n_other, n_detectors))
    w_in_block_cur_other = normal(w_in_cur_other, (n_other, n_detectors))

    w_in_no_bias = np.block([
            [w_in_block_cur_heart, w_in_block_fut_heart],
            [w_in_block_cur_other, w_in_block_fut_other]
        ])
    

    # Set everything but bias
    w_in = np.zeros((n_reservoir, n_input))
    w_in[:,:-1] = w_in_no_bias

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

    # ---------------------- Leaky mask------------------------

    leaky_mask = np.zeros((n_reservoir, 1))
    local_leaky = pars.get("local_heart_leak_alpha")
    other_min = pars.get("leaky_alpha_min")
    other_max = pars.get("leaky_alpha_max")
    leaky_mask[:n_local,:] = np.ones((n_local, 1)) * local_leaky
    leaky_mask[n_local:,:] = np.random.random((n_other, 1)) * (other_max - other_min) + other_min

    return w_in, w, w_out, leaky_mask

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
            density= 1- pars.get("sparse_density"),
            format="csr"
        ) *  (w_max - w_min)

    w.data = w.data + w_min
    w = w.toarray()

    sr = calc_sr(w)

    w = (w / sr) * spectral_radius
    # ---------------------- W_in -----------------------------

    n_detectors = len(heart_pars.get("detectors"))
    n_input = n_detectors * 2 + 1
    w_in_weights = pars.get("w_in")
    w_in = normal(w_in_weights, size=(n, n_input))

    # Setup bias
    w_in_bias = pars.get("w_in_bias")
    w_in[:,-1] = normal(w_in_bias, size=n)

    # ---------------------- W_out ----------------------------
    n_output = len(heart_pars.get("injectors"))
    n_readouts = n_input + n
    w_out = np.random.normal(
        0,
        0.5,
        (n_output, n_readouts)
    )

    # ---------------------- Leaky mask------------------------

    leaky_alpha_min = pars.get("leaky_alpha_min")
    leaky_alpha_max = pars.get("leaky_alpha_max")
    leaky_mask = np.random.random((n, 1)) * (leaky_alpha_max - leaky_alpha_min) + leaky_alpha_min 

    return w_in, w, w_out, leaky_mask


def spatial(pars, heart_pars):

    if heart_pars.exists("__ss_shape"):
        shape = heart_pars.get("__ss_shape")
    else:
        shape = (heart_pars.get("gridx"), heart_pars.get("gridy"))
    
    n_state = shape[0] * shape[1]

    x_per_u = pars.get("spatial_neruons_per_input")
    n =  x_per_u * n_state
    pars.override("n_reservior", n)
    w_sigma = pars.get("spatial_w_sigma")
    w_in_pars = pars.get("spatial_w_in")
    spectral_radius = pars.get("spectral_radius")
    manhattan = pars.get("spatial_manhattan")

    # ---------------------- W ------------------------------
        
    w = np.zeros((n, n))
    for u_base in range(n_state):
        for x_base1 in range(x_per_u):
            x_index1 = x_per_u * u_base + x_base1
            for x_base2 in range(x_per_u):
                x_index2 = x_per_u * u_base + x_base2
                w[x_index1, x_index2] = np.random.random() * w_sigma * 2 - w_sigma

    sr = calc_sr(w)

    w = (w / sr) * spectral_radius

    w = sp.bsr_array(w)
    # ---------------------- W_in -----------------------------

    n_input = n_state * 2 + 1
    w_in = np.zeros((n, n_input))

    indices = np.arange(n_state).reshape(shape)
    tenses = 2

    for i in range(shape[0]):
        for j in range(shape[1]):
            for tense in range(tenses):
                u_base = indices[i, j]
                offset = tense * n_state
                neighbours = get_neighbours(
                    [i, j],
                    shape,
                    indices,
                    manhattan=manhattan,
                    passable="top"
                )

                for neighbour, distance in neighbours:
                    for x_base in range(x_per_u):
                        x_index = x_per_u * u_base + x_base
                        neighbour_index = neighbour + offset
                        w_in[x_index, neighbour_index] = normal(w_in_pars) / (distance * 0.5 + 1)
    
                    

    # Setup bias
    w_in_bias = pars.get("spatial_w_bias")
    w_in[:,-1] = normal(w_in_bias, size=n)

    # ---------------------- W_out ----------------------------
    n_output = len(heart_pars.get("injectors"))
    n_readouts = n_input + n
    w_out = np.random.normal(
        0,
        0.5,
        (n_output, n_readouts)
    )

    # ---------------------- Leaky mask------------------------

    leaky_alpha_min = pars.get("leaky_alpha_min")
    leaky_alpha_max = pars.get("leaky_alpha_max")
    leaky_mask = np.random.random((n, 1)) * (leaky_alpha_max - leaky_alpha_min) + leaky_alpha_min 
    
    return w_in, w, w_out, leaky_mask


def material(pars, heart_pars):

    n = pars.get("n_reservior")
    w_sigma = pars.get("material_sigma")
    spectral_radius = pars.get("spectral_radius")

    # ---------------------- W ------------------------------
    w = sp.rand(
            n,
            n, 
            density= 1 - pars.get("material_density"),
            format="csr"
        )*  w_sigma * 2 

    w.data = w.data - w_sigma
    w = w.toarray()

    sr = calc_sr(w)

    w = (w / sr) * spectral_radius
    # ---------------------- W_in -----------------------------

    pca_dim = pars.get("pca_dim")
    if pca_dim > 0:
        n_state = pca_dim
    elif heart_pars.exists("__ss_shape"):
        shape = heart_pars.get("__ss_shape")
        n_state = shape[0] * shape[1]
    else:
        n_state = heart_pars.get("gridx") * heart_pars.get("gridy")
        
    n_input = n_state * 2 + 1
    #n_half = int(np.ceil(n / 2))
    w_in_weights = pars.get("material_w_in")
    w_in = np.zeros((n, n_input))
    w_in[:,:-1] = normal(w_in_weights, size=(n, n_state * 2))

    # Setup bias
    w_in_bias = pars.get("material_w_bias")
    w_in[:,-1] = normal(w_in_bias, size=n)

    # ---------------------- W_out ----------------------------
    n_output = len(heart_pars.get("injectors"))
    n_readouts = n_input + n
    w_out = np.random.normal(
        0,
        0.5,
        (n_output, n_readouts)
    )

    # ---------------------- Leaky mask------------------------

    leaky_alpha_min = pars.get("leaky_alpha_min")
    leaky_alpha_max = pars.get("leaky_alpha_max")
    leaky_mask = np.random.random((n, 1)) * (leaky_alpha_max - leaky_alpha_min) + leaky_alpha_min 
    
    return w_in, w, w_out, leaky_mask

def get_architecture(pars, heart_pars):
    
    method = pars.get("w_method")

    if method == "local":
        w_in, w, w_out, leaky_mask = heartlike(
            pars,
            heart_pars
        )
    
    elif method == "sparse":
        w_in, w, w_out, leaky_mask = sparse(
            pars,
            heart_pars
        )
    elif method == "material":
        w_in, w, w_out, leaky_mask = material(
            pars,
            heart_pars
        )
    elif method == "spatial":
        w_in, w, w_out, leaky_mask = spatial(
            pars,
            heart_pars
        )
    else:
        raise ValueError(f"Unknown method: '{method}'")

    return w_in, w, w_out, leaky_mask


if __name__ == "__main__":
    
    from settings import Params
    from os.path import join
    import os
    # heart_shape = (4, 3)
    # indicies = np.arange(heart_shape[0] * heart_shape[1]).reshape(heart_shape)
    # ns = get_neighbours((1, 0), heart_shape, indicies, manhattan=2)

    pars = Params("./doctor_params.json")
    heart = join(os.getcwd(), 'hearts', pars.get("dataset"), 'params.json')
    heart_pars = Params(heart)


    w_in, w, w_out, _ = spatial(
        pars,
        heart_pars
    )



    for i, row in enumerate(w_in):
        non_zero = np.sum(~(row == 0))
        print(non_zero)

    # print(w.blocksize)
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # s = np.array_repr(
    #         w.toarray()[:,:]
    # ).replace(
    #     "\n",
    #     ""
    # ).replace(
    #     "array",
    #     ""
    # ).replace(
    #     "]",
    #     "]\n"
    # ).replace(
    #     ")",
    #     ""
    # ).replace(
    #     "(",
    #     ""
    # ).replace(
    #     "  ",
    #     ""
    # )

    # s = f" {s}"

    # print(s)