import numpy as np
from tqdm import tqdm

# what you need
# weights computed for x and y derivative approximation
# location of the central node and its neighbours
def resolving_power(w: dict,
                    s: float,
                    neigh_dist_xy: dict,
                    n_samples: int,
                    k_samples: int = 100
                    ) -> (dict, dict):

    pi = np.pi

    # Computing Nyquist wavenumber based on average particle distance
    k_ny = pi / s

    #k_ny_dict = {}
    #for key, val in neigh_dist_xy.items():
    #    k_ny_dict[key] = np.mean(val)

    # Samples to scale Nyquist wavenumber
    samples = np.linspace(0, 1, k_samples)

    # Fixed wavenumber
    theta = 0 # np.linspace(0, pi, 3)

    # Scaled sampled Nyquist wavenumber
    k_scaled = samples * k_ny

    # Scaling fixed wavenumbers by sampled Nyquist wavenumber
    # essentially chooses how much of the total wave number is in the real (x dim) or imaginary (y dim) part
    k_x = k_scaled * np.cos(theta) # real
    k_y = k_scaled * np.sin(theta) # imaginary

    # Sampling nodes from domain to do resolving power analysis
    rng = np.random.default_rng(42)
    ls = list(neigh_dist_xy.keys())
    ls = np.array(ls)
    n_samples = min(n_samples, len(w))
    coor_idx = rng.choice(len(ls), size=n_samples, replace=False)
    coor_to_test = ls[coor_idx]

    k_final = []

    # Computing ||k||, although I don't actually use it so that k_hat and k_eff range from [0,1]
    k = np.sqrt(k_x ** 2 + k_y ** 2)

    # Normalising k to be [0,1]
    k_hat = k * k_x / k_ny #* k_x
    #k_hat = k_x / k_ny  # * k_x

    # looping over each sample in k_x
    for i in tqdm(range(len(k_x)), desc="Processing k_x"):
        tmp = []
        # loops over each node that will be used to compute the resolving power
        for c_node in coor_to_test:
            n_dist = neigh_dist_xy[tuple(c_node)] # obtains the x and y distances of all neighbours of the node
            S_sin, S_cos = 0, 0                   # initialise the real and imaginary parts of k_eff
            # loops over each neighbour of a central node
            for j, neigh in enumerate(n_dist):
                eta = k_x[i] * neigh[0] + k_y[i] * neigh[1]         # computes eta = (k_x * x_ji + k_y * y_ji)
                S_sin += np.sin(eta) * w[tuple(c_node)][j]          # computes sin(eta) * w_ji
                S_cos += (1 - np.cos(eta)) * w[tuple(c_node)][j]    # computes (1 - cos(eta)) * w_ji

            # normalises the sum of the real and imaginary parts by k_ny
            R = S_sin * k[i] / k_ny
            I = S_cos * k[i]/ k_ny

            # below some data structuring
            tmp.append([R, I, k_hat[i]])

        k_final.append(tmp)

    k_final = np.array(k_final)
    k_tmp   = np.sqrt(np.mean(k_final[..., :-1] ** 2, axis=1))
    k_final = np.concat((k_tmp, k_final[:, 0, -1][:, None]), axis=1) # averaging for all nodes


    return k_final
