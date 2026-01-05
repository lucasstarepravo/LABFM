import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs


def plot_kernel(features, labels=None, alpha=0.6, size = 2):
    # flatten all arrays consistently
    x = features[:, :, 0].flatten()
    y = features[:, :, 1].flatten()
    if labels is not None:
        c = labels.flatten()

    plt.figure(figsize=(6, 6))
    if labels is not None:
        sc = plt.scatter(x, y, c=c, cmap='viridis', s=size, alpha=alpha)
    else:
        sc = plt.scatter(x, y, cmap='viridis', s=size, alpha=alpha)
    plt.xlabel('x distance')
    plt.ylabel('y distance')
    plt.title('Neighbour offsets coloured by target')
    plt.axis('equal')
    plt.colorbar(sc, label='Target value')
    plt.show()

mpl.rcParams.update({
    "savefig.format": "pdf",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "axes.unicode_minus": False,
})
def plot_stability(
    results: dict,
    kernel: str | int,
    resolution: int,
    diff_operator: str,
    save: bool = False,
    filename: str = "spectrum.pdf",
    column: str = "single",        # "single" or "double"
    show_axes0: bool = True,
    point_size: float = 8.0,
) -> None:
    print('Computing eigenvalues')
    # (results, model_or_polynomial, resolution, diff_operator)

    if kernel not in [2, 4, 5, 8, 'gnn', 'wc2', 'q_s']:
        raise ValueError("Value of the kernel should either be a LABFM polynomial"
                         " in results one of the following kernels 'gnn', 'wc2' "
                         "'q_s'")


    if diff_operator not in ['dx', 'dy', 'laplace']:
        raise ValueError("Differential operator must be one of the following"
                         " 'dx', 'dy', or 'laplace'")

    # extract the correct results from variable results that has the desired resolution
    # and the kernel specified
    if (resolution, kernel) not in results.keys():
        raise ValueError('Kernel-resolution combination specified not in results')

    # extracting all attributes from object
    attrs = results[(resolution, kernel)]

    # extracting weights from attributes
    if diff_operator == 'dx': weights = attrs.x
    if diff_operator == 'dy': weights = attrs.y
    if diff_operator == 'laplace': weights = attrs.laplace

    h = attrs.h
    # extracting coordinates from attributes
    coor = attrs.coordinates
    neigh_coor = attrs._neigh_coor

    # surface value is not actually required, I only need the order of the position of the nodes i.e. coor
    A = np.zeros((len(coor), len(coor)))

    coord_to_idx = {tuple(x): i for i, x in enumerate(coor)}


    if isinstance(h, dict):
        for i in range(len(coor)):
            loc = coor[i]
            if tuple(loc) not in weights.keys(): continue

            for j in range(neigh_coor[tuple(loc)].shape[0]):
                n_j = neigh_coor[tuple(loc)][j]
                neigh_idx = coord_to_idx[tuple(n_j)]
                A[i, neigh_idx] = np.array(weights[tuple(loc)][j] * h[tuple(loc)])

            A[i, i] = 0
            A[i, i] = - np.sum(A[i, :])
    else:
        if kernel in ['wc2', 'q_s']: h = h ** 3
        for i in range(len(coor)):
            loc = coor[i]
            if tuple(loc) not in weights.keys(): continue

            for j in range(neigh_coor[tuple(loc)].shape[0]):
                n_j = neigh_coor[tuple(loc)][j]
                neigh_idx = coord_to_idx[tuple(n_j)]
                A[i, neigh_idx] = np.array(weights[tuple(loc)][j]*h)

            A[i, i] = 0
            A[i, i] = - np.sum(A[i, :])

    A_masked = np.ma.masked_values(A, 0)
    A_sparse = csr_matrix(A)


    #kernel_plot_title = {
    #    'q_s': 'Quintic Spline',
    #    'wc2': 'Wendland C2',
    #    'gnn': 'GNN',
    #    2: r'LABFM $2^{\mathrm{nd}}$ Order',
    #    4: r'LABFM $4^{\mathrm{th}}$ Order',
    #    6: r'LABFM $6^{\mathrm{th}}$ Order',
    #    8: r'LABFM $8^{\mathrm{th}}$ Order',
    #}

    vals = np.linalg.eigvals(A)

    real_parts = vals.real
    imag_parts = vals.imag

    figsize = (3.25, 2.6) if column == "single" else (6.75, 2.8)
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        real_parts,
        imag_parts,
        s=point_size,
        alpha=0.8,
        linewidths=0,
    )

    # Axis labels with μ (not λ)
    #ax.set_xlabel(r"$\mathrm{Re}(\mu)$", fontsize=9)
    #ax.set_ylabel(r"$\mathrm{Im}(\mu)$", fontsize=9)
    ax.tick_params(labelsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_axes0:
        ax.axhline(0.0, linewidth=0.8)
        ax.axvline(0.0, linewidth=0.8)

    ax.grid(True, which="major", linewidth=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.25)
    ax.minorticks_on()

    fig.tight_layout()

    if save:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.01, transparent=True)
        plt.close(fig)
    else:
        plt.show()

mpl.rcParams.update({
    "savefig.format": "pdf",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "axes.unicode_minus": False,
})

def plot_resolving_p(
    results,
    save=False,
    filename_prefix="resolving_power",
    column="single",
    show_legend=True,
):

    from functions.res_power import resolving_power

    labels = {
        2: r"LABFM$_{p=2}$",
        4: r"LABFM ($p=4$)",
        6: r"LABFM ($p=6$)",
        8: r"LABFM ($p=8$)",
        "q_s": r"Quintic spline",
        "wc2": r"Wendland C2",
        "gnn": r"GNN$_{p=2}$(ours)",
    }

    colors = {
        2: "tab:blue",
        4: "tab:red",
        6: "tab:green",
        8: "tab:gray",
        "q_s": "tab:purple",
        "wc2": "tab:cyan",
        "gnn": "tab:orange",
    }

    figsize = (3.25, 2.6) if column == "single" else (6.75, 2.8)

    # --- Create figures once
    fig_re, ax_re = plt.subplots(figsize=figsize)
    fig_im, ax_im = plt.subplots(figsize=figsize)

    k_hat_ref = None


    for (resolution, kernel) in results.keys():
        attrs = results[resolution, kernel]
        sph = True if kernel in ['q_s', 'wc2'] else False
        res_power = resolving_power(attrs.x, attrs.s, sph, attrs._neigh_xy, k_samples=100, n_samples=1e10)

        k_hat = res_power[:, 2]
        re_keff = res_power[:, 0]
        im_keff = res_power[:, 1]

        idx = np.argsort(k_hat)
        k_hat = k_hat[idx]
        re_keff = re_keff[idx]
        im_keff = im_keff[idx]

        if k_hat_ref is None:
            k_hat_ref = k_hat
            ax_re.plot(
                k_hat_ref,
                k_hat_ref,
                linestyle="--",
                linewidth=1.2,
                color="black",
                label="Spectral accuracy",
            )

        ax_re.plot(
            k_hat,
            re_keff,
            linewidth=1.6,
            color=colors[kernel],
            label=labels[kernel],
        )

        ax_im.plot(
            k_hat,
            im_keff,
            linewidth=1.6,
            color=colors[kernel],
            label=labels[kernel],
        )

    # --- Formatting (Real part)
    ax_re.set_xlabel(r"$\omega / \omega_{\text{Ny}}$", fontsize=9)
    ax_re.set_ylabel(r"$\omega_{\text{eff}} / \omega_{\text{Ny}}$", fontsize=9)
    ax_re.tick_params(labelsize=8)
    ax_re.spines["top"].set_visible(False)
    ax_re.spines["right"].set_visible(False)
    ax_re.grid(True, linewidth=0.35)

    # --- Formatting (Imaginary part)
    ax_im.set_xlabel(r"$\omega / \omega_{\text{Ny}}$", fontsize=9)
    ax_im.set_ylabel(r"$\Im(\omega_{\text{eff}} / \omega_{\text{Ny}})$", fontsize=9)
    ax_im.tick_params(labelsize=8)
    ax_im.spines["top"].set_visible(False)
    ax_im.spines["right"].set_visible(False)
    ax_im.grid(True, linewidth=0.35)

    if show_legend:
        ax_re.legend(fontsize=7.5, frameon=False)
        ax_im.legend(fontsize=7.5, frameon=False)

    fig_re.tight_layout()
    fig_im.tight_layout()

    if save:
        fig_re.savefig(f"{filename_prefix}_real.pdf", bbox_inches="tight")
        fig_im.savefig(f"{filename_prefix}_imag.pdf", bbox_inches="tight")
        plt.close(fig_re)
        plt.close(fig_im)
    else:
        plt.show()

def plot_resolving_p_real(
    results,
    save=False,
    filename_prefix="resolving_power",
    column="single",
    show_legend=True,
    use_inset=True,                 # Laplacian inset
    zoom_y=False,                   # Laplacian log-zoom feature
    zoom_pad=1.05,
    inset_loc="upper right",
    inset_size=("35%", "25%"),
    inset_xlim=(0.5, 0.9),
    inset_ylim=(0.0, 0.02),

    use_inset_x=False,              # NEW: Gradient inset
    inset_loc_x="upper right",       # NEW
    inset_size_x=("25%", "25%"),    # NEW
    inset_xlim_x=(0.0, 0.2),        # NEW
    inset_ylim_x=(5e-3, 0.2),        # NEW
):
    from functions.res_power import resolving_power2

    if use_inset or use_inset_x:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    labels = {
        2: r"LABFM$_{p=2}$",
        4: r"LABFM ($p=4$)",
        6: r"LABFM ($p=6$)",
        8: r"LABFM ($p=8$)",
        "q_s": r"Quintic spline",
        "wc2": r"Wendland C2",
        "gnn": r"GNN$_{p=2}$(ours)",
    }

    colors = {
        2: "tab:blue",
        4: "tab:red",
        6: "tab:green",
        8: "tab:gray",
        "q_s": "tab:purple",
        "wc2": "tab:cyan",
        "gnn": "tab:orange",
    }

    figsize = (3.25, 2.6) if column == "single" else (6.75, 2.8)

    fig_x, ax_x = plt.subplots(figsize=figsize)
    fig_lap, ax_lap = plt.subplots(figsize=figsize)

    k_hat_ref = None
    x_curves = {}     # NEW: store gradient curves for inset
    lap_curves = {}

    for (resolution, kernel) in results.keys():
        attrs = results[(resolution, kernel)]
        sph_bool = kernel in ["q_s", "wc2"]

        res_power_x, res_power_lap = resolving_power2(
            w_x=attrs.x,
            w_l=attrs.laplace if not sph_bool else None,
            w_y=attrs.y if sph_bool else None,
            s=attrs.s,
            sph_bool=sph_bool,
            neigh_dist_xy=attrs._neigh_xy,
            neigh_r=attrs._neigh_r if sph_bool else None,
            rho=attrs.rho if sph_bool else None,
            neigh_coor=attrs._neigh_coor if sph_bool else None,
            n_samples=int(1e10),
        )

        # ---- X derivative modal response
        k_hat_x = res_power_x[:, 1]
        re_keff_x = res_power_x[:, 0]
        idx = np.argsort(k_hat_x)
        k_hat_x = k_hat_x[idx]
        re_keff_x = re_keff_x[idx]

        # ---- Laplacian modal response
        k_hat_l = res_power_lap[:, 1]
        re_lap_eff = res_power_lap[:, 0]
        idx2 = np.argsort(k_hat_l)
        k_hat_l = k_hat_l[idx2]
        re_lap_eff = re_lap_eff[idx2]

        # Reference curves (plotted once)
        if k_hat_ref is None:
            k_hat_ref = k_hat_x

            ax_x.plot(
                k_hat_ref, k_hat_ref,
                linestyle="--", linewidth=1.2, color="black",
                label="Spectral",
            )

            ax_lap.plot(
                k_hat_ref, (k_hat_ref ** 2),
                linestyle="--", linewidth=1.2, color="black",
                label="Spectral",
            )

        # Plot lines only
        ax_x.plot(
            k_hat_x, re_keff_x,
            linewidth=1.6,
            color=colors.get(kernel, "k"),
            label=labels.get(kernel, str(kernel)),
        )
        x_curves[kernel] = (k_hat_x, re_keff_x)   # NEW

        x_l = k_hat_l ** 2
        y_l = re_lap_eff ** 2

        ax_lap.plot(
            x_l, y_l,
            linewidth=1.6,
            color=colors.get(kernel, "k"),
            label=labels.get(kernel, str(kernel)),
        )
        lap_curves[kernel] = (x_l, y_l)

    # --- Formatting: X
    ax_x.set_xlabel(r"$\omega / \omega_{\mathrm{Ny}}$", fontsize=9)
    ax_x.set_ylabel(r"$\omega_{\mathrm{eff}} / \omega_{\mathrm{Ny}}$", fontsize=9)
    ax_x.tick_params(labelsize=8)
    ax_x.spines["top"].set_visible(False)
    ax_x.spines["right"].set_visible(False)
    ax_x.grid(True, linewidth=0.35)

    # --- Formatting: Laplacian
    ax_lap.set_xlabel(r"$\omega / \omega_{\mathrm{Ny}}$", fontsize=9)
    ax_lap.set_ylabel(r"$(\omega_{\mathrm{eff}} / \omega_{\mathrm{Ny}})^2$", fontsize=9)
    ax_lap.tick_params(labelsize=8)
    ax_lap.spines["top"].set_visible(False)
    ax_lap.spines["right"].set_visible(False)
    ax_lap.grid(True, linewidth=0.35)

    # --- Laplacian zoom feature (unchanged)
    if zoom_y and len(lap_curves) > 0:
        x_threshold = 0.8

        high_x_vals = []
        all_vals = []

        for (x_l, y_l) in lap_curves.values():
            all_vals.append(y_l[y_l > 0.0])

            mask = (x_l >= x_threshold) & (y_l > 0.0)
            if np.any(mask):
                high_x_vals.append(y_l[mask])

        all_vals = np.concatenate(all_vals)
        high_x_vals = np.concatenate(high_x_vals)

        ymin = np.min(high_x_vals)
        ymax = np.max(all_vals)

        ax_lap.set_yscale("log")
        ax_lap.set_ylim(ymin, zoom_pad * ymax)

    # --- Laplacian inset (unchanged)
    if use_inset and len(lap_curves) > 0 and (k_hat_ref is not None):
        axins = inset_axes(
            ax_lap,
            width=inset_size[0],
            height=inset_size[1],
            loc=inset_loc,
            borderpad=0.8,
        )

        for kernel, (x_l, y_l) in lap_curves.items():
            axins.plot(x_l, y_l, linewidth=1.2, color=colors.get(kernel, "k"))

        axins.plot(k_hat_ref**2, k_hat_ref**2, linestyle="--", linewidth=1.0, color="black")

        axins.set_xlim(*inset_xlim)
        axins.set_ylim(*inset_ylim)

        axins.tick_params(labelsize=7)
        axins.grid(True, linewidth=0.25)

        mark_inset(ax_lap, axins, loc1=2, loc2=4, fc="none", ec="0.4", linewidth=0.6)

    # --- NEW: Gradient inset
    if use_inset_x and len(x_curves) > 0 and (k_hat_ref is not None):
        axins_x = inset_axes(
            ax_x,
            width=inset_size_x[0],
            height=inset_size_x[1],
            loc=inset_loc_x,
            borderpad=0.8,
        )
        axins_x.set_yscale("log")

        for kernel, (xx, yy) in x_curves.items():
            axins_x.plot(xx, yy, linewidth=1.2, color=colors.get(kernel, "k"))

        # Reference in inset
        axins_x.plot(k_hat_ref, k_hat_ref, linestyle="--", linewidth=1.0, color="black")

        axins_x.set_xlim(*inset_xlim_x)
        axins_x.set_ylim(*inset_ylim_x)

        axins_x.tick_params(labelsize=7)
        axins_x.grid(True, linewidth=0.25)

        mark_inset(ax_x, axins_x, loc1=2, loc2=4, fc="none", ec="0.4", linewidth=0.6)

    if show_legend:
        ax_x.legend(fontsize=7.5, frameon=False)
        ax_lap.legend(fontsize=7.5, frameon=False)

    fig_x.tight_layout()
    fig_lap.tight_layout()

    if save:
        fig_x.savefig(f"{filename_prefix}_real_dx.pdf", bbox_inches="tight")
        fig_lap.savefig(f"{filename_prefix}_real_laplace.pdf", bbox_inches="tight")
        plt.close(fig_x)
        plt.close(fig_lap)
    else:
        plt.show()


def plot_convergence(
    results,
    derivative="dx",
    size=14,
    save=False,
    filename="convergence.pdf",
    column="single",            # "single" or "double"
    rasterize_points=False,     # True if many points
    show_legend=True,           # <--- ADD THIS
):

    # --- Collect data
    allowed = [2, 4, 6, 8, "q_s", "wc2", "gnn"]
    poly_data = {}

    for k, v in results.items():
        # k assumed to be (h_or_inv?, degree) as in your code
        poly_degree = k[1]
        if poly_degree not in allowed:
            continue

        s_value = 1.0 / float(k[0])  # keep your convention
        l2_value = getattr(v, f"{derivative}_l2")

        if poly_degree not in poly_data:
            poly_data[poly_degree] = {"s": [], "l2": []}
        poly_data[poly_degree]["s"].append(s_value)
        poly_data[poly_degree]["l2"].append(float(np.asarray(l2_value).reshape(-1)[0]))

    # --- Styling maps (no hard requirement to use these exact colors; keep if you want)
    colors = {
        2: "tab:blue",
        4: "tab:red",
        6: "tab:green",
        8: "tab:gray",
        "q_s": "tab:purple",
        "wc2": "tab:cyan",
        "gnn": "tab:orange",
    }
    labels = {
        2: r"LABFM$_{p=2}$",
        4: r"LABFM ($p=4$)",
        6: r"LABFM ($p=6$)",
        8: r"LABFM ($p=8$)",
        "q_s": r"Quintic spline",
        "wc2": r"Wendland C2",
        "gnn": r"GNN$_{p=2}$(ours)",
    }

    # --- Figure size (ICML)
    if column == "single":
        figsize = (3.25, 2.4)
    else:
        figsize = (6.75, 2.6)

    fig, ax = plt.subplots(figsize=figsize)

    # --- Plot each method/degree (sorted)
    # Prefer a deterministic order in legend
    order = [2, 4, 6, 8, "q_s", "wc2", "gnn"]
    for key in order:
        if key not in poly_data:
            continue

        s = np.asarray(poly_data[key]["s"], dtype=float)
        l2 = np.asarray(poly_data[key]["l2"], dtype=float)

        # sort so lines connect correctly
        idx = np.argsort(s)
        s, l2 = s[idx], l2[idx]

        ax.plot(
            s, l2,
            color=colors.get(key, "k"),
            linewidth=1.4,
            marker="o",
            markersize=3.5,
            label=labels.get(key, str(key)),
            rasterized=rasterize_points,
        )

        # Optional: emphasize scatter separately (often unnecessary since markers exist)
        ax.scatter(
            s, l2,
            color=colors.get(key, "k"),
            s=size,
            zorder=3,
            rasterized=rasterize_points,
        )

    # --- Axes / scales
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$s/H$", fontsize=9)
    ax.set_ylabel(r"$L_2$ norm", fontsize=9)  # clearer than "L2 norm" in a paper
    ax.tick_params(labelsize=8)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Log ticks: major + minor, mathtext formatting
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))

    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # --- Grid: light and unobtrusive
    ax.grid(True, which="major", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3)

    # --- Legend (optional)
    if show_legend:
        ax.legend(
            fontsize=7.5,
            frameon=False,
            ncol=1,
            loc="best",
            handlelength=1.8,
            borderaxespad=0.2,
        )


    fig.tight_layout()

    if save:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.01, transparent=True)
        plt.close(fig)
    else:
        plt.show()


def plot_multi_convergence():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

    def plot_convergence_multi_gnn_from_arrays(
            x,
            wendland,
            quintic_spline,
            gnn_curves,
            save=False,
            filename="convergence_gnn_multi.pdf",
            column="single",
            show_legend=True,
            plot_baselines=True,  # NEW
            rasterize_points=False,
            size=14,
    ):
        """
        ICML-quality convergence plot with shared x-axis.

        Style rules:
          - Wendland C2 color fixed to tab:cyan
          - Quintic spline color fixed to tab:purple
          - GNN: color encodes p, linestyle encodes n
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        wendland = np.asarray(wendland, dtype=float).reshape(-1)
        quintic_spline = np.asarray(quintic_spline, dtype=float).reshape(-1)

        if not (len(x) == len(wendland) == len(quintic_spline)):
            raise ValueError("x, wendland, and quintic_spline must have the same length.")

        for (p, n), y in gnn_curves.items():
            y = np.asarray(y, dtype=float).reshape(-1)
            if len(y) != len(x):
                raise ValueError(f"GNN curve (p={p}, n={n}) length {len(y)} != len(x) {len(x)}")

        # Sort by x so lines connect correctly
        idx = np.argsort(x)
        xs = x[idx]
        wendland_s = wendland[idx]
        quintic_spline_s = quintic_spline[idx]

        # Fixed colors for baselines; GNN color by p
        colors = {
            "q_s": "tab:purple",
            "wc2": "tab:cyan",
            ("gnn", 2): "tab:orange",
            ("gnn", 3): "tab:green",
        }

        # Line pattern by n
        linestyles_by_n = {
            25: "-",
            50: "-.",
            100: ":",
        }

        markers_by_n = {
            25: "o",  # circle
            50: "s",  # square
            100: "^",  # triangle
        }

        # Figure size (ICML)
        figsize = (3.25, 2.4) if column == "single" else (6.75, 2.6)
        fig, ax = plt.subplots(figsize=figsize)

        # --- Baselines (optional)
        if plot_baselines:
            ax.plot(
                xs, wendland_s,
                color=colors["wc2"],
                linewidth=1.4,
                marker="o",
                markersize=3.5,
                label=r"Wendland C2",
                rasterized=rasterize_points,
            )
            ax.scatter(
                xs, wendland_s,
                color=colors["wc2"],
                s=size,
                zorder=3,
                rasterized=rasterize_points,
            )

            ax.plot(
                xs, quintic_spline_s,
                color=colors["q_s"],
                linewidth=1.4,
                marker="o",
                markersize=3.5,
                label=r"Quintic spline",
                rasterized=rasterize_points,
            )
            ax.scatter(
                xs, quintic_spline_s,
                color=colors["q_s"],
                s=size,
                zorder=3,
                rasterized=rasterize_points,
            )

        # --- GNN curves: color by p, linestyle by n
        for (p, n) in sorted(gnn_curves.keys(), key=lambda t: (t[0], t[1])):
            y = np.asarray(gnn_curves[(p, n)], dtype=float).reshape(-1)[idx]
            c = colors.get(("gnn", p), "k")
            ls = linestyles_by_n.get(n, "-")
            mk = markers_by_n.get(n, "o")
            ax.plot(
                xs, y,
                color=c,
                linestyle=ls,
                linewidth=1.6,
                marker=mk,
                markersize=3.2,
                label=rf"GNN$_{{p={p}}}^{{{n}}}$",
                rasterized=rasterize_points,
            )

        # Axes / scales
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$s/H$", fontsize=9)
        ax.set_ylabel(r"$L_2$ norm", fontsize=9)
        ax.tick_params(labelsize=8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Log ticks
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

        # Grid
        ax.grid(True, which="major", linewidth=0.4)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3)

        # Legend outside right
        if show_legend:
            fig.subplots_adjust(right=0.65)
            ax.legend(
                fontsize=7.5,
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                handlelength=1.8,
            )

        fig.tight_layout()

        if save:
            fig.savefig(filename, bbox_inches="tight", pad_inches=0.01, transparent=True)
            plt.close(fig)
        else:
            plt.show()

    x = [0.1, 0.05263157894736842, 0.02040816326530612, 0.010101010101010102]
    wendland = [0.16567004630218568, 0.053359692339833904, 0.03966624082755235, 0.034949022146574325]
    quintic_spline = [0.4861887508937166, 0.06677455033544513, 0.022146332643423975, 0.027153522917126385]
    gnn_curves = {
        (2, 25): [0.07694, 0.01628981176660375, 0.0029017187754190443, 0.001657795730744722],
        (3, 25): [0.013798644444020134, 0.007932784519998003, 0.008539042094940134, 0.008398186692277885],
        (2, 50): [0.1666442409840651, 0.033052798247446995, 0.00488257835108097, 0.0013225614586430158],
        (3, 50): [0.014081012435824387, 0.00400636397355218, 0.0038526196683536774, 0.0037103060567052334],
        (2, 100): [0.3664808366381433, 0.0639612445605845, 0.009434345695846174, 0.0031468489905554678],
        (3, 100): [0.056321939648400765, 0.00636019809546398, 0.004075878161163876, 0.00391762909586591],
    }
    plot_convergence_multi_gnn_from_arrays(
        x, wendland, quintic_spline, gnn_curves,
        plot_baselines=False,
        save=True
    )


def plot_multi_power():
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_modal_response_dx_from_arrays(
            curves_by_pn,
            spectral_from=(2, 25),
            save=False,
            filename="modal_response_dx.pdf",
            column="single",
            show_legend=True,
            legend_spectral_only=True,
            marker_every=3,
            markersize=3.8,
            linewidth=1.6,
            # --- NEW: inset options (added only; everything else unchanged)
            use_inset_x=False,
            inset_loc_x="upper right",
            inset_size_x=("25%", "25%"),
            inset_xlim_x=(0.0, 0.3),
            inset_ylim_x=(5e-3, 0.3),
            use_log_inset_y=True,
    ):
        # Fixed styling (as requested)
        colors = {
            ("gnn", 2): "tab:orange",
            ("gnn", 3): "tab:green",
        }
        linestyles_by_n = {25: "-", 50: "-.", 100: ":"}
        markers_by_n = {25: "o", 50: "s", 100: "^"}

        if use_inset_x:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        figsize = (3.25, 2.6) if column == "single" else (6.75, 2.8)
        fig, ax = plt.subplots(figsize=figsize)

        # --- Spectral reference y=x (use k_hat from the chosen curve)
        ref = np.asarray(curves_by_pn[spectral_from], dtype=float)
        k_hat_ref = ref[:, 1]
        idx = np.argsort(k_hat_ref)
        k_hat_ref = k_hat_ref[idx]
        ax.plot(
            k_hat_ref, k_hat_ref,
            linestyle="--", linewidth=1.2, color="black",
            label="Spectral",
        )

        # --- Plot curves (color by p, linestyle/marker by n)
        x_curves = {}  # NEW: store for inset
        for (p, n) in sorted(curves_by_pn.keys(), key=lambda t: (t[0], t[1])):
            arr = np.asarray(curves_by_pn[(p, n)], dtype=float)
            y = arr[:, 0]  # Re(k_eff)
            x = arr[:, 1]  # k_hat
            idx = np.argsort(x)
            x, y = x[idx], y[idx]

            ax.plot(
                x, y,
                color=colors.get(("gnn", p), "k"),
                linestyle=linestyles_by_n.get(n, "-"),
                marker=markers_by_n.get(n, "o"),
                markevery=marker_every,
                markersize=markersize,
                linewidth=linewidth,
                label="_nolegend_" if legend_spectral_only else rf"$\mathrm{{GNN}}_{{p={p}}}^{{{n}}}$",
            )
            x_curves[(p, n)] = (x, y)  # NEW

        # --- Formatting (match your earlier style)
        ax.set_xlabel(r"$\omega / \omega_{\mathrm{Ny}}$", fontsize=9)
        ax.set_ylabel(r"$\omega_{\mathrm{eff}} / \omega_{\mathrm{Ny}}$", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linewidth=0.35)

        # --- NEW: Gradient inset (added only)
        if use_inset_x and len(x_curves) > 0:
            axins = inset_axes(
                ax,
                width=inset_size_x[0],
                height=inset_size_x[1],
                loc=inset_loc_x,
                borderpad=0.8,
            )
            if use_log_inset_y:
                axins.set_yscale("log")

            for (p, n), (xx, yy) in x_curves.items():
                axins.plot(
                    xx, yy,
                    color=colors.get(("gnn", p), "k"),
                    linestyle=linestyles_by_n.get(n, "-"),
                    linewidth=1.2,
                )

            # Reference in inset
            axins.plot(k_hat_ref, k_hat_ref, linestyle="--", linewidth=1.0, color="black")

            axins.set_xlim(*inset_xlim_x)
            axins.set_ylim(*inset_ylim_x)

            axins.tick_params(labelsize=7)
            axins.grid(True, linewidth=0.25)

            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", linewidth=0.6)

        if show_legend:
            ax.legend(fontsize=7.5, frameon=False, loc="best")

        fig.tight_layout()

        if save:
            fig.savefig(filename, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    p2_n25 = [
        [0.006494680846030477, 0.006496120244264603],
        [0.01944762084569398, 0.019488360732793808],
        [0.032291578784662725, 0.03248060122132301],
        [0.04495484075406424, 0.04547284170985222],
        [0.05736709500373613, 0.058465082198381424],
        [0.06945997482370123, 0.07145732641220093],
        [0.0811675836650568, 0.08444956690073013],
        [0.09242699773949233, 0.09744180738925934],
        [0.10317874156727531, 0.11043404787778854],
        [0.11336723222038037, 0.12342628836631775],
        [0.12294118832752539, 0.13641852140426636],
        [0.13185400026786753, 0.14941076934337616],
        [0.14006405837620653, 0.16240300238132477],
        [0.14753503641066038, 0.17539525032043457],
        [0.15423612798947448, 0.18838749825954437],
        [0.16014223418229873, 0.20137973129749298],
        [0.16523410093813062, 0.21437197923660278],
        [0.16949840554229695, 0.2273642122745514],
        [0.17292779181352244, 0.2403564602136612],
        [0.175520854274537, 0.253348708152771],
        [0.1772820720512865, 0.2663409411907196],
        [0.1782216937724902, 0.2793331742286682],
        [0.1783555752492674, 0.2923254072666168],
        [0.1777049722108976, 0.3053176701068878],
        [0.17629629085523255, 0.3183099031448364],
        [0.1741607994399728, 0.33130213618278503],
        [0.17133430459434437, 0.34429436922073364],
        [0.1678567964722878, 0.35728663206100464],
        [0.16377206730306143, 0.37027886509895325],
        [0.15912730833179442, 0.38327109813690186],
        [0.15397269059394625, 0.39626333117485046],
        [0.14836093545307738, 0.40925559401512146],
        [0.1423468813778976, 0.42224782705307007],
        [0.13598705408019154, 0.4352400600910187],
        [0.12933924793157864, 0.4482322931289673],
        [0.1224621275932305, 0.4612245559692383],
        [0.11541486011777109, 0.4742167890071869],
        [0.10825678952496422, 0.4872090220451355],
        [0.10104716812902456, 0.5002012848854065],
        [0.09384496179318338, 0.5131934881210327],
        [0.08670874976792395, 0.5261857509613037],
        [0.07969674343639649, 0.5391780138015747],
        [0.0728669508684638, 0.5521702170372009],
        [0.06627751230900562, 0.5651624798774719],
        [0.059987218107189995, 0.5781547427177429],
        [0.054056179313467366, 0.5911469459533691],
        [0.0485465218343281, 0.6041392087936401],
        [0.04352276751532204, 0.6171314120292664],
        [0.039051191899047724, 0.6301236748695374],
        [0.03519691814457581, 0.6431159377098083],
        [0.032017108550482856, 0.6561081409454346],
        [0.029549213633300195, 0.6691004037857056],
        [0.02779602669060924, 0.6820926666259766],
        [0.02671386864667241, 0.6950848698616028],
        [0.026212285412148213, 0.7080771327018738],
        [0.026168317314032484, 0.7210693359375],
        [0.026448920204275486, 0.734061598777771],
        [0.02693112724860499, 0.747053861618042],
        [0.02751382766203254, 0.7600460648536682],
        [0.028121285182882236, 0.7730383276939392],
        [0.028701509568860414, 0.7860305905342102],
        [0.029222461337935213, 0.7990227937698364],
        [0.02966787662833855, 0.8120150566101074],
        [0.030033484650279933, 0.8250072598457336],
        [0.030323819379730416, 0.8379995226860046],
        [0.030549582740952694, 0.8509917855262756],
        [0.030725446381261057, 0.8639839887619019],
        [0.030868185984368794, 0.8769762516021729],
        [0.030995076850354176, 0.8899685144424438],
        [0.031122519706989, 0.9029607176780701],
        [0.031264900473433284, 0.9159529805183411],
        [0.031433710437547564, 0.9289452433586121],
        [0.03163695997138343, 0.9419374465942383],
        [0.03187890859664493, 0.9549297094345093],
        [0.03216011023017242, 0.9679219126701355],
        [0.032477742008548215, 0.9809141755104065],
    ]

    p3n25 = [
        [0.006493238052350099, 0.006496120244264603],
        [0.019477482341440717, 0.019488360732793808],
        [0.032454927232670856, 0.03248060122132301],
        [0.045420694687447576, 0.04547284170985222],
        [0.05836939388521968, 0.058465082198381424],
        [0.07129492582852341, 0.07145732641220093],
        [0.0841902975754925, 0.08444956690073013],
        [0.0970474487190785, 0.09744180738925934],
        [0.10985709262983515, 0.11043404787778854],
        [0.12260857485039908, 0.12342628836631775],
        [0.13528975087341727, 0.13641852140426636],
        [0.1478868853497083, 0.14941076934337616],
        [0.1603845745595327, 0.16240300238132477],
        [0.17276569373739914, 0.17539525032043457],
        [0.18501137057108222, 0.18838749825954437],
        [0.1971009859006181, 0.20137973129749298],
        [0.2090122023260066, 0.21437197923660278],
        [0.22072102109703048, 0.2273642122745514],
        [0.23220186730956474, 0.2403564602136612],
        [0.24342770307512812, 0.253348708152771],
        [0.2543701679697387, 0.2663409411907196],
        [0.2649997457101281, 0.2793331742286682],
        [0.27528595565586794, 0.2923254072666168],
        [0.28519756740060603, 0.3053176701068878],
        [0.2947028363998889, 0.3183099031448364],
        [0.30376975829193414, 0.33130213618278503],
        [0.31236633930589236, 0.34429436922073364],
        [0.3204608799234878, 0.35728663206100464],
        [0.3280222687680279, 0.37027886509895325],
        [0.3350202835423174, 0.38327109813690186],
        [0.34142589572623955, 0.39626333117485046],
        [0.34721157567719463, 0.40925559401512146],
        [0.3523515947531436, 0.42224782705307007],
        [0.35682232109911427, 0.4352400600910187],
        [0.3606025058033406, 0.4482322931289673],
        [0.3636735562382083, 0.4612245559692383],
        [0.36601979355248804, 0.4742167890071869],
        [0.36762869147347643, 0.4872090220451355],
        [0.36849109380853934, 0.5002012848854065],
        [0.3686014083027836, 0.5131934881210327],
        [0.36795777481052677, 0.5261857509613037],
        [0.3665622060698184, 0.5391780138015747],
        [0.364420699728213, 0.5521702170372009],
        [0.3615433206507344, 0.5651624798774719],
        [0.35794425294343496, 0.5781547427177429],
        [0.35364182154381885, 0.5911469459533691],
        [0.3486584836575119, 0.6041392087936401],
        [0.3430207907529658, 0.6171314120292664],
        [0.3367593222552435, 0.6301236748695374],
        [0.32990859249660504, 0.6431159377098083],
        [0.32250693287299975, 0.6561081409454346],
        [0.3145963515044526, 0.6691004037857056],
        [0.3062223729795307, 0.6820926666259766],
        [0.29743386094548774, 0.6950848698616028],
        [0.28828282633813807, 0.7080771327018738],
        [0.2788242238608533, 0.7210693359375],
        [0.26911573882522916, 0.734061598777771],
        [0.259217565526189, 0.747053861618042],
        [0.24919217676625138, 0.7600460648536682],
        [0.23910408174079142, 0.7730383276939392],
        [0.22901956597108805, 0.7860305905342102],
        [0.2190064020162652, 0.7990227937698364],
        [0.2091335130236662, 0.8120150566101074],
        [0.1994705626473408, 0.8250072598457336],
        [0.19008743469694878, 0.8379995226860046],
        [0.18105355501758957, 0.8509917855262756],
        [0.17243699871632914, 0.8639839887619019],
        [0.16430332191129166, 0.8769762516021729],
        [0.15671406480229583, 0.8899685144424438],
        [0.1497248998841327, 0.9029607176780701],
        [0.14338345304192548, 0.9159529805183411],
        [0.13772690885853744, 0.9289452433586121],
        [0.1327796165017499, 0.9419374465942383],
        [0.12855101449399162, 0.9549297094345093],
        [0.12503425174725927, 0.9679219126701355],
        [0.12220585627837695, 0.9809141755104065],
    ]

    p2n50 = [
        [0.006492436882390538, 0.006496120244264603],
        [0.019402603616688455, 0.019488360732793808],
        [0.03208981523017099, 0.03248060122132301],
        [0.04440930042830697, 0.04547284170985222],
        [0.05622201260899027, 0.058465082198381424],
        [0.0673967742687021, 0.07145732641220093],
        [0.07781227941147906, 0.08444956690073013],
        [0.08735891827098342, 0.09744180738925934],
        [0.09594039217629395, 0.11043404787778854],
        [0.10347509054269344, 0.12342628836631775],
        [0.10989720666416176, 0.13641852140426636],
        [0.11515757412827635, 0.14941076934337616],
        [0.11922421116237933, 0.16240300238132477],
        [0.12208256594222122, 0.17539525032043457],
        [0.12373546173839384, 0.18838749825954437],
        [0.12420274663014157, 0.20137973129749298],
        [0.12352065827392052, 0.21437197923660278],
        [0.12174091977830012, 0.2273642122745514],
        [0.11892958802616935, 0.2403564602136612],
        [0.11516568074261192, 0.253348708152771],
        [0.11053961321185947, 0.2663409411907196],
        [0.10515147983582235, 0.2793331742286682],
        [0.09910921982700767, 0.2923254072666168],
        [0.09252671051841149, 0.3053176701068878],
        [0.08552183659306492, 0.3183099031448364],
        [0.07821459000254134, 0.33130213618278503],
        [0.07072526536781801, 0.34429436922073364],
        [0.06317283289264368, 0.35728663206100464],
        [0.055673602433742884, 0.37027886509895325],
        [0.04834035276919358, 0.38327109813690186],
        [0.04128221880040848, 0.39626333117485046],
        [0.03460586672950995, 0.40925559401512146],
        [0.02841895851018554, 0.42224782705307007],
        [0.02283777037319049, 0.4352400600910187],
        [0.018001872412530238, 0.4482322931289673],
        [0.014096540293517772, 0.4612245559692383],
        [0.011359969689096736, 0.4742167890071869],
        [0.009979381266012801, 0.4872090220451355],
        [0.009826825106085513, 0.5002012848854065],
        [0.01041513507754943, 0.5131934881210327],
        [0.011240430297774513, 0.5261857509613037],
        [0.011992530364032975, 0.5391780138015747],
        [0.012529915603860194, 0.5521702170372009],
        [0.012812150256148803, 0.5651624798774719],
        [0.012856961773460376, 0.5781547427177429],
        [0.012718032594023683, 0.5911469459533691],
        [0.012472794858819818, 0.6041392087936401],
        [0.012213020110496827, 0.6171314120292664],
        [0.012033417084231024, 0.6301236748695374],
        [0.012015945980839883, 0.6431159377098083],
        [0.012212300809962598, 0.6561081409454346],
        [0.012632282747935117, 0.6691004037857056],
        [0.013245137937687425, 0.6820926666259766],
        [0.013992642180570504, 0.6950848698616028],
        [0.014805702506285693, 0.7080771327018738],
        [0.01561732319244925, 0.7210693359375],
        [0.016369924906499928, 0.734061598777771],
        [0.017018343104841065, 0.747053861618042],
        [0.017530430319651676, 0.7600460648536682],
        [0.017886602244162317, 0.7730383276939392],
        [0.018079017829203105, 0.7860305905342102],
        [0.018110673926325288, 0.7990227937698364],
        [0.01799448415492685, 0.8120150566101074],
        [0.017752305027885017, 0.8250072598457336],
        [0.01741380405788172, 0.8379995226860046],
        [0.01701500336469725, 0.8509917855262756],
        [0.016596279205759665, 0.8639839887619019],
        [0.016199591350492877, 0.8769762516021729],
        [0.01586483377143386, 0.8899685144424438],
        [0.01562551902043922, 0.9029607176780701],
        [0.01550450648580467, 0.9159529805183411],
        [0.01551090861324441, 0.9289452433586121],
        [0.015639257151271534, 0.9419374465942383],
        [0.015871306984290824, 0.9549297094345093],
        [0.01617986581581058, 0.9679219126701355],
        [0.016533416203987322, 0.9809141755104065],
    ]

    p3n50 = [
        [0.006495908883280476, 0.006496120244264603],
        [0.019487053445330207, 0.019488360732793808],
        [0.03247569993452323, 0.03248060122132301],
        [0.04545860086509008, 0.04547284170985222],
        [0.05843017997970888, 0.058465082198381424],
        [0.07138168851358699, 0.07145732641220093],
        [0.08430044692202966, 0.08444956690073013],
        [0.09716919322869746, 0.09744180738925934],
        [0.10996555677533489, 0.11043404787778854],
        [0.1226616733311234, 0.12342628836631775],
        [0.13522395430508974, 0.13641852140426636],
        [0.1476130192643833, 0.14941076934337616],
        [0.15978379717195695, 0.16240300238132477],
        [0.1716857978057741, 0.17539525032043457],
        [0.18326355080141324, 0.18838749825954437],
        [0.19445720576783396, 0.20137973129749298],
        [0.20520328305964142, 0.21437197923660278],
        [0.21543556114329504, 0.2273642122745514],
        [0.22508608315869327, 0.2403564602136612],
        [0.23408626233263513, 0.253348708152771],
        [0.24236806341790848, 0.2663409411907196],
        [0.24986523537046965, 0.2793331742286682],
        [0.2565145690837345, 0.2923254072666168],
        [0.2622571532060647, 0.3053176701068878],
        [0.2670396008936788, 0.3183099031448364],
        [0.2708152208010185, 0.33130213618278503],
        [0.2735451066746437, 0.34429436922073364],
        [0.275199121572564, 0.35728663206100464],
        [0.2757567549436003, 0.37027886509895325],
        [0.27520783352484834, 0.38327109813690186],
        [0.2735530701936323, 0.39626333117485046],
        [0.27080443847946706, 0.40925559401512146],
        [0.26698536433088194, 0.42224782705307007],
        [0.26213073086647615, 0.4352400600910187],
        [0.25628669614138944, 0.4482322931289673],
        [0.2495103283502748, 0.4612245559692383],
        [0.2418690672861851, 0.4742167890071869],
        [0.23344002520039403, 0.4872090220451355],
        [0.22430914437522723, 0.5002012848854065],
        [0.2145702326313205, 0.5131934881210327],
        [0.20432390151251725, 0.5261857509613037],
        [0.19367643483294142, 0.5391780138015747],
        [0.18273861731688806, 0.5521702170372009],
        [0.17162455367485688, 0.5651624798774719],
        [0.160450506707668, 0.5781547427177429],
        [0.14933377732430092, 0.5911469459533691],
        [0.13839163705985644, 0.6041392087936401],
        [0.12774030063242284, 0.6171314120292664],
        [0.11749388628540787, 0.6301236748695374],
        [0.10776324783291222, 0.6431159377098083],
        [0.09865446903349137, 0.6561081409454346],
        [0.09026669370924052, 0.6691004037857056],
        [0.08268885962831744, 0.6820926666259766],
        [0.07599490485025324, 0.6950848698616028],
        [0.07023728980318482, 0.7080771327018738],
        [0.06543940522156157, 0.7210693359375],
        [0.061588585212867086, 0.734061598777771],
        [0.058632477327552064, 0.747053861618042],
        [0.05648143364207435, 0.7600460648536682],
        [0.05501773892172228, 0.7730383276939392],
        [0.05410965215383457, 0.7860305905342102],
        [0.053626301463223405, 0.7990227937698364],
        [0.05344968777049228, 0.8120150566101074],
        [0.05348188772776292, 0.8250072598457336],
        [0.05364752712094065, 0.8379995226860046],
        [0.05389273917444453, 0.8509917855262756],
        [0.05418206258465331, 0.8639839887619019],
        [0.054494469557424305, 0.8769762516021729],
        [0.054819320751685094, 0.8899685144424438],
        [0.05515271131996189, 0.9029607176780701],
        [0.05549444453634679, 0.9159529805183411],
        [0.055845726834173846, 0.9289452433586121],
        [0.056207588247618914, 0.9419374465942383],
        [0.056579971718728375, 0.9549297094345093],
        [0.05696139192142096, 0.9679219126701355],
        [0.0573490360003591, 0.9809141755104065],
    ]

    p2n100 = [
        [0.006490669026222246, 0.006496120244264603],
        [0.019331238786608244, 0.019488360732793808],
        [0.03175375628622998, 0.03248060122132301],
        [0.043493544981826976, 0.04547284170985222],
        [0.05430644513860455, 0.058465082198381424],
        [0.06397600004617175, 0.07145732641220093],
        [0.07231966931500769, 0.08444956690073013],
        [0.07919385706229921, 0.09744180738925934],
        [0.08449758839386529, 0.11043404787778854],
        [0.0881747193642462, 0.12342628836631775],
        [0.0902146214493913, 0.13641852140426636],
        [0.09065133924591569, 0.14941076934337616],
        [0.08956127731819673, 0.16240300238132477],
        [0.08705952660518357, 0.17539525032043457],
        [0.08329499049055622, 0.18838749825954437],
        [0.07844451372340953, 0.20137973129749298],
        [0.07270625243400845, 0.21437197923660278],
        [0.06629254963486353, 0.2273642122745514],
        [0.05942259767489733, 0.2403564602136612],
        [0.052315178018826, 0.253348708152771],
        [0.04518177199367915, 0.2663409411907196],
        [0.03822033908880875, 0.2793331742286682],
        [0.03161007212115831, 0.2923254072666168],
        [0.025507478668269988, 0.3053176701068878],
        [0.020044228331433562, 0.3183099031448364],
        [0.015327328309463664, 0.33130213618278503],
        [0.011442004481501442, 0.34429436922073364],
        [0.008455226219968736, 0.35728663206100464],
        [0.006406732927628657, 0.37027886509895325],
        [0.005256461274487337, 0.38327109813690186],
        [0.0048134382011348524, 0.39626333117485046],
        [0.0048063420689714595, 0.40925559401512146],
        [0.005058433654418412, 0.42224782705307007],
        [0.005524839102101492, 0.4352400600910187],
        [0.0062086105113275895, 0.4482322931289673],
        [0.007086537269652208, 0.4612245559692383],
        [0.008092283049669685, 0.4742167890071869],
        [0.009134896414893103, 0.4872090220451355],
        [0.010120711949786683, 0.5002012848854065],
        [0.010966897136210506, 0.5131934881210327],
        [0.011607954321197891, 0.5261857509613037],
        [0.0119984136514498, 0.5391780138015747],
        [0.012113642648916475, 0.5521702170372009],
        [0.011949641817774932, 0.5651624798774719],
        [0.011522235848151569, 0.5781547427177429],
        [0.01086593725626318, 0.5911469459533691],
        [0.01003274708469843, 0.6041392087936401],
        [0.009091107623888987, 0.6171314120292664],
        [0.008124853782456741, 0.6301236748695374],
        [0.007230666386187403, 0.6431159377098083],
        [0.0065093666487956335, 0.6561081409454346],
        [0.00604351301474878, 0.6691004037857056],
        [0.00586265530835246, 0.6820926666259766],
        [0.005923557238681119, 0.6950848698616028],
        [0.006132817157721269, 0.7080771327018738],
        [0.006391770156034321, 0.7210693359375],
        [0.006626379403877849, 0.734061598777771],
        [0.006795042278543811, 0.747053861618042],
        [0.006885318010244583, 0.7600460648536682],
        [0.0069076811115489824, 0.7730383276939392],
        [0.006888691374561832, 0.7860305905342102],
        [0.006863413197064354, 0.7990227937698364],
        [0.006866716245815136, 0.8120150566101074],
        [0.0069242980936097724, 0.8250072598457336],
        [0.007045893613229579, 0.8379995226860046],
        [0.007223409555913385, 0.8509917855262756],
        [0.007434621019464828, 0.8639839887619019],
        [0.007650250682969354, 0.8769762516021729],
        [0.007841291376680808, 0.8899685144424438],
        [0.007984554371709203, 0.9029607176780701],
        [0.008065968072959158, 0.9159529805183411],
        [0.008081979377285097, 0.9289452433586121],
        [0.008039524777791442, 0.9419374465942383],
        [0.007954835450189586, 0.9549297094345093],
        [0.007851110890421148, 0.9679219126701355],
        [0.007755007727648637, 0.9809141755104065],
    ]

    p3n100 = [
        [0.006497429729859086, 0.006496120244264603],
        [0.019489742611216235, 0.019488360732793808],
        [0.03247252901570555, 0.03248060122132301],
        [0.0454332513242164, 0.04547284170985222],
        [0.05835044641330283, 0.058465082198381424],
        [0.07119084885765448, 0.07145732641220093],
        [0.08390717164885485, 0.08444956690073013],
        [0.09643668182517995, 0.09744180738925934],
        [0.1087006722770412, 0.11043404787778854],
        [0.12060488921707596, 0.12342628836631775],
        [0.13204092919672591, 0.13641852140426636],
        [0.1428885722963564, 0.14941076934337616],
        [0.15301897160750821, 0.16240300238132477],
        [0.16229857577926335, 0.17539525032043457],
        [0.17059362347272158, 0.18838749825954437],
        [0.1777750180076748, 0.20137973129749298],
        [0.18372336882941565, 0.21437197923660278],
        [0.1883339747277128, 0.2273642122745514],
        [0.19152152255131566, 0.2403564602136612],
        [0.1932242845141971, 0.253348708152771],
        [0.19340761662461536, 0.2663409411907196],
        [0.19206658936782917, 0.2793331742286682],
        [0.1892276182148693, 0.2923254072666168],
        [0.18494900414851786, 0.3053176701068878],
        [0.17932034126241395, 0.3183099031448364],
        [0.17246079748320056, 0.33130213618278503],
        [0.1645163233670325, 0.34429436922073364],
        [0.15565589047486858, 0.35728663206100464],
        [0.14606690279248608, 0.37027886509895325],
        [0.1359499598033542, 0.38327109813690186],
        [0.12551317586447758, 0.39626333117485046],
        [0.11496627497718906, 0.40925559401512146],
        [0.10451467990281155, 0.42224782705307007],
        [0.09435379612590776, 0.4352400600910187],
        [0.0846636501442733, 0.4482322931289673],
        [0.07560397495242432, 0.4612245559692383],
        [0.06730974754943893, 0.4742167890071869],
        [0.0598870987294022, 0.4872090220451355],
        [0.05340950101344935, 0.5002012848854065],
        [0.04791431493873617, 0.5131934881210327],
        [0.043400248391308065, 0.5261857509613037],
        [0.039826964776709814, 0.5391780138015747],
        [0.037118428239240234, 0.5521702170372009],
        [0.035170806922900665, 0.5651624798774719],
        [0.03386376712957529, 0.5781547427177429],
        [0.03307211170743716, 0.5911469459533691],
        [0.03267465231130903, 0.6041392087936401],
        [0.03255904965131838, 0.6171314120292664],
        [0.03262348413791403, 0.6301236748695374],
        [0.03277696995394815, 0.6431159377098083],
        [0.03293979189338129, 0.6561081409454346],
        [0.03304467660085327, 0.6691004037857056],
        [0.033038583821652305, 0.6820926666259766],
        [0.0328846552959605, 0.6950848698616028],
        [0.032563813556424605, 0.7080771327018738],
        [0.032075597194544725, 0.7210693359375],
        [0.03143792900960821, 0.734061598777771],
        [0.03068558287096479, 0.747053861618042],
        [0.029867147357197244, 0.7600460648536682],
        [0.029040333697429106, 0.7730383276939392],
        [0.028265642850158874, 0.7860305905342102],
        [0.02759880367609687, 0.7990227937698364],
        [0.027083034821272958, 0.8120150566101074],
        [0.026742824421172935, 0.8250072598457336],
        [0.0265810535260306, 0.8379995226860046],
        [0.026580497021491383, 0.8509917855262756],
        [0.026709238376163572, 0.8639839887619019],
        [0.02692819620703247, 0.8769762516021729],
        [0.02719855480393392, 0.8899685144424438],
        [0.027487440649137926, 0.9029607176780701],
        [0.02777112677685957, 0.9159529805183411],
        [0.028035832332993543, 0.9289452433586121],
        [0.02827662182452605, 0.9419374465942383],
        [0.028495067629027274, 0.9549297094345093],
        [0.028696351508355845, 0.9679219126701355],
        [0.02888642516522458, 0.9809141755104065],
    ]

    curves = {
        (2, 25): p2_n25,
        (3, 25): p3n25,
        (2, 50): p2n50,
        (3, 50): p3n50,
        (2, 100): p2n100,
        (3, 100): p3n100,
    }
    plot_modal_response_dx_from_arrays(curves, show_legend=True, save=True, marker_every=5, linewidth=1.2,
                                       use_inset_x=True, use_log_inset_y=False)


def plot_error(log_x=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

    paths = {
        r"GNN$_{p=2}$, $s=1/70$": "/home/lucas/phd/Fortran/sunset_code/paraview_files/tg/gnn/l2_1by70.out",
        r"GNN$_{p=2}$, $s=1/140$": "/home/lucas/phd/Fortran/sunset_code/paraview_files/tg/gnn_hres/l2_1by140.out",
    }

    data = {}
    for label, pth in paths.items():
        arr = np.loadtxt(pth)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"{pth} does not look like a 2-column file.")
        t = arr[:, 0].astype(float)
        e = arr[:, 1].astype(float)
        idx = np.argsort(t)
        data[label] = (t[idx], e[idx])

    figsize = (3.25, 2.4)
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["tab:orange", "tab:green"]
    for (label, (t, e)), c in zip(data.items(), colors):
        ax.plot(t, e, linewidth=1.5, color=c, label=label)

    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$t^*$", fontsize=9)
    ax.set_ylabel(r"$L_2$ error", fontsize=9)
    ax.tick_params(labelsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_formatter(NullFormatter())

    if log_x:
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_locator(plt.NullLocator())  # optional but clean

    ax.grid(True, which="major", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3)

    ax.legend(fontsize=7.2, frameon=False, loc="lower right", handlelength=1.8)

    fig.tight_layout()
    out_path = "l2_gnn_p2_convergence.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
