module FigureS9

# reviewer response figure 3

using StatsBase, PyPlot, KernelDensity, Printf
using Figure6_7, Figure9, UCDColors, Plot

collate_data() = Figure6_7.collate_data()

function make_figure(d::Dict)

    h = figure()
    h.set_size_inches((9,8))

    rh = [1.0, 1.0, 1.0]
    rs = [0.08, 0.1, 0.1, 0.08]
    cw = [1.0, 1.0, 1.0]
    cs = [0.1, 0.08, 0.08, 0.05]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)
    ax = permutedims(reshape(ax, 3, 3), (2,1))

    foreach(default_axes, ax)

    colors = Dict{String,Any}("grating"=>RED, "msequence"=>BLUE, "awake"=>GOLD)

    edges = -0.05:0.05:0.5

    for (k, stim) in enumerate(["grating", "msequence", "awake"])
        @printf("%s:\n", uppercase(stim))
        for (j, model) in enumerate(["isi", "ff", "fr"])
            @printf("  %s (mean, median, min, max):\n", uppercase(model))
            rri = Vector{Float64}(d[stim]["rri"][model])

            hst = fit(Histogram, rri, edges)

            ac = kde_lscv(rri, boundary=(-0.5, 1.0))

            c = hst.weights ./ (sum(hst.weights) * step(edges))

            x, y = Figure9.hist_points(edges, c)
            ax[k,j].plot(x, y, color=colors[stim], linewidth=1)

            path = matplotlib.path.Path(hcat(x, y))
            dist = matplotlib.patches.PathPatch(path, color=colors[stim], alpha=0.5)
            ax[k,j].add_patch(dist)

            if stim != "awake"
                ax[k,j].plot(ac.x, ac.density, linewidth=3, color=colors[stim], zorder=4)
            end

            ax[k,j].set_xlim(-0.06, 0.51)

            yl = ax[k,j].get_ylim()
            md = median(rri)
            ax[k,j].plot([md, md], yl, "--", color="grey", linewidth=2, label="median")

            @printf("\t%0.3f & %0.3f & %0.3f & %0.3f\n\n", mean(rri), md, minimum(rri), maximum(rri))
        end
    end

    ax[3,3].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

    ax[1,1].legend(frameon=false, fontsize=14)

    ax[1,1].set_title("ISI model", fontsize=16)
    ax[1,2].set_title("Retinal model", fontsize=16)
    ax[1,3].set_title("Combined model", fontsize=16)

    foreach(ax[:,1]) do cax
        cax.set_ylabel("# of pairs", fontsize=14)
    end

    foreach(ax[3,:]) do cax
        cax.set_xlabel(L"\mathcal{I}_{\mathrm{Bernoulli}}", fontsize=14)
    end

    h.text(0.5, 0.99, "Gratings", fontsize=18, color=RED, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.65, "Binary white noise", fontsize=18, color=BLUE, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.33, "Awake", fontsize=18, color=GOLD, horizontalalignment="center", verticalalignment="top")

    labels = ["A","B","C","D","E","F","G","H","I"]
    foreach((cax,lab)->Plot.axes_label(h, cax, lab), permutedims(ax, (2,1)), labels)

    return h, ax
end

centers(x::StepRangeLen) = x[1:end-1] .+ (step(x)/2)

end
