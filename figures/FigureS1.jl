module FigureS1

using Figure3, PaperUtils

using PyPlot, ColorTypes, Plot, UCDColors

const DSet = Dict{String,Any}
# ============================================================================ #
function collate_data()
    d1 = Figure3.collate_data(true, 0.004, 0.1)
    d2 = Figure3.collate_data(true, 0.006, 0.05)
    return (d1, d2)
end
# ============================================================================ #
function make_figure(data::Tuple{DSet,DSet})

    h = figure()
    h.set_size_inches((10,8))

    rh = [1.0, 1.0]
    rs = [0.12, 0.17, 0.07]
    cw = [1.0, 1.0]
    cs = [0.08, 0.05, 0.02]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)

    foreach(default_axes, ax)

    names = Dict("msequence"=>"Binary white noise", "grating"=>"Gratings")
    colors = Dict("msequence"=>BLUE, "grating"=>RED)
    titles = ["Deadtime \$\\geq\$ 100ms, ISI \$\\leq\$ 4ms","Deadtime \$\\geq\$ 50ms, ISI \$\\leq\$ 6ms"]
    title_loc = [0.99, 0.5]

    t_xf = range(-0.2, -0.001, length=200)
    t_hf = range(-0.2, -0.001, length=200)

    kax = [1,2]
    for (k, d) in enumerate(data)
        for type in ["msequence", "grating"]
            N = size(d[type]["xf"], 2)

            # population CH-retina plot
            y, lo, hi = filter_ci(PaperUtils.normalize(d[type]["xf"]))
            plot_with_error(t_xf, y, lo, hi, RGB(colors[type]...), ax[kax[1]], linewidth=3, label=names[type] * "(n=$(N))")

            ax[kax[1]].plot([t_xf[1], -.001], [0,0], ":", color="gray", linewidth=2)

            # population CH-LGN plot
            y, lo, hi = filter_ci(PaperUtils.normalize(d[type]["hf"]))
            plot_with_error(t_hf, y, lo, hi, RGB(colors[type]...), ax[kax[2]], linewidth=3, label=names[type] * " (n=$(N))")

            ax[kax[2]].plot([t_hf[1], -.001], [0,0], ":", color="gray", linewidth=2)
        end
        h.text(0.5, title_loc[k], titles[k], fontsize=20, horizontalalignment="center", verticalalignment="top", figure=h)
        kax .+= 2
    end

    foreach(x->x.set_ylim(-0.15, 0.4), ax)
    ax[2].set_yticklabels([])
    ax[4].set_yticklabels([])

    for k in 1:4
        if mod(k, 2) == 1
            ax[k].legend(frameon=false, fontsize=14)
        end
        if k == 1 || k == 3
            ax[k].set_ylabel("Filter weight (A.U.)", fontsize=14)
        end
        if 2 < k < 5
            ax[k].set_xlabel("Time before spike (seconds)", fontsize=14)
        end
    end

    Plot.axes_label(h, ax[1], "A")
    Plot.axes_label(h, ax[2], "B", -0.23)
    Plot.axes_label(h, ax[3], "C")
    Plot.axes_label(h, ax[4], "D", -0.23)

    ax[1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))
    ax[3].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))

    ax[2].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))
    ax[4].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))
end
# ============================================================================ #
end
