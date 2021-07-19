module Figure6B
using Figure6, UCDColors, GAPlot, Plot, PyPlot

function make_figure(d::Dict{String,Any})

    h = figure()
    h.set_size_inches((9,6))

    rh = [1.0, 0.33]
    rs = [0.15, 0.1, 0.1]
    cw = [1.0, 0.6]
    cs = [0.1, 0.12, 0.03]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)

    ax[3].remove()
    Figure6.bottom_align(ax[1], ax[4])
    deleteat!(ax, 3)

    foreach(default_axes, ax)

    xv = [0.0, 2.0]

    sax = Figure6.plot_one(d, "awake", ax, [GREEN, PURPLE], xv, inset_length=30, yloc=0.1, ci=false, labels=["Low activity", "High activity"])

    bbox = sax.get_position()
    p0 = bbox.p0 .- [.1, .07]
    p1 = bbox.p1 .- [.05, .15]
    bbox.update_from_data_xy([p0, p1], ignore=true)
    sax.set_position(bbox)
    sax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))

    ax[1].legend(frameon=false, fontsize=14, loc="upper center")
    ax[1].set_xlim(ax[1].get_xlim().*0.97)

    Figure6.format_filter_plot(ax[1], 0.1)

    ax[2] = remove_children(h, ax[2])
    ax[3] = remove_children(h, ax[3])

    v, lo, hi = GAPlot.cumming_plot(d["awake"]["rri_q1"], d["awake"]["rri_q2"], ax=ax[[2,3]], colors=[GREEN, PURPLE], dcolor=GOLD)

    ax[2].set_ylim(0, ax[2].get_ylim()[2])

    ax[2].set_ylabel(L"\mathcal{I}_{Bernoulli}", fontsize=14)
    ax[2].set_xticklabels(["Low\nactivity", "High\nactivity"], fontsize=14)

    ax[3].set_ylabel("Paired median\ndifference", fontsize=14)

    foreach((x,l) -> Plot.axes_label(h, x, l), ax[1:2], ["A","B"])

    h.text(0.5, 0.995, "Awake", fontsize=24, color=GOLD, horizontalalignment="center", verticalalignment="top")

    return h
end

function remove_children(h, ax)
    pos = ax.get_position()
    ax.remove()
    return default_axes(h.add_axes(pos))
end

end
