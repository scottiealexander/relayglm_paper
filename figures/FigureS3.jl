module FigureS3

using Figure7, Figure6, RelayGLM, UCDColors
using PyPlot, PyCall, Plot

const DSet = Dict{String,Any}
# ============================================================================ #
function collate_data()
    return Figure6.collate_data(RRI, 0.250), Figure6.collate_data(RRI, 0.125)
end
# ============================================================================ #
function make_figure(data::Tuple{DSet,DSet}, color_scheme::String="grwhpu")
    h = figure()
    h.set_size_inches((9,9.5))

    rh = [1.0, 1.0]
    rs = [0.1, 0.17, 0.06]
    cw = [1.0, 1.0]
    cs = [0.1, 0.12, 0.03]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)

    len = size(data[1]["grating"]["xf_q1"], 1)
    t = range(-len*0.001, -0.001, length=len)

    sax = Vector{PyCall.PyObject}(undef, 4)
    for k in eachindex(ax)
        default_axes(ax[k])
        if mod(k, 2) == 0
            pos = [0.16, 0.47, 0.35, 0.55]
        else
            pos = [0.35, 0.45, 0.35, 0.55]
        end
        sax[k] = Figure6.add_subplot_axes(ax[k], pos)
        default_axes(sax[k])
        sax[k].set_yticklabels([])
        ki = length(t)-30
        sax[k].plot([t[ki], t[end]], [0,0], "--", color="black", linewidth=1)
        sax[k].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    end

    if color_scheme == "grbkpu"
        col = [GREEN, [.3, .3, .3], [0., 0., 0.], PURPLE]
    elseif color_scheme == "grwhpu"
        col = [GREEN, LIGHTGREEN, LIGHTPURPLE, PURPLE]
    elseif color_scheme == "full"
        col = [GREEN, ORANGE, GOLD, PURPLE]
    else
        error(color_scheme * " is not a valid color scheme")
    end

    titles = Dict("msequence"=>"Binary white noise", "grating"=>"Gratings")
    colors = Dict("msequence"=>BLUE, "grating"=>RED)

    kax = 1
    for d in data
        for (k, typ) in enumerate(["msequence", "grating"])
            ax[kax].plot([t[1], t[end]], [0,0], "--", color="black", linewidth=2)

            mn = +Inf
            mx = -Inf

            for q in 1:4
                name = "q" * string(q)
                mnt, mxt = Figure6.filter_plot(d, t, typ, name, ax[kax], sax[kax], col[q], 30)

                mn = min(mn, mnt)
                mx = max(mx, mxt)
            end

            Figure6.inset_box(t, mn, mx, ax[kax], 30)

            if k == 1
                ax[kax].legend(frameon=false, fontsize=14, loc="upper left")
            end

            Figure6.format_filter_plot(ax[kax])

            ax[kax].set_title(titles[typ], fontsize=18, color=colors[typ])

            kax += 1
        end
    end

    labels = ["A","B","C","D"]

    for (k, label) in zip(1:4, labels)
        Plot.axes_label(h, ax[k], label)
    end

    h.text(0.5, 0.995, "Spike classification window size: 250ms", fontsize=24, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.495, "Spike classification window size: 125ms", fontsize=24, horizontalalignment="center", verticalalignment="top")

    return sax[2]
end
# ============================================================================ #
end
