module Figure7

using Figure6, Plot, GAPlot, SimpleStats, UCDColors, RelayGLM.RelayUtils
using PyPlot, KernelDensity, StatsBase, Bootstrap, Printf, Statistics

# ============================================================================ #
collate_data() = Figure6.collate_data(RRI, 0.1)
# ============================================================================ #
function make_figure(d::Dict{String,Any})

    row_height = [1.0]
    row_spacing = [0.12, 0.14]
    col_width = [1.0, 0.33]
    col_spacing = [0.12, 0.1, 0.03]

    hf = figure()
    hf.set_size_inches((7,5))
    ax = axes_layout(hf, row_height=row_height, row_spacing=row_spacing,
        col_width=col_width, col_spacing=col_spacing)

    foreach(default_axes, ax)

    colors = Dict("grating"=>RED, "msequence"=>BLUE)
    labels = ["Binary white noise", "Gratings"]

    ids = sort(intersect(d["msequence"]["ids"], d["grating"]["ids"]))
    absd = zeros(length(ids), 2)

    len = size(d["grating"]["xf_q1"], 1)
    kt = 1:len #len-30:len

    offset = [2, 5]

    for (k, lab) in enumerate(["msequence", "grating"])

        println(uppercase(lab))

        hi = d[lab]["xf_q4"]
        lo = d[lab]["xf_q1"]

        ad = assess(hi[kt,:], lo[kt,:], ird)
        ac = kde_lscv(ad, boundary=(-0.02, 0.2))

        e = range(0.0, 0.2, length=16)
        hst = fit(Histogram, ad, e)
        c = hst.weights ./ (sum(hst.weights) * step(e))

        x, y = hist_points(e, c)
        ax[1].plot(x, y, color=colors[lab], linewidth=1)

        path = matplotlib.path.Path(hcat(x, y))
        dist = matplotlib.patches.PathPatch(path, color=colors[lab], alpha=0.5)
        ax[1].add_patch(dist)

        y = ac.density

        ax[1].plot(ac.x, y, linewidth=3, color=colors[lab], zorder=4, label=labels[k] * " (N=$(size(hi, 2)))")

        m = median(ad)
        kmn = findfirst(>(m), e) - 1

        ax[1].plot(m, c[kmn] + offset[k], "v", markersize=14, color=colors[lab])

        val, l, h = GAPlot.distribution(bmedian(ad), ax[2], x=0.0, y=0, xoffset=0, draw_axis=false, color=colors[lab], sep=0.02, alpha=0.8)

        @printf("    Abs difference | median: %.3f (MAD %.3f, 95%% CI [%.3f, %.3f])\n", val, SimpleStats.mad(ad), l, h)

        ks = map(x->findfirst(isequal(x), d[lab]["ids"]), ids)
        absd[:,k] = ad[ks]
    end

    ax[1].legend(frameon=false, fontsize=14, loc="upper right", bbox_to_anchor=(1.15, 1.0))

    ax[1].set_xlabel(L"\int{|\mathrm{Q1} - \mathrm{Q4}|}", fontsize=14)
    ax[1].set_ylabel("Density (A.U.)", fontsize=14)
    ax[1].set_ylim(0, 32)
    xl = ax[1].get_xlim()
    ax[1].set_xlim(0.0-(xl[2]-xl[1])*0.12, xl[2])

    ax[1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))

    xl = ax[2].get_xlim()
    ax[2].set_xlim(-0.02, xl[2])

    yl = ax[2].get_ylim()
    ax[2].set_ylim(0, yl[2])

    ax[2].set_xticklabels([])
    ax[2].set_xlabel("Probability", fontsize=14)


    ax[2].set_ylabel("Median \$\\int{|\\mathrm{Q1} - \\mathrm{Q4}|}\$", fontsize=14)
    ax[2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))

    for (k, label) in enumerate(["A","B"])
        Plot.axes_label(hf, ax[k], label)
    end

    println()

    v, p = paired_permutation_test(median, absd[:,2], absd[:,1])
    tmp = absd[:,2] .- absd[:,1]
    v, lo, hi = confint(bmedian(tmp), BCaConfInt(0.95), 1)

    @printf("Abs difference pptest (grating - mseq): N = %d, %.3f (MAD %.3f 95%% CI [%.3f, %.3f] p = %.5f)\n", length(tmp), v, SimpleStats.mad(tmp), lo, hi, p)

    return ax[1]
end
# ============================================================================ #
ird(x::Vector{<:Real}, y::Vector{<:Real}) = RelayUtils.trapz(abs.(x .- y)) .* 0.001
# ============================================================================ #
function assess(hi::Matrix{<:Real}, lo::Matrix{<:Real}, f::Function)
   r = zeros(size(hi, 2))
   for k in eachindex(r)
       r[k] = f(hi[:,k], lo[:,k])
   end
   return r
end
# ============================================================================ #
function hist_points(e, c)
    x = Vector{Float64}(undef, 0)
    y = Vector{Float64}(undef, 0)
    append!(x, [e[1], e[1]])
    append!(y, [0, c[1]])
    for k = 1:length(e)-1
        append!(x, [e[k], e[k+1]])
        append!(y, [c[k], c[k]])
        if k == length(c)
            append!(x, [e[k+1], e[k+1]])
            append!(y, [c[k], 0])
        else
            append!(x, [e[k+1], e[k+1]])
            append!(y, [c[k], c[k+1]])
        end
    end

    return x, y
end
# ============================================================================ #
end