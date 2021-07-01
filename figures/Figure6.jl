module Figure6

using DatabaseWrapper, RelayGLM, Progress, SpkCore, SimpleStats
using PaperUtils, UCDColors, Plot, GAPlot
using Statistics, Printf, Bootstrap, PyPlot, ColorTypes, KernelDensity, StatsBase

using RelayGLM.RelayUtils
import Distributions
import Dates

const Strmbol = Union{String,Symbol}
const FFSPAN = 200

# ============================================================================ #
"""
`collate_data(::Type{T}, twin::Real=0.1) where T <: RelayGLM.PerformanceMetric`

`twin` - the duration of the time window preceding each target spike over
which LGN spikes are counted for partitioning
"""
function collate_data(::Type{T}, twin::Real=0.1) where T <: RelayGLM.PerformanceMetric

    NQUARTILE = 3
    bin_size = 0.001

    d = Dict{String, Any}()
    tmp = Dict{String,Strmbol}("grating" => "(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)

    key = RelayGLM.key_name(T)

    for (type, ptrn) in tmp
        exc = copy(PaperUtils.EXCLUDE[type])
        if type == "msequence"
            push!(exc, 102)
        end
        db = get_database(ptrn, id -> !in(id, exc))
        d[type] = Dict{String, Any}()
        d[type]["ids"] = get_ids(db)

        for q in 1:NQUARTILE
            name = "q" * string(q)
            d[type]["xf_" * name] = Matrix{Float64}(undef, FFSPAN, length(db))
            d[type]["efficacy_" * name] = Vector{Float64}(undef, length(db))
            d[type]["contribution_" * name] = Vector{Float64}(undef, length(db))
            d[type][key * "_" * name] = init_output(T, length(db))
            d[type]["nlli_" * name] = Vector{Float64}(undef, length(db))

            if q == 1
                d[type]["nret"] = Vector{Float64}(undef, length(db))
                d[type]["nlgn"] = Vector{Float64}(undef, length(db))
            end
        end

        show_progress(0.0, 0, "$(type): ", "(0 of $(length(db)))")

        for k in 1:length(db)
            t1 = time()
            ret, lgn, _, _ = get_data(db, k)

            kq = rate_split(ret, lgn, round(Int, twin / bin_size))

            for q in 1:NQUARTILE
                name = "q" * string(q)
                kspk = kq[q]
                ef, cn, nret, nlgn = stats(ret, lgn, kspk, FFSPAN)
                res = run_one(T, kspk, ret, lgn, bin_size)
                d[type]["xf_" * name][:,k] = get_coef(res, :ff)
                d[type]["efficacy_" * name][k] = ef
                d[type]["contribution_" * name][k] = cn
                d[type][key * "_" * name][k] = mean(res.metric)
                d[type]["nlli_" * name][k] = res.nlli

                if q == 1
                    d[type]["nret"][k] = nret
                    d[type]["nlgn"][k] = nlgn
                end
            end

            elap = time() - t1
            show_progress(k/length(db), 0, "$(type): ", "($(k) of $(length(db)) @ $(elap))")
        end
        println()
    end
    return d
end
# ============================================================================ #
function make_figure(d; show_inset::Bool=true, color_scheme::String="grwhpu", io::IO=stdout)

    N = length(filter(x->match(r"xf_q\d", x) != nothing, keys(d["grating"])))

    h = figure()
    h.set_size_inches((9,10.0))

    rh = [1.0, 0.33, 1.0, 0.33, 1.0, 0.33]
    rs = [0.09, 0.025, 0.15, 0.025, 0.15, 0.025, 0.06]
    cw = [1.0, 1.0]
    cs = [0.1, 0.12, 0.03]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)
    foreach(x->x.remove(), ax[[3,7,11]])
    bottom_align(ax[1], ax[4])
    bottom_align(ax[5], ax[8])
    bottom_align(ax[9], ax[12])

    deleteat!(ax, [3, 7, 11])

    foreach(default_axes, ax)

    len = size(d["grating"]["xf_q1"], 1)
    t = range(-len*0.001, -0.001, length=len)

    if color_scheme == "grbkpu"
        col = [GREEN, [.3, .3, .3], [0., 0., 0.], PURPLE]
    elseif color_scheme == "grwhpu"
        col = [GREEN, LIGHTGREEN, LIGHTPURPLE, PURPLE]
    elseif color_scheme == "full"
        # col = [GREEN, ORANGE, GOLD, PURPLE]
        col = [GREEN, [.3, .3, .3], PURPLE]
    else
        error(color_scheme * " is not a valid color scheme")
    end

    xmn, xmx = N == 4 ? (-1.5, 8.0) : (-1.0, 5.5)
    foreach(x->x.set_xlim(xmn, xmx), ax[[2,3,5,6,8,9]])

    titles = ["Binary white noise", "Gratings", "Awake"]
    title_color = [BLUE, RED, GOLD]
    if N == 4
        xv = [0.0, 2.25, 4.0, 6.25]
    else
        xv = [0.0, 2.0, 4.0]
    end
    inset_length = show_inset ? 30 : 0

    for (k,typ) in enumerate(["msequence", "grating", "awake"])

        kax = (k-1)*3+1
        plot_one(d, typ, ax[kax:kax+2], col, xmn, xmx, xv, inset_length=inset_length, yloc=0.1)

        if k == 1
            ax[kax].legend(frameon=false, fontsize=14, loc="upper left")
        end

        format_filter_plot(ax[kax], 0.1)

        ax[kax+1].set_xticks(xv)
        ax[kax+1].set_xticklabels([])
        ax[kax+1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax[kax+1].set_ylabel(L"\mathcal{I}_{Bernoulli}", fontsize=14)

        ymx = max(0.2, ax[kax+2].get_ylim()[2])
        yts = ymx <= 0.5 ? 0.1 : 0.3
        ax[kax+2].set_ylim(-0.02, ymx)
        ax[kax+2].set_xticks(xv)
        ax[kax+2].set_xticklabels(map(x->"Q"*string(x), 1:N), fontsize=14)
        ax[kax+2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(yts))
        ax[kax+2].set_ylabel("Median\n\$\\mathcal{I}_{Bernoulli}\$", fontsize=14)
    end

    labels = ["A","B","C","D","E","F"]

    for (k, label) in zip([1,2,4,5,7,8], labels)
        Plot.axes_label(h, ax[k], label)
    end

    h.text(0.5, 0.997, "Binary white noise", fontsize=24, color=BLUE, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.667, "Gratings", fontsize=24, color=RED, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.334, "Awake", fontsize=24, color=GOLD, horizontalalignment="center", verticalalignment="top")

    return h
end
# ============================================================================ #
function plot_one(d, typ, ax, colors, xmn, xmx, xv; inset_length::Integer=30, yloc::Real=0.5)

    len = size(d[typ]["xf_q1"], 1)
    t = range(-len*0.001, -0.001, length=len)

    N = length(filter(x->match(r"xf_q\d", x) != nothing, keys(d[typ])))

    if inset_length > 0
        sax = add_subplot_axes(ax[1], [0.35, 0.45, 0.35, 0.55])
        default_axes(sax)
        sax.set_yticklabels([])
        ki = length(t)-inset_length
        sax.plot([t[ki], t[end]], [0,0], "--", color="black", linewidth=1)
        sax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(yloc))
    end

    ax[1].plot([t[1], t[end]], [0,0], "--", color="black", linewidth=2)

    mn = +Inf
    mx = -Inf
    for q in 1:N
        name = "q" * string(q)
        mnt, mxt = filter_plot(d, t, typ, name, ax[1], sax, colors[q], inset_length)

        mn = min(mn, mnt)
        mx = max(mx, mxt)

        Plot.swarmplot(d[typ]["rri_"*name], xv[q], ax=ax[2], markersize=14, color=colors[q], xs=0.7, ys=0.1)

        bs = bmedian(d[typ]["rri_"*name])
        GAPlot.distribution(bs, ax[3], x=xv[q], draw_axis=false, color=colors[q], peak=1.2, sep=0.17)
        ax[3].plot([xmn, xmx], [0,0], "--", linewidth=1, color="gray", alpha=0.75)
    end

    if inset_length > 0
        inset_box(t, mn, mx, ax[1], inset_length)
    end

end
# ============================================================================ #
function inset_box(t::AbstractVector{<:Real}, mn::Real, mx::Real, ax, inset_length)
    ki = length(t)-inset_length
    xe = 0.004
    xl = [xe    t[ki] t[ki] xe;
          t[ki] t[ki] xe    xe]
    yl = [mn mn mx mx;
          mn mx mx mn]
    ax.plot(xl, yl, "--", color="black", linewidth=1)
end
# ============================================================================ #
function filter_plot(d::Dict{String,Any}, t::AbstractVector{<:Real}, typ, name, ax, sax, col, inset_length)
    val, lo, hi = filter_ci(PaperUtils.normalize(d[typ]["xf_"*name]))
    plot_with_error(t, val, lo, hi, RGB(col...), ax, linewidth=3, label=uppercase(name), ferr=5.0)

    mn = 0
    mx = 0
    if inset_length > 0
        ki = length(t)-inset_length:length(t)
        plot_with_error(t[ki], val[ki], lo[ki], hi[ki], RGB(col...), sax, linewidth=3, ferr=5.0)
        mn = minimum(lo[ki])
        mx = maximum(hi[ki])
    end

    return mn, mx
end
# ============================================================================ #
function format_filter_plot(ax, yloc::Real=0.5)
    ax.set_xlabel("Time before spike (seconds)", fontsize=14)
    ax.set_ylabel("Filter weight (A.U.)", fontsize=14)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
end
# ============================================================================ #
function run_one(::Type{T}, kuse::AbstractVector{<:Integer}, ret::AbstractVector{<:Real}, lgn::AbstractVector{<:Real}, bin_size::Real) where T <: RelayGLM.PerformanceMetric

    response = wasrelayed(ret[kuse], lgn)

    ps = PredictorSet()
    ps[:ff] = Predictor(ret, ret[kuse], CosineBasis(length=FFSPAN, offset=2, nbasis=16, b=10, ortho=false, bin_size=bin_size))
    lm = 2.0 .^ range(-3.5, 3, length=5)
    glm = GLM(ps, response, RidgePrior, [lm])

    return cross_validate(T, Binomial, Logistic, glm, nfold=10, shuffle_design=true)
end
# ============================================================================ #
function filter_stats(f1::AbstractVector{<:Real}, f2::AbstractVector{<:Real}, bin_size::Real)
    return cor(f1, f2), RelayUtils.trapz(abs.(f1 .- f2)) .* bin_size
end
# ============================================================================ #
function rate_split(ret::Vector{Float64}, lgn::Vector{Float64}, kwin::Integer)
    p, _ = psth(lgn, ret, -(kwin + 1):-1, 0.001)
    fr = vec(sum(p, dims=1))
    # return quantile_groups(fr, [0.25, 0.5, 0.75])
    return quantile_groups(fr, [0.334, 0.667])
end
# ============================================================================ #
function stats(ret::Vector{Float64}, lgn::Vector{Float64}, kidx::Vector{Int}, win::Integer=FFSPAN)

    nret, nlgn, nrel, ntrg = relayed_contributed_counts(ret, lgn, win)

    eff = nrel ./ nret
    con = ntrg ./ nlgn
    return nanmean(eff[kidx]), nanmean(con[kidx]), nanmean(nret), nanmean(nlgn)
end
# ============================================================================ #
function relayed_contributed_counts(ret::Vector{Float64}, lgn::Vector{Float64}, win_length::Integer=60)
    krel = findall(>(0.0), RelayUtils.relay_status(ret, lgn))
    ktrig = PaperUtils.contribution(ret, lgn)

    # number of retinal spikes w/in each "win_length" length window preceding
    # each retinal spike
    p, _ = psth(ret, ret, -win_length:-1, 0.001)
    nr = vec(sum(p, dims=1))

    # numer of relayed retinal spikes ...
    p, _ = psth(ret[krel], ret, -win_length:-1, 0.001)
    nrel = vec(sum(p, dims=1))

    # number of lgn spikes ...
    p, _ = psth(lgn, ret, -win_length:-1, 0.001)
    nl = vec(sum(p, dims=1))

    # number of triggered lgn spikes ...
    p, _ = psth(lgn[ktrig], ret, -win_length:-1, 0.001)
    ntrg = vec(sum(p, dims=1))

    return nr, nl, nrel, ntrg
end
# ============================================================================ #
init_output(::Type{JSDivergence}, n::Integer) = Vector{Tuple{Float64,Float64}}(undef, n)
init_output(::Type{<:RelayGLM.PerformanceMetric}, n::Integer) = Vector{Float64}(undef, n)
# ============================================================================ #
function add_subplot_axes(ax,rect)
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[1:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[1]
    y = infig_position[2]
    width *= rect[3]
    height *= rect[4]
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[1].get_size()
    y_labelsize = subax.get_yticklabels()[1].get_size()
    x_labelsize *= rect[3]^0.5
    y_labelsize *= rect[4]^0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax
end
# ============================================================================ #
function bottom_align(ax1, ax2)
    b1 = ax1.get_position()
    b2 = ax2.get_position()
    b3 = matplotlib.transforms.Bbox([[b1.xmin, b2.ymin], [b1.xmax, b1.ymax]])
    ax1.set_position(b3)
end
# ============================================================================ #
function difference_plot(d; io::IO=stdout)

    col = [GREEN, [0., 0., 0.], PURPLE]

    len = size(d["grating"]["xf_q1"], 1)
    t = range(-len*0.001, -0.001, length=len)

    h = figure()
    ax = default_axes()

    typ = "msequence"

    ax.plot([t[1], t[end]], [0, 0], "--", color="gray", alpha=0.75, linewidth=2)

    for q in 2:4
        name = "q" * string(q)
        val, lo, hi = filter_ci(d[typ]["xf_q1"] .- d[typ]["xf_"*name])
        plot_with_error(t, val, lo, hi, RGB(col[q-1]...), ax, linewidth=3, label="Q1-" * uppercase(name), ferr=5.0)

        y1 = d[typ]["rri_"*name] .- d[typ]["rri_q1"]
        bs = bmedian(y1)
        val, lo, hi = confint(bs, BCaConfInt(0.95), 1)
        md = mad(y1)
        _, p = SimpleStats.paired_permutation_test(median, d[typ]["rri_"*name], d[typ]["rri_q1"])
        @printf(io, "\tMedian RRI difference (%s - Q1): %.3f (MAD %.3f 95%% CI [%.3f, %.3f] p = %.4f, n = %d\n", uppercase(name), val, md, lo, hi, p, length(y1))

    end

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.025))
    ax.set_xlabel("Time before spike (seconds)", fontsize=14)
    ax.set_ylabel("Filter weight difference (A.U.)", fontsize=14)
    ax.legend(frameon=false, fontsize=14, loc="upper left")

    h.tight_layout()

    return h
end
# ============================================================================ #
end
