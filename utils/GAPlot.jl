module GAPlot
using PyPlot, Statistics, Printf, Bootstrap, ImageFiltering, StatsBase
using Plot, SimpleFitting

import SimpleStats

export Upper, Lower, dmean, dmedian, dmean_std

import Base.copy, Base.size, Bootstrap.draw!

# ============================================================================ #
# hacks to make Bootstrap.jl work for independent samples
const DSet = Tuple{Vector{Float64},Vector{Float64}}

Base.copy(d::DSet) = (copy(d[1]), copy(d[2]))
Base.size(d::DSet) = (length(d[1]), length(d[2]))

# draw a random sample from each group (w/ replacement) that has the same size
# as the number of samples in that group
function Bootstrap.draw!(src::DSet, dst::DSet)
    Bootstrap.sample!(src[1], dst[1])
    Bootstrap.sample!(src[2], dst[2])
end

# mean / median difference for independent samples
dmean(d::DSet) = SimpleStats.nanmean(d[1]) - SimpleStats.nanmean(d[2])
dmean(d::AbstractVector{<:Real}) = SimpleStats.nanmean(d)
dmedian(d::DSet) = SimpleStats.nanmedian(d[1]) - SimpleStats.nanmedian(d[2])
dmedian(d::AbstractVector{<:Real}) = SimpleStats.nanmedian(d)

# equal variance pooling for two groups
function pooledvar(x, y)
    nx = length(x)
    ny = length(y)
    return ((nx-1)*SimpleStats.nanvar(x) + (ny-1)*SimpleStats.nanvar(y)) / (nx + ny - 2)
end

# mean + std for use with StudentConfInt(): i.e. for gaplot() the first argument
# <f> *MUST* be dmean_std when ci=StudentConfInt(x), otherwise it doesn't matter
function dmean_std(d::DSet)
    m = SimpleStats.nanmean(d[1]) - SimpleStats.nanmean(d[2])
    return m, sqrt(pooledvar(d[1], d[2]))
end

function dmean_std(d::AbstractVector{<:Real})
    return SimpleStats.nanmean(d), SimpleStats.nanstd(d)
end
# ============================================================================ #
mutable struct UniquePairings{T<:AbstractVector}
    pt1::Int
    pt2::Int
    list::T
end
unique_pairings(x::AbstractVector) = UniquePairings(1,2,x)
Base.IteratorSize(u::UniquePairings) = Base.HasLength()
Base.IteratorEltype(u::UniquePairings) = Base.HasEltype()
Base.length(u::UniquePairings) = round(Int, (length(u.list) * (length(u.list) - 1))/2)
Base.eltype(u::UniquePairings{T}) where T = Tuple{T,T}
Base.iterate(u::UniquePairings) = iterate(u, 1)
# ---------------------------------------------------------------------------- #
function Base.iterate(u::UniquePairings, k::Integer)
    k > length(u) && return nothing
    pair = (u.pt1, u.pt2)
    u.pt2 += 1
    if u.pt2 > length(u.list)
        u.pt2 = u.pt1 + 2
        u.pt1 += 1
    end
    return pair, k+1
end
# ============================================================================ #
function smooth(x::AbstractVector{<:Real}, sigma::Real, bin_size::Real=0.001)
    return imfilter(x, (KernelFactors.IIRGaussian(sigma/bin_size),))
end
# ============================================================================ #
function get_dist(x::AbstractVector{<:Real}, nbin::Integer=min(floor(Int, length(x)/10), 30), sigma::Real=3.0)

    mn, mx = extrema(x)

    r = mx - mn
    e = range(mn - 0.05*r, mx + 0.05*r, length=nbin)
    hst = fit(Histogram, x, e)
    st = step(e)
    c = smooth(Vector{Float64}(hst.weights), sigma*st, st)

    return e[1:end-1] .+ step(e)/2, c
end
# ============================================================================ #
function roundn(x::Real, n::Integer)
    f = 10.0^-n
    return round(x * f) / f
end
# ============================================================================ #
# function distribution(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, ax; x::Real=0, xoffset::Real=0)
const Triple = Tuple{Float64,Float64,Float64}
function distribution(bs::Bootstrap.BootstrapSample, ax; x::Real=0, y::Real=0,
    xoffset::Real=0, draw_axis::Bool=true, ci::Triple=confint(bs, BCaConfInt(0.95), 1),
    color::Union{String,Vector{Float64},Nothing}="gray", sep::Real=0.05, alpha::Real=1.0, peak::Real=0.25)

    bins, counts = get_dist(bs.t1[1], 50)

    # peak = 0.25
    xpn = (counts ./ maximum(counts)) .* peak
    xn = vcat(0, xpn, 0) .+ (x + sep)
    yn = vcat(bins[1], bins, bins[end])

    val, lo, hi = ci

    # y = median(a)

    _, kf = findmin(abs.(val .- bins))
    yn = (yn .- bins[kf]) .+ (y + val)

    verticies = hcat(xn, yn)

    path = matplotlib.path.Path(verticies)
    dist = matplotlib.patches.PathPatch(path, color=color, alpha=alpha)
    ax.add_patch(dist)

    # large dot indicating the measured effect size
    ax.plot(x, y + val, ".", markersize=12, color="black")

    # line through the dot illustrating the 95% CI
    ax.plot([x, x], [y+lo, y+hi], color="black", linewidth=2)

    if draw_axis
        # vertical axis for resampled distribution
        pad2 = peak + 0.07 + sep
        pad3 = pad2 + 0.02

        if lo < 0 && hi > 0
            yx = [yn[1], yn[end]]
            yt = [yn[1], y, y + val, yn[end]]
        else
            yx = val < 0 ? [yn[1], y] : [yn[end], y]
            yt = [y, y + val, yx[1]]
        end

        ax.plot([x + pad2, x + pad2], yx, color="black", linewidth=2)

        # horizontal tick marks for the vertical axis
        xt = repeat([x + pad2, x + pad3], 1, length(yt))
        yt = repeat(reshape(yt, 1, :), 2, 1)
        ax.plot(xt, yt, color="black", linewidth=2)
        ytl = yt .- y

        rn = -1
        if ytl[end] - ytl[1] < 0.2
            rn = -2
        elseif ytl[end] - ytl[1] < 0.02
            rn = -3
        end

        # tick labels
        for k in eachindex(ytl)
            ax.text(x + pad3 + 0.015, yt[k], string(roundn(ytl[k], rn)), fontsize=10, verticalalignment="center", horizontalalignment="left")
        end

        ax.plot([2 + xoffset, x + pad2], y .+ [val, val], "--", color="black", linewidth=1)
        # ax.plot([2, x + pad2], [bm, bm], "--", color="black", linewidth=1)
        ax.plot([1 + xoffset, x + pad2], [y, y], "--", color="black", linewidth=1)
    end

    return val, lo, hi
end
# ============================================================================ #
const ColorSpec = Union{Vector{String}, Vector{<:AbstractVector{<:Real}}}
# ============================================================================ #
function gaplot(f::Function, a::AbstractVector{<:Real}, b::AbstractVector{<:Real};
    names::Vector{<:AbstractString}=String[], ax=nothing, colors::ColorSpec=String[],
    xoffset::Real=0, method::Bootstrap.ConfIntMethod=PercentileConfInt(0.95))

    if ax == nothing
        h, ax = subplots(1,1)
        h.set_size_inches((6,5))
        default_axes(ax)
    end

    for (k,d) in enumerate((a,b))
        if k <= length(colors)
            col = colors[k]
        else
            col = nothing
        end
        ax.plot(fill(k + xoffset, length(d)), d, ".", markersize=12, alpha=0.75, color=col)[1]
    end

    ax.set_xticks([1,2] .+ xoffset)
    if !isempty(names)
        ax.set_xticklabels(names, fontsize=14)
    end
    # ax.set_xlim([0.5, 2.5])
    ax.set_ylabel("Value", fontsize=14)
    ax.set_xlabel("Group name", fontsize=14)

    bs = bootstrap(f, (b, a), BasicSampling(5000))

    if typeof(method) == BCaConfInt
        error("BCaConfInt does not currenly work for independent samples bootstrapping due to use of jack-knife procedure")
    elseif typeof(method) == StudentConfInt
        # kind of a clunky interface...
        val, lo, hi = confint(bs, Bootstrap.straps(bs, 2), method, 1)
    else
        val, lo, hi = confint(bs, method, 1)
    end

    val, lo, hi = distribution(bs, ax, x=2.15 + xoffset, y=f(a)[1], xoffset=xoffset, ci=(val, lo, hi))

    return val, lo, hi
end
# ============================================================================ #
function paired_gaplot(a::AbstractVector{<:Real}, b::AbstractVector{<:Real};
    names::Vector{<:AbstractString}=String[], ax=nothing, colors::ColorSpec=String[],
    xoffset::Real=0)

    if ax == nothing
        h, ax = subplots(1,1)
        h.set_size_inches((6,5))
        default_axes(ax)
    end

    x = repeat([1, 2] .+ xoffset, 1, length(a))
    y = transpose(hcat(a, b))
    ax.plot(x, y, linewidth=2, alpha=0.5, color="gray")

    for (k,d) in enumerate((a,b))
        if k <= length(colors)
            col = colors[k]
        else
            col = nothing
        end
        ax.plot(fill(k + xoffset, length(d)), d, ".", markersize=12, alpha=0.75, color=col)[1]
    end

    ax.set_xticks([1,2] .+ xoffset)
    if !isempty(names)
        ax.set_xticklabels(names, fontsize=14)
    end
    # ax.set_xlim([0.5, 2.5])
    ax.set_ylabel("Value", fontsize=14)
    ax.set_xlabel("Group name", fontsize=14)

    bs = bootstrap(median, b .- a, BasicSampling(5000))
    val, lo, hi = distribution(bs, ax, x=2.15 + xoffset, y=median(a), xoffset=xoffset)

    #p = SimpleStats.paired_permutation_test(median, a, b, 5000, tail=SimpleStats.Both)
    return val, lo, hi#, p
end
# ============================================================================ #
@enum TriangleType Upper Lower
function ul(b::Integer, keep::TriangleType=Lower)
    idx = Vector{Int}()
    for k = 2:b
       ks = b*(k-1)+1
       ke = b*b-(k-2)
       append!(idx, ks:b+1:ke)
    end
    if keep == Upper
        idx .= (b*b+1) .- idx
    end
    return sort(idx)
end
# ============================================================================ #
function paired_gamatrix(data::Vector{Vector{Float64}};
    names::Vector{<:AbstractString}=String[],
    ax=nothing,
    tri::TriangleType=Lower,
    colors::ColorSpec=String[]
    )

    @assert(length(data) >= 3, "At least 3 data set are required to form a GA plot matrix")

    n = length(data)
    if ax == nothing
        N = n - 1
        h, ax = subplots(n-1, n-1)
        ax = vec(ax)
        krm = ul(n-1, tri)
        for k in krm
            ax[k].remove()
        end
        deleteat!(ax, krm)
        h.set_size_inches((7,6))
    end

    if isempty(names)
        names = map(string, 1:length(data))
    end

    kp = 1
    for (k1, k2) in unique_pairings(data)
        # @show(k1, k2)
        default_axes(ax[kp])
        paired_gaplot(data[k1], data[k2], names=names[[k1,k2]], ax=ax[kp], colors=colors[[k1,k2]])
        kp += 1
    end
    tight_layout()
    return ax
end
# ============================================================================ #
function cor_plot(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, z::AbstractVector{<:Real}=Float64[];
    method::Symbol=:spearman, ax=nothing, xoffset::Real=0.0, xlim::Tuple{<:Real,<:Real}=(NaN,NaN), args...)

    if ax == nothing
        h = figure()
        ax = axes_layout(h, row_height=[1.0], row_spacing=[0.08, 0.05],
            col_width=[1.0, 0.33], col_spacing=[0.17, 0.05, 0.1])

        foreach(default_axes, ax)
    end

    color = get(args, :color, nothing)
    linewidth = get(args, :linewidth, 2.0)
    markersize = get(args, :markersize, 12)
    label = get(args, :label, "")
    sep = get(args, :sep, 0.02)

    ax[1].plot(a, b, ".", markersize=markersize, color=color, label=label)

    yi, m = line_fit(a, b)
    if all(isnan, xlim)
        x = [extrema(a)...]
        xr = (x[2] - x[1]) * 0.1
        x[1] -= xr
        x[2] += xr
    else
        x = xlim
    end
    y = m .* x .+ yi
    ax[1].plot(x, y, "--", color=color, linewidth=linewidth, alpha=0.6)

    if isempty(z)
        if method == :spearman
            bs = SimpleStats.bspearman(a, b)
        else
            bs = SimpleStats.bcor(a, b)
        end
    else
        bs = SimpleStats.bcor_partial(a, b, z)
    end
    val, lo, hi = distribution(bs, ax[2], x=xoffset, y=0, xoffset=0, draw_axis=false, color=color, sep=sep)

    ax[2].set_ylim(-1, 1)
    ax[2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))

    ax[2].spines["left"].set_visible(false)
    ax[2].spines["right"].set_visible(true)
    ax[2].spines["right"].set_linewidth(3.0)
    ax[2].yaxis.tick_right()

    return val, lo, hi
end
# ============================================================================ #
function slopeplot(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, ax; xoff::Real=0, colors=[nothing,nothing])
    x1 = fill(1, length(a)) .+ xoff
    x2 = fill(2, length(a)) .+ xoff

    ax.plot(vcat(x1', x2'), vcat(a', b'), color=[0.3,0.3,0.3], linewidth=2.0, alpha=0.8)

    ax.plot(x1, a, ".", markersize=12, color=colors[1], alpha=0.7)
    ax.plot(x2, b, ".", markersize=12, color=colors[2], alpha=0.7)

    ax.set_xticks([1,2] .+ xoff)
end
# ============================================================================ #
function cumming_plot(a::AbstractVector{<:Real}, b::AbstractVector{<:Real};
    ax=nothing, xoffset::Real=0, colors=[nothing,nothing], dcolor=nothing,
    method::Symbol=:median, gap::Real=0.008, paired::Bool=true)

    if method == :median
        bs = SimpleStats.bmedian(b .- a)
    else
        bs = SimpleStats.bmean(b .- a)
    end

    if ax == nothing
        gs = Dict{Symbol,Vector{Float64}}(:height_ratios=>[1.0, 0.33])
        h, ax = subplots(2, 1, gridspec_kw=gs)
        foreach(default_axes, ax)
    end

    slopeplot(a, b, ax[1], xoff=xoffset, colors=[[0.3,0.3,0.3], [0.3,0.3,0.3]])

    fc = method == :median ? median : mean
    fe = method == :median ? SimpleStats.mad : std

    yv1 = fc(a)
    yv2 = fc(b)

    # x1 = [0.85 + xoffset, 0.85 + xoffset]
    # y1 = [
    #       yv1-fe(a) yv1+fe(a);
    #       yv1-gap   yv1+gap
    #       ]
    # ax[1].plot(hcat(x1, x1), y1, linewidth=6.0, color=colors[1])
    #
    # x2 = [2.15 + xoffset, 2.15 + xoffset]
    # y2 = [
    #       yv2-fe(b) yv2+fe(b);
    #       yv2-gap   yv2+gap
    #       ]
    # ax[1].plot(hcat(x2, x2), y2, linewidth=6.0, color=colors[2])

    x1 = [0.85 + xoffset, 0.85 + xoffset]
    y1 = [yv1-fe(a), yv1+fe(a)]
    ax[1].plot(x1, y1, linewidth=4.0, color=colors[1])
    ax[1].plot(x1[1], yv1, ".", markersize=18, color=colors[1])

    x2 = [2.15 + xoffset, 2.15 + xoffset]
    y2 = [yv2-fe(b), yv2+fe(b)]
    ax[1].plot(x2, y2, linewidth=4.0, color=colors[2])
    ax[1].plot(x2[1], yv2, ".", markersize=18, color=colors[2])

    ax[1].set_xlim(0.7, 2.3)

    val, lo, hi = distribution(bs, ax[2], draw_axis=false, color=dcolor, sep=0.012)

    ax[2].plot([0.0, 0.3], [0.0, 0.0], "--", color="black", linewidth=1.0)
    ax[2].spines["bottom"].set_visible(false)
    ax[2].set_xticks([])

    return val, lo, hi
end
# ============================================================================ #
function test(a=Float64[], b=Float64[])
    # if isempty(a)
    #     a = randn(30)
    #     b = a .+ (0.5 .* randn(30) .+ 1.0)
    # end

    if isempty(a)
        a = [8.885, 14.38, 8.015, 5.835, 5.47, 12.06, 11.72, 10.315, 5.065, 8.235, 15.08, 13.485, 11.3, 9.82, 9.565]
        b = [6.625, 2.3, 11.975, 3.65, 8.325, 9, 6.675, 5.35, 3.025, 0.7, 8.375, 8.235, 6.91, 8.59, 9.265]
    end

    p = paired_gaplot(a, b, names=["A","B"], colors=[[1.,0.,0.],[0.,0.,1.]])
    @info("p = $(p)")
    return a, b
end
# ============================================================================ #
end
