module Plot

using PyPlot, Colors, ColorTypes, StatsBase
using PyCall

export plot_with_error, plot_hist, qplot, qbar, closeall, default_figure,
    default_axes, vline, hline, edges2centers, axes_layout, axes_label

const RealVec = AbstractVector{<:Real}
# ============================================================================ #
function plot_with_error(x::RealVec, y::RealVec, yerr::RealVec,
    col::AbstractString, ax=nothing; args...)
    return plot_with_error(x, y, yerr, parse(Colorant, col), ax; args...)
end
# ---------------------------------------------------------------------------- #
function plot_with_error(x::RealVec, y::RealVec, yerr::RealVec,
    col::ColorTypes.RGB, ax=nothing; args...)
    return plot_with_error(x, y, y.-yerr, y.+yerr, col, ax; args...)
end
# ---------------------------------------------------------------------------- #
function plot_with_error(x::RealVec, y::RealVec, ylo::RealVec, yhi::RealVec,
    col::ColorTypes.RGB, ax=nothing; linewidth=4.0, alpha=0.5, label="", ferr=8.0, args...)
    if ax == nothing
        ax = default_axes()
    end
    col_array = [col.r, col.g, col.b]
    fcol, ecol = shading_color(col_array, ferr)
    ax.fill_between(x, ylo, yhi, facecolor=fcol, edgecolor=ecol, alpha=alpha)
    return ax.plot(x, y, "-", color=col_array, linewidth=linewidth, label=label, args...)
end
# ---------------------------------------------------------------------------- #
function shading_color(col::RealVec, ferr::Real=6.0)
    fedge = 0.25
    orig = convert(HSV, RGB(col...))
    hsv = HSV(orig.h, orig.s/ferr, 1.0 - (abs(1.0 - orig.s)^ferr/ferr))
    col_err = hsv2rgb(hsv)
    col_edge = (1.0 - fedge) * col_err + fedge * col
    return col_err, col_edge
end
# ---------------------------------------------------------------------------- #
function hsv2rgb(x::ColorTypes.HSV{T}) where {T<:Number}
    rgb = convert(RGB, x)
    return T[rgb.r, rgb.g, rgb.b]
end
# ============================================================================ #
plot_hist(x::AbstractVector) = plot_hist(x, round(Int, length(x)*0.1))
function plot_hist(x::AbstractVector, n::Integer)
    mn, mx = extrema(x)
    pad = (mx - mn) * 0.01
    e = range(mn - pad, mx + pad, length=n)
    return plot_hist(x, e)
end
function plot_hist(x::AbstractVector, e::AbstractVector)
    hst = fit(Histogram, x, e)
    return _plot_hist(e, hst.weights)
end
# ============================================================================ #
edges2centers(edges::AbstractVector) = edges[1:end-1] .+ (step(edges)/2.0)
# ============================================================================ #
function _plot_hist(edges::AbstractVector, counts::AbstractVector)
    return qbar(edges2centers(edges), counts, step(edges))
end
# ============================================================================ #
function vline(x::Number; args...)
    plot([x, x], ylim(); args...)
end
# --------------------------------------------------------------------------- #
function vline(x::AbstractArray; args...)
    for item in x
        vline(item; args...)
    end
end
# ============================================================================ #
function hline(y::Number; args...)
    plot(xlim(), [y, y]; args...)
end
# --------------------------------------------------------------------------- #
function hline(y::AbstractArray; args...)
    for item in y
        hline(item; args...)
    end
end
# ============================================================================ #
function qplot(x, y)
    h = default_figure()
    p = plot(x, y, color="blue", linewidth=3.0)

    return h
end
# ============================================================================ #
function qplot(y)
    x = 1:length(y)
    h = qplot(x, y)
    return h
end
# ============================================================================ #
function qbar(x, y, width, col="black")
    h = default_figure()

    #width = absolute bar width
    # add error with yerr=...
    p = bar(x, y, width, color=col)

    return h
end
# ============================================================================ #
function qbar(y)
    x = 1:length(y)
    h = qbar(x-0.5, y, 1)
    return h
end
# ============================================================================ #
function closeall()
    plt[:close]()
end
# ============================================================================ #
function default_figure(h=nothing)
    if h == nothing
        h = figure(facecolor="white")
    elseif isa(h, PyPlot.Figure)
        h.set_facecolor("white")
        ax = h.get_axes()
        if !isempty(ax)
            for x in ax
                delaxes(x)
            end
        end
    end

    ax = h.add_axes(default_axes())

    return h, ax
end
# ============================================================================ #
function default_axes(ax=nothing, width=3.0)
    if ax == nothing
        ax =PyPlot.axes()
    end
    #set ticks to face out
    ax.tick_params(direction="out", length=width*2, width=width)

    #turn off top and right axes
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)

    #remove top and right tick marks
    tmp = ax.get_xaxis()
    tmp.tick_bottom()

    tmp = ax.get_yaxis()
    tmp.tick_left()

    ax.spines["left"].set_linewidth(width)
    ax.spines["bottom"].set_linewidth(width)

    return ax
end
# ============================================================================ #
function axes_layout(h::PyPlot.Figure; row_height::RealVec=Float64[], row_spacing::RealVec=Float64[],
    col_width::RealVec=Float64[], col_spacing::RealVec=Float64[])

    nrow = length(row_height)
    ncol = length(col_width)
    ax = Vector{PyCall.PyObject}(undef, nrow*ncol)

    height, btm = axes_position(row_height, spacing=row_spacing, vertical=true)

    for k in eachindex(height)
        width, left = axes_position(col_width, spacing=col_spacing)
        for j in eachindex(width)
            ax[(k-1)*ncol + j] = h.add_axes([left[j], btm[k], width[j], height[k]])
        end
    end
    return ax
end
# ============================================================================ #
function axes_position(width::AbstractVector{<:Real}; pad::Real=0.05, spacing::AbstractVector{<:Real}=fill(pad, length(width)+1), vertical::Bool=false)
    @assert(length(spacing) > length(width), "spacing array does not contain enough items!")
    widthsc = (width ./ sum(width)) .* ((1.0-spacing[end]) - sum(spacing[1:end-1]))
    if vertical
        left = 1 .- cumsum(widthsc .+ spacing[1:end-1])
    else
        left = cumsum(spacing[1:end-1]) .+ vcat(0.0, cumsum(widthsc[1:end-1]))
    end
    return widthsc, left
end
# ============================================================================ #
function axes_label(h, ax, lab::String, x::Real=NaN)

    yl = ax.get_ylim()

    px = matplotlib.transforms.Bbox.from_bounds(10, 10, 60, 10)
    transf = ax.transData.inverted()
    tmp = px.transformed(transf)

    if isnan(x)
        field = :x1
        ytl = ax.get_yticklabels()
        if !isempty(ytl)
            ht = ax.get_yticklabels()[end]
        else
            ht = ax.yaxis.get_label()
            if isempty(ht.get_text())
                ht = ax
                field = :x0
            end
        end

        bbpx = ht.get_window_extent(renderer=h.canvas.get_renderer())
        bbdata = bbpx.transformed(transf)
        x = getproperty(bbdata, field) - tmp.width
    end

    ax.text(x, yl[2] + tmp.height, lab, fontsize=30, verticalalignment="bottom", horizontalalignment="left")
    return x
end
# ============================================================================ #
function hist_indicies(x::AbstractVector{<:Real}, edges::AbstractVector{<:Real})
    ki = [Vector{Int}() for k in 1:(length(edges)-1)]
    for k in eachindex(x)
        kb = searchsortedlast(edges, x[k])
        if 0 < kb < length(edges)
            push!(ki[kb], k)
        end
    end
    return ki
end
# ============================================================================ #
function vsortperm(x::AbstractVector{<:Real})
    ks = sortperm(x, rev=true)
    idx = zeros(Int, length(x))
    k1 = 1
    k2 = length(x)
    for k in eachindex(ks)
        if mod(k, 2) == 0
            idx[k2] = ks[k]
            k2 -=1
        else
            idx[k1] = ks[k]
            k1 += 1
        end
    end
    return idx
end
# ============================================================================ #
function swarmplot(y::AbstractVector{<:Real}, x::Real=0; markersize::Real=12, ax=PyPlot.gca(), color=nothing, xs::Real=1.0, ys::Real=3.0)

    mn, mx = extrema(y)

    hm = ax.plot(fill(Float64(x), length(y)), y, ".", markersize=markersize, color=color)[1]

    M = ax.transData.inverted().get_matrix()
    xscale = M[1,1]
    yscale = M[2,2]

    vstep = markersize * yscale * ys
    hstep = markersize * xscale * xs

    edges = mn - (vstep/2) : vstep : mx + (vstep/2)

    ki = hist_indicies(y, edges)

    xd = hm.get_xdata()

    for k in eachindex(ki)
        n = length(ki[k])
        if n > 1
            # ax.plot([-hstep*10, +hstep*10], [edges[k], edges[k]], color="black")
            l = floor(n/2)
            if mod(n, 2) == 0
                l -= 0.5
            end
            ks = vsortperm(y[ki[k]])
            xd[ki[k][ks]] .+= range(-l*hstep, l*hstep, length=n)
        end
    end

    hm.set_xdata(xd)

    return hm
end
# ============================================================================ #
end #END MODULE

# ============================================================================ #
# NOTES
# ============================================================================ #
#  #update figure
#  h[:canvas][:draw]()
#
#  #close all figures
#  plt[:close]()
