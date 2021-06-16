module Figure1

using PaperUtils
using PairsDB, SpkTuning, SpkCore, Plot, UCDColors, MSequenceUtils, MSequence
using Statistics, PyPlot, Printf, ImageFiltering

# ============================================================================ #
function collate_data(id::Integer=214, normalize::Bool=true)

    d = Dict{String,Any}()

    # ------------------------------------------------------------------------ #
    d["grating"] = Dict{String,Any}()
    db = get_database("(?:contrast|area|grating)")

    tf = get_uniform_param(db[id=id], "temporal_frequency")
    ret, lgn, evt, lab = get_data(db, id=id, ffile=x->x["temporal_frequency"]==tf)
    d["grating"]["xcorr"] = get_xcorr(ret, lgn, 0.0005, 0.03)
    d["grating"]["shift"] = get_xcorr(ret, lgn .+ (1.0 / tf), 0.0005, 0.03)

    d["grating"]["nret"] = length(ret)
    d["grating"]["nlgn"] = length(lgn)
    d["grating"]["nrelayed"] = PaperUtils.count_relayed(ret, lgn)
    d["grating"]["ntriggered"] = length(PaperUtils.contribution(ret, lgn))

    # ------------------------------------------------------------------------ #

    db = get_database("msequence")
    fpt = get_uniform_param(db[id=id], "frames_per_term")
    fps = get_uniform_param(db[id=id], "refresh_rate")

    ifi = fpt / fps

    ret, lgn, evt, lab = get_data(db, id=id)

    ret_rf = sta(ret, evt, lab, ifi)
    lgn_rf = sta(lgn, evt, lab, ifi)

    if any(isnan, ret_rf) || any(isnan, lgn_rf)
        error("Found NANs in RF!")
    end

    d["msequence"] = Dict{String,Any}("ret"=>Dict{String,Any}(), "lgn"=>Dict{String,Any}())
    # d["msequence"]["ifi"] = ifi

    if normalize
        fxfm = zscore
        d["msequence"]["zscore"] = true
    else
        d["msequence"]["zscore"] = false
        fxfm = x -> x
    end

    rk = peakframe(ret_rf)
    lk = peakframe(lgn_rf)

    d["msequence"]["ret"]["data"] = fxfm(mean_squeeze(ret_rf[:,:,rk-1:rk+1]))
    d["msequence"]["ret"]["p"] = gaussian_fit(d["msequence"]["ret"]["data"])

    d["msequence"]["lgn"]["data"] = fxfm(mean_squeeze(lgn_rf[:,:,rk-1:rk+1]))
    d["msequence"]["lgn"]["p"] = gaussian_fit(d["msequence"]["lgn"]["data"])

    d["msequence"]["xcorr"] = get_xcorr(ret, lgn, 0.0005, 0.03)
    d["msequence"]["nret"] = length(ret)
    d["msequence"]["nlgn"] = length(lgn)
    d["msequence"]["nrelayed"] = PaperUtils.count_relayed(ret, lgn)
    d["msequence"]["ntriggered"] = length(PaperUtils.contribution(ret, lgn))

    return d
end
# ============================================================================ #
function make_figure(d::Dict{String,Any}, id::Integer=214, io::IO=stdout)

    h = figure()
    h.set_size_inches((9.5,8.5))

    rh = [1.0, 1.0]
    rs = [0.08, 0.09, 0.07]
    cw = [1.2, 0.05, 1.2, 0.05]
    cs = [0.1, 0.02, 0.08, 0.02, 0.08]

    ax = axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)

    foreach(default_axes, ax[[5,7]])

    for k in [6,8]
        re = ax[k].get_position().x1
        ax[k].remove()
        pos = ax[k-1].get_position()
        ax[k-1].set_position([pos.x0, pos.y0, re-pos.x0, pos.height])
    end
    deleteat!(ax, [6,8])

    # ------------------------------------------------------------------------ #
    # MSEQUENCE

    cmap = matplotlib.colors.ListedColormap(clay_color())
    smth = [0,0]

    im1 = show_rf(d["msequence"]["ret"]["data"], ax[1], cmap, smth)
    im2 = show_rf(d["msequence"]["lgn"]["data"], ax[3], cmap, smth)
    cl1 = im1.get_clim()
    cl2 = im2.get_clim()
    cl = [min(cl1[1], cl2[1]), max(cl1[2], cl2[2])]

    foreach((im1,im2)) do x
        cl = x.get_clim()
        mx = maximum(abs.(cl))
        x.set_clim(-mx, mx)
    end

    if d["msequence"]["zscore"]
        ticks = nothing
    else
        ticks = -15:5:15
    end

    cb = plt.colorbar(im1, cax=ax[2], ticks=ticks)
    cb.outline.remove()

    cb = plt.colorbar(im2, cax=ax[4])
    cb.outline.remove()

    x, y = ellipse(vcat(d["msequence"]["lgn"]["p"], d["msequence"]["lgn"]["p"][end]))
    ax[1].plot(x, y, color="white", linewidth=2)

    x, y = ellipse(vcat(d["msequence"]["ret"]["p"], d["msequence"]["ret"]["p"][end]))
    ax[3].plot(x, y, color="black", linewidth=2)

    for axc in ax[[1,3]]
        for spine in axc."spines".values()
            spine.set_visible(false)
        end
        axc.set_xticks([])
        axc.set_yticks([])
    end

    ax[1].set_title("RGC", fontsize=16)
    ax[3].set_title("LGN", fontsize=16)

    kxc = 5

    y, x = d["msequence"]["xcorr"]
    ax[kxc].bar(x, y, 0.0005, color="black", linewidth=3)
    ax[kxc].set_title("Binary white noise", fontsize=16)
    ax[kxc].set_ylabel("# of spikes", fontsize=14)
    ax[kxc].set_xlabel("Time (seconds, 0 = retinal spike)", fontsize=14)
    mn, mx = extrema(x)
    ax[kxc].set_xlim(mn .- 0.00025, mx + 0.00025)

    yl = ax[kxc].get_ylim()
    yr = yl[2] - yl[1]

    ax[kxc].text(0.03, yl[2] - yr * 0.25, "RGC = $(stringify(d["msequence"]["nret"]))", fontsize=12, verticalalignment="center", horizontalalignment="right")
    ax[kxc].text(0.03, yl[2] - yr * 0.3, "LGN =   $(stringify(d["msequence"]["nlgn"]))", fontsize=12, verticalalignment="center", horizontalalignment="right")

    # ------------------------------------------------------------------------ #
    # GRATINGS
    kxc = 6
    y, x = d["grating"]["xcorr"]
    ax[kxc].bar(x, y, 0.0005, color="black")
    y, x = d["grating"]["shift"]
    ax[kxc].plot(x, y, color="red", linewidth=2)
    ax[kxc].set_title("Gratings", fontsize=16)
    mn, mx = extrema(x)
    ax[kxc].set_xlim(mn .- 0.00025, mx + 0.00025)

    yl = ax[kxc].get_ylim()
    yr = yl[2] - yl[1]

    ax[kxc].text(0.03, yl[2] - yr * 0.25, "RGC = $(stringify(d["grating"]["nret"]))", fontsize=12, verticalalignment="center", horizontalalignment="right")
    ax[kxc].text(0.03, yl[2] - yr * 0.3, "LGN = $(stringify(d["grating"]["nlgn"]))", fontsize=12, verticalalignment="center", horizontalalignment="right")

    # ------------------------------------------------------------------------ #

    foreach(x -> Plot.axes_label(h, x[1], x[2]), zip(ax[[1,3,5,6]], ["A","B","C","D"]))

    if d["msequence"]["zscore"]
        lab = "Normalized activity"
    else
        lab = "Spikes / second"
    end
    for cax in ax[[2,4]]
        cax.set_xlim(0, 1)
        cax.text(3.8, 0.5, lab, fontsize=12, rotation=-90, horizontalalignment="left", verticalalignment="center")
    end

    println(io, "Pair $(id)")

    eff = d["msequence"]["nrelayed"] / d["msequence"]["nret"]
    con = d["msequence"]["ntriggered"] / d["msequence"]["nlgn"]
    @printf(io, "\tmsequence: efficacy = %.3f contribution = %.3f\n", eff, con)

    eff = d["grating"]["nrelayed"] / d["grating"]["nret"]
    con = d["grating"]["ntriggered"] / d["grating"]["nlgn"]
    @printf(io, "\tgratings: efficacy = %.3f contribution = %.3f\n", eff, con)

end
# ============================================================================ #
function maxnorm(x::AbstractArray{<:Real})
    mn, mx = extrema(x)
    return (x .- mn) ./ (mx - mn)
end
# ---------------------------------------------------------------------------- #
zscore(x::AbstractArray{<:Real}) = (x .- mean(x)) ./ std(x)
# ============================================================================ #
@inline function mean_squeeze(x::Array{<:Real, N}) where N
    return dropdims(mean(x, dims=N), dims=N)
end
# ============================================================================ #
function show_rf(x::Matrix{<:Real}, ax, cmap, sm::Vector{<:Real}=[0,0])
    if any(>(0), sm)
        img = imfilter(x, KernelFactors.IIRGaussian((sm[1],sm[2])))
    else
        img = x
    end
    ax.imshow(img, interpolation="bilinear", cmap=cmap)
end
# ============================================================================ #
function stringify(x::Integer)
    str = "$(x)"
    len = length(str)
    if len > 3
        buf = str[end-2:end]
        for k = (len-3):-1:1
            if mod(len-k, 3) == 0
                buf = "," * buf
            end
            buf = str[k] * buf
        end
    end
    return buf
end
# ============================================================================ #
end
