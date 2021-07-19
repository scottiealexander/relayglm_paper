module FigureS2

using PyPlot, Plot, Statistics, SimpleStats, SimpleFitting, UCDColors, Printf
using GAPlot, DatabaseWrapper, RelayGLM
import PaperUtils, Figure45
# ============================================================================ #
@enum SpikeType AllSpikes TriggeredSpikes
const Strmbol = Union{String,Symbol}
# ============================================================================ #
collate_data() = Figure45.collate_data();
# ============================================================================ #
"""
`make_figure(d::Dict{String,Any}; force::Bool=false, io::IO=stdout, inc::Symbol=:all, denom::SpikeType=AllSpikes)`

Where `d` is a results dictionary returned by `Figure45.collate_data()`

See also: `FigureS2.collate_data()`
"""
function make_figure(d::Dict{String,Any}; force::Bool=false, io::IO=stdout, inc::Symbol=:all, denom::SpikeType=AllSpikes)

    h = figure()
    h.set_size_inches((5.5,9.6))

    row_height = [1.0, 1.0, 1.0]
    row_spacing = [0.08, 0.08, 0.08, 0.06]
    col_width = [1.0, 0.33]
    col_spacing = [0.15, 0.06, 0.1]

    ax = axes_layout(h, row_height=row_height, row_spacing=row_spacing,
        col_width=col_width, col_spacing=col_spacing)

    foreach(default_axes, ax)

    for typ in keys(d)
        if !haskey(d[typ], "contribution") || !haskey(d[typ], "efficacy") || force
            ids = d[typ]["ids"]
            d[typ]["efficacy"] = zeros(length(ids))
            d[typ]["contribution"] = zeros(length(ids))
            for k in eachindex(ids)
                ef, cn = get_eff_cont(typ, ids[k])
                d[typ]["efficacy"][k] = ef
                d[typ]["contribution"][k] = cn
            end
        end
    end

    figure_ab(d, ax[1:4], inc=inc, io=io)
    figure_c(d, ax[5:end], denom=denom, io=io)

    return h
end
# ============================================================================ #
function figure_ab(d, ax; io::IO=stdout, inc::Symbol=:all)

    metric = "rri"
    dg = deepcopy(d["grating"])
    dm = deepcopy(d["msequence"])

    if inc == :common
        ids = intersect(d["grating"]["ids"], d["msequence"]["ids"])
        gidx = map(ids) do id
            findfirst(isequal(id), d["grating"]["ids"])
        end
        midx = map(ids) do id
            findfirst(isequal(id), d["msequence"]["ids"])
        end
        dg["ids"] = ids
        dm["ids"] = ids
        for key in keys(dg[metric])
            dg[metric][key] = dg[metric][key][gidx]
            dm[metric][key] = dm[metric][key][midx]
        end
        for key in ["efficacy", "contribution"]
            dg[key] = dg[key][gidx]
            dm[key] = dm[key][midx]
        end
    end

    lab = L"\mathrm{Residual}\ \mathcal{I}_{Bernoulli}"

    names = Dict{String,String}("ff"=>"RH", "fr"=>"CH")

    rg, pg, glo, ghi = single_figure(dg, metric, "contribution", "ff", ax[1], ax[2], RED, "Gratings (n=$(length(dg["ids"])))", f3="", z="efficacy")
    rm, pm, mlo, mhi = single_figure(dm, metric, "contribution", "ff", ax[1], ax[2], BLUE, "Binary white noise (n=$(length(dm["ids"])))",
        xoffset=0.4, show_legend=true, f3="", z="efficacy")

    # ra, pa, alo, ahi = single_figure(d["awake"], metric, "contribution", "ff", ax[1], ax[2], GOLD, "Awake (n=8)", xoffset=0.8, f3="", z="efficacy")

    ax[1].get_legend().set_bbox_to_anchor([0.02, 1.3])

    xl = "Residual contribution"
    ax[1].set_xlabel(xl, fontsize=14)

    pre = "RH"
    ax[1].set_ylabel(lab * " (RH)", fontsize=14)
    yl = "Residual " * pre

    ax[2].set_xticks([0.05, 0.7])
    ax[2].set_xticklabels(["Gratings", "Binary\nwhite noise"], fontsize=10)

    println(io, "*"^80)
    println(io, "$(xl) vs. $(yl)")
    @printf(io, "\tGratings: R = %.3f, 95%% CI [%.3f, %.3f], p = %.3f\n\tMSequence: R = %.3f, 95%% CI [%.3f, %.3f], p = %.3f\n", rg, glo, ghi, pg, rm, mlo, mhi, pm)

    z = ""#"efficacy"

    rg, pg, glo, ghi = single_figure(dg, metric, "contribution", "fr", ax[3], ax[4], RED, "Gratings", f3="ff", z=z)
    rm, pm, mlo, mhi = single_figure(dm, metric, "contribution", "fr", ax[3], ax[4], BLUE, "Binary white noise",
        xoffset=0.4, show_legend=false, f3="ff", z=z)

    # ra, pa, alo, ahi = single_figure(d["awake"], metric, "contribution", "fr", ax[3], ax[4], GOLD, "Awake",
    #     xoffset=0.8, show_legend=false, f3="ff", z=z)

    ax[3].set_ylim(-0.007, 0.09)

    xl = isempty(z) ? "Contribution" : "Residual contribution"
    ax[3].set_xlabel(xl, fontsize=14)

    yl = isempty(z) ? "" : "Residual "

    ax[3].set_ylabel(yl * "\$\\Delta \\mathcal{I}_{Bernoulli}\$ (RH - CH)", fontsize=14)
    yl = yl * "CH - RH"

    ax[4].set_xticks([0.05, 0.7])
    ax[4].set_xticklabels(["Gratings", "Binary\nwhite noise"], fontsize=10)

    println(io, "*"^80)
    println(io, "$(xl) vs. $(yl)")
    @printf(io, "\tGratings: R = %.3f, 95%% CI [%.3f, %.3f], p = %.3f\n\tMSequence: R = %.3f, 95%% CI [%.3f, %.3f], p = %.3f\n", rg, glo, ghi, pg, rm, mlo, mhi, pm)


     foreach(ax[[1,3]], ["A","B"]) do cax, lab
        Plot.axes_label(cax.figure, cax, lab)
    end

    return ax
end
# ============================================================================ #
function figure_c(d::Dict, ax; denom::SpikeType=AllSpikes, io::IO=stdout)

    metric = "rri"

    colors = Dict("grating"=>RED, "msequence"=>BLUE)
    labels = Dict("grating"=>"Gratings", "msequence"=>"Binary white noise")
    offsets = Dict("grating"=>0, "msequence"=>0.4)

    println("*"^80)
    println("Non-cardinal burst spikes (%) vs. CH - RH")

    for k in ["msequence", "grating"]

        burst = get_burst_percent(d, k, denom)

        di = d[k][metric]["fr"] .- d[k][metric]["ff"]

        N = length(d[k]["ids"])

        _, lo, hi = GAPlot.cor_plot(burst, di, ax=ax, color=colors[k], label=labels[k] * " (n=$(N))", xoffset=offsets[k], sep=0.05)

        r, p = SimpleStats.cor_permutation_test(burst, di, method=:spearman)

        print(io, "\t", titlecase(k), ": ")
        @printf(io, "R = %.3f, 95%% CI [%.3f, %.3f], p = %.3f\n", r, lo, hi, p)
    end

    xlab = denom == AllSpikes ? "Non-cardinal burst spikes (%)" : "% of triggered spikes in bursts"

    ax[1].set_xlabel(xlab, fontsize=14)
    ax[1].set_ylabel(L"\Delta \mathcal{I}_{Bernoulli}\ \mathrm{(CH - RH)}", fontsize=14)

    # ax[2].set_xticks([0, 0.6])
    # ax[2].set_xticklabels(["Gratings", "Binary\nwhite noise"], fontsize=12)
    ax[2].set_xticks([0.05, 0.7])
    ax[2].set_xticklabels(["Gratings", "Binary\nwhite noise"], fontsize=10)
    ax[2].plot(ax[2].get_xlim(), [0, 0], "--", color="black", linewidth=1.5)

    ax[1].set_ylim(-0.007, 0.09)

    Plot.axes_label(ax[1].figure, ax[1], "C")
end
# ============================================================================ #
function fetch_data(d::Dict, f1::String, f2::String="")::Vector{Float64}
    if isempty(f2)
        return d[f1]
    else
        if eltype(d[f1][f2]) <: Tuple
            return getfield.(d[f1][f2], 1)
        else
            return d[f1][f2]
        end
    end
    return Float64[]
end
# ============================================================================ #
function single_figure(d, metric::String, f1::String, f2::String, ax1, ax2, color, label::String;
    sep::Real=0.05, xoffset::Real=0.0, show_legend::Bool=false, f3::String="", z::String="")

    ax2.set_xlim(-0.05, 0.8)
    ax2.plot(ax2.get_xlim(), [0.0, 0.0], "--", color="black", linewidth=1.0)

    d1 = fetch_data(d, f1)
    d2 = fetch_data(d, metric, f2)

    if isempty(f3)
        y = d2
    else
        d3 = fetch_data(d, metric, f3)
        y = d2 .- d3
    end

    if !isempty(z)
        zd = fetch_data(d, z)

        b, m = line_fit(zd, d1)
        xc = d1 .- (zd.*m .+ b)

        b, m = line_fit(zd, y)
        yc = y .- (zd.*m .+ b)

        # spearman cor
        val, lo, hi = GAPlot.cor_plot(xc, yc, ax=[ax1,ax2], color=color, label=label, xoffset=xoffset, sep=0.05, method=:spearman)
        _, p = SimpleStats.cor_permutation_test(xc, yc)
    else
        _, lo, hi = GAPlot.cor_plot(d1, y, ax=[ax1,ax2], color=color, label=label, xoffset=xoffset, sep=0.05)
        val, p = SimpleStats.cor_permutation_test(d1, y, method=:spearman)
    end

    if show_legend
        ax1.legend(frameon=true, loc="upper left", bbox_to_anchor=[0.45, 1.25], fontsize=12)
    end

    return (val, p, lo, hi)
end
# ============================================================================ #
function get_burst_percent(d::Dict, type::String, denom::SpikeType=AllSpikes)

    tmp = Dict{String,Strmbol}("grating" => "(?:contrast|area|grating)",
        "msequence" => "msequence", "awake" => :weyand)

    db = get_database(tmp[type], id -> !in(id, PaperUtils.EXCLUDE[type]))

    N = length(d[type]["ids"])
    burst_pct = fill(NaN, N)

    for id in d[type]["ids"]
        ret, lgn, _, _ = get_data(db, id=id)

        # make sure burst_pct is in the same order as the data Dict
        k = findfirst(isequal(id), d[type]["ids"])

        # indicies of all burst spikes (Vector{Vector{Int}}) so length(kb)
        # is the number of bursts, and kb[k] are the indicies of the spikes
        # that comprise the k'th burst, thus length(kb[k]) is the number of
        # spikes in the k'th burst
        kb = PaperUtils.burst_spikes(lgn, 0.004, 0.1)

        if denom == AllSpikes
            # number of *non-cardinal* burst spikes
            nnc = map(x->length(x)-1, kb)

            burst_pct[k] = (sum(nnc) / length(lgn)) * 100.0

        elseif denom == TriggeredSpikes

            # indicies of all triggered lgn spikes
            ktrg = PaperUtils.contribution(ret, lgn)

            # indicies of all non-cardinal burst spikes
            knc = vcat(map(x->x[2:end], kb)...)

            # number of non-cardinal burst spikes that were triggered
            nnct = sum(in(ktrg), knc)
            burst_pct[k] = (nnct / length(ktrg)) * 100.0
        end

    end

    return burst_pct
end
# ============================================================================ #
function get_eff_cont(typ, id)
    tmp = Dict{String,Strmbol}("grating" => "(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)
    db = get_database(tmp[typ], id -> !in(id, PaperUtils.EXCLUDE[typ]))
    ret, lgn, _, _ = get_data(db; id=id)
    return sum(wasrelayed(ret, lgn)) / length(ret), length(PaperUtils.contribution(ret, lgn)) / length(lgn)
end
# ============================================================================ #
end
