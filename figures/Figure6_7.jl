module Figure6_7

using GAPlot, Plot, UCDColors, SimpleStats, DatabaseWrapper, RelayGLM, Progress
using RelayGLM.RelayISI, PaperUtils, HyperParamsCV
using LinearAlgebra, Statistics, PyPlot, ColorTypes, Bootstrap, Printf
import JSON

const Strmbol = Union{String,Symbol}

# ============================================================================ #
function collate_data()
    return open(JSON.parse,
        joinpath(@__DIR__, "..", "preprocessed_data", "figure6_7.json"), "r")
end
# ============================================================================ #
# remove all pairs whose I_{Bernoulli} (averaged across models) is below the median
function filter_by_rri!(d)

    rm = Dict{String,Vector{Int}}()

    for stim in ["grating","msequence"]

        rri = vec(mean(hcat(d[stim]["rri"]["isi"], d[stim]["rri"]["ff"], d[stim]["rri"]["fr"]), dims=2))

        h = floor(Int, length(rri)/2)
        krm = sort(sortperm(rri)[1:h])
        rm[stim] = d[stim]["ids"][krm]

        for model in ["isi","ff","fr"]
            deleteat!(d[stim]["rri"][model], krm)
        end

        deleteat!(d[stim]["ids"], krm)
    end

    rm["awake"] = Int[]

    return d, rm
end
# ============================================================================ #
function collate_data_quick(::Type{T}) where T <: RelayGLM.PerformanceMetric

    bin_size = 0.001
    mxisi = roundn.(10.0 .^ range(log10(0.03), log10(0.5), length=8), -3)

    d = Dict{String, Any}()
    tmp = Dict{String,Strmbol}("grating" => "(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)
    par = HyperParamsCV.load_parameters()

    key = RelayGLM.key_name(T)

    for (type, ptrn) in tmp

        db = get_database(ptrn, id -> !in(id, PaperUtils.EXCLUDE[type]))
        d[type] = Dict{String, Any}()

        d[type][key] = Dict{String, Any}()

        d[type]["ids"] = get_ids(db)

        d[type][key]["ff"] = init_output(T, length(db))
        d[type][key]["fb"] = init_output(T, length(db))
        d[type][key]["isi"] = init_output(T, length(db))
        d[type][key]["sigma"] = init_output(T, length(db))
        d[type][key]["isimax"] = init_output(T, length(db))
        d[type][key]["null"] = zeros(length(db))

        for k in 1:length(db)

            ret, lgn, _, _ = get_data(db, k)
            id = get_id(db[k])

            ff, fb = run_one(T, ret, lgn,
                par[id][type]["ff_temporal_span"], # ff span
                par[id][type]["fb_temporal_span"], # fb span
                par[id][type]["fb_nbasis"],        # fb nbasis
                bin_size
            )

            if !ff.converged
                @warn("Pair $(id) [$(type)] failed to converge: FF")
            end

            if !fb.converged
                @warn("Pair $(id) [$(type)] failed to converge: FR")
            end

            d[type][key]["ff"][k] = mean(ff.metric)
            d[type][key]["fb"][k] = mean(fb.metric)
            d[type][key]["null"][k] = ff.null_nlli

            res, sigma, isimax = isi_cross_validate(T, ret, lgn, mxisi, 0.001, 10)

            d[type][key]["isi"][k] = mean(res)
            d[type][key]["sigma"][k] = sigma
            d[type][key]["isimax"][k] = isimax

            show_progress(k/length(db), 0, "$(type): ", "($(k) of $(length(db)))")
        end
        println()
    end
    return d
end
# ============================================================================ #
function get_efficacy(db, ids)
    eff = zeros(length(ids))
    for k in 1:length(ids)
        ret, lgn, _, _ = get_data(db, id=ids[k])
        status = wasrelayed(ret, lgn)
        eff[k] = sum(status) / length(ret)
    end
    return eff
end
# ============================================================================ #
function rri(ef::Real, df::Real=1.0)
    lh = (ef * log(ef) + (1-ef) * log(1-ef))
    lb = (df - 1) * log(2) # -(1 - df) * log(2)
    return (lb - lh) / log(2)
end
# ============================================================================ #
function normalize_rri!(d)

    tmp = Dict{String,Strmbol}("grating" => "(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)

    out = Dict{String,Any}()

    for k in keys(d)
        db = get_database(tmp[k], id -> !in(id, PaperUtils.EXCLUDE[k]))
        eff = get_efficacy(db, d[k]["ids"])
        out[k] = eff
        for model in keys(d[k]["rri"])
            d[k]["rri"][model] ./= rri.(eff)
        end
    end

    return d, out
end
# ============================================================================ #
function make_figure(d::Dict{String,Any}, metric::String="rri"; io::IO=stdout)

    names = [("isi", "ISI model"), ("ff", "Retinal\nmodel"), ("fr", "Combined\nmodel")]
    colors = [[0.,0.,0.], GREEN, PURPLE]
    idx = [(1,2), (1,3), (2,3)]

    # ------------------------------------------------------------------------ #
    # Figure 4

    h2 = figure()
    h2.set_size_inches((7,7))

    rh = [1.0, 0.3]
    rs = [0.08, 0.08, 0.05]
    cw = [1.0, 1.0, 1.0]
    cs = [0.14, 0.08, 0.08, 0.04]

    tmp = Plot.axes_layout(h2, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)
    ax2 = reshape(tmp[[1,4,2,5,3,6]], 2, 3)
    foreach(default_axes, ax2)

    exclude = Int[] #[108, 210]

    println(io, "*"^80)
    println(io, "MSEQUENCE:")
    for k in eachindex(idx)
        k1, k2 = idx[k]
        d1 = data(d["msequence"], metric, names[k1][1], exclude)
        d2 = data(d["msequence"], metric, names[k2][1], exclude)
        v, lo, hi = GAPlot.cumming_plot(d1, d2, ax=ax2[:,k], colors=colors[[k1,k2]], dcolor=BLUE)
        _, p = SimpleStats.paired_permutation_test(median, d1, d2, 5_000)

        er = mad(d1 .- d2)

        ax2[1,k].set_xticklabels([names[k1][2], names[k2][2]], fontsize=12)

        println(io, "\t$(replace(names[k1][2], "\n"=>" ")) vs. $(replace(names[k2][2], "\n"=>" "))")
        @printf(io, "\t\tmedian %s difference: %.3f bits/event (MAD %.3f 95%% CI [%.3f, %.3f] p = %.4f)\n", uppercase(metric), v, er, lo, hi, p)
        println(io)
    end

    stp = 0.01
    label = false
    for cax in ax2[1,:]
        cax.set_ylim(-0.01, 0.5)
        if !label
            cax.set_ylabel(L"\mathcal{I}_{Bernoulli}\ \mathrm{(bits/event)}", fontsize=14)
            label = true
        end
    end

    lo, up = axes_lim(ax2[2,:], 0.05)
    for (k,cax) in enumerate(ax2[2,:])
        cax.set_ylim(lo, up)
        cax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(stp))
    end

    ax2[2,1].set_ylabel("Paired median\ndifference", fontsize=14)

    foreach(x -> Plot.axes_label(h2, x[1], x[2]), zip(ax2[1,:], ["A","B","C"]))

    # ------------------------------------------------------------------------ #
    # Figure 5

    h3 = figure()
    h3.set_size_inches((7,7))

    tmp = Plot.axes_layout(h3, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)
    ax3 = reshape(tmp[[1,4,2,5,3,6]], 2, 3)
    foreach(default_axes, ax3)

    println(io, "*"^80)
    println(io, "GRATINGS:")
    for k in eachindex(idx)
        k1, k2 = idx[k]
        d1 = data(d["grating"], metric, names[k1][1])
        d2 = data(d["grating"], metric, names[k2][1])
        v, lo, hi = GAPlot.cumming_plot(d1, d2, ax=ax3[:,k], colors=colors[[k1,k2]], dcolor=RED)
        _, p = SimpleStats.paired_permutation_test(median, d1, d2)

        er = mad(d1 .- d2)

        ax3[1,k].set_xticklabels([names[k1][2], names[k2][2]], fontsize=12)

        println(io, "\t$(replace(names[k1][2], "\n"=>" ")) vs. $(replace(names[k2][2], "\n"=>" "))")
        @printf(io, "\t\tmedian %s difference: %.3f bits/event (MAD %.3f 95%% CI [%.3f, %.3f] p = %.4f)\n", uppercase(metric), v, er, lo, hi, p)
        println(io)
    end

    stp = 0.05

    label = false
    for cax in ax3[1,:]
        cax.set_ylim(-0.01, 0.5)
        if !label
            cax.set_ylabel(L"\mathcal{I}_{Bernoulli}\ \mathrm{(bits/event)}", fontsize=14)
            label = true
        end
    end

    lo, up = axes_lim(ax3[2,:], 0.05)
    for (k,cax) in enumerate(ax3[2,:])
        cax.set_ylim(lo, up)
        cax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(stp))
    end

    ax3[2,1].set_ylabel("Paired median\ndifference", fontsize=14)

    foreach(x -> Plot.axes_label(h2, x[1], x[2]), zip(ax3[1,:], ["A","B","C"]))

    # ------------------------------------------------------------------------ #
    # Figure 6

    h4 = figure()
    h4.set_size_inches((7,7))

    cs[1] += 0.01

    tmp = Plot.axes_layout(h4, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)
    ax4 = reshape(tmp[[1,4,2,5,3,6]], 2, 3)
    foreach(default_axes, ax4)

    println(io, "*"^80)
    println(io, "AWAKE:")
    for k in eachindex(idx)
        k1, k2 = idx[k]
        d1 = data(d["awake"], metric, names[k1][1])
        d2 = data(d["awake"], metric, names[k2][1])
        v, lo, hi = GAPlot.cumming_plot(d1, d2, ax=ax4[:,k], colors=colors[[k1,k2]], dcolor=GOLD)
        _, p = SimpleStats.paired_permutation_test(median, d1, d2)

        er = mad(d1 .- d2)

        ax4[1,k].set_xticklabels([names[k1][2], names[k2][2]], fontsize=12)

        println(io, "\t$(replace(names[k1][2], "\n"=>" ")) vs. $(replace(names[k2][2], "\n"=>" "))")
        @printf(io, "\t\tmedian %s difference: %.3f bits/event (MAD %.3f 95%% CI [%.3f, %.3f] p = %.4f)\n", uppercase(metric), v, er, lo, hi, p)
        println(io)
    end

    stp = 0.05

    label = false
    for cax in ax4[1,:]
        cax.set_ylim(-0.01, 0.5)
        if !label
            cax.set_ylabel(L"\mathcal{I}_{Bernoulli}\ \mathrm{(bits/event)}", fontsize=14)
            label = true
        end
    end

    lo, up = axes_lim(ax4[2,:], 0.05)
    for (k,cax) in enumerate(ax4[2,:])
        cax.set_ylim(lo, up)
        cax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(stp))
    end

    ax4[2,1].set_ylabel("Paired median\ndifference", fontsize=14)

    foreach(x -> Plot.axes_label(h2, x[1], x[2]), zip(ax4[1,:], ["A","B","C"]))

    # ------------------------------------------------------------------------ #

    return d
end
# ============================================================================ #
function run_one(::Type{T}, ret::AbstractVector{<:Real}, lgn::AbstractVector{<:Real},
    ffspan::Integer, fbspan::Integer, fbnb::Integer, bin_size::Real) where T <: RelayGLM.PerformanceMetric

    response = wasrelayed(ret, lgn)
    lm1 = 2.0 .^ range(2, 14, length=8)

    ps1 = PredictorSet()
    ps1[:ff] = Predictor(ret, ret, DefaultBasis(length=Int(ffspan), offset=2, bin_size=bin_size))
    glm1 = GLM(ps1, response, SmoothingPrior, [lm1])
    result1 = cross_validate(T, Binomial, Logistic, glm1, nfold=10, shuffle_design=true)

    ps2 = PredictorSet()
    ps2[:ff] = Predictor(ret, ret, CosineBasis(length=Int(ffspan), offset=2, nbasis=16, b=10, ortho=false, bin_size=bin_size))
    ps2[:fb] = Predictor(lgn, ret, CosineBasis(length=Int(fbspan), offset=2, nbasis=Int(fbnb), b=6, ortho=false, bin_size=bin_size))
    glm2 = GLM(ps2, response)
    result2 = cross_validate(T, Binomial, Logistic, glm2, nfold=10, shuffle_design=true)

    return result1, result2
end
# ============================================================================ #
init_output(::Type{JSDivergence}, n::Integer) = Vector{Tuple{Float64,Float64}}(undef, n)
init_output(::Type{<:RelayGLM.PerformanceMetric}, n::Integer) = Vector{Float64}(undef, n)
# ============================================================================ #
function data(d::Dict, metric::String, typ::String, exc::Vector{Int}=Int[])

    out = convert(Vector{Float64}, d[metric][typ])

    if isempty(exc)
        keep = 1:length(out)
    else
        keep = findall(!in(exc), d["ids"])
    end
    return out[keep]
end
# ============================================================================ #
function axes_lim(ax, pct::Real = 0.0)
    mn = +Inf
    mx = -Inf
    for cax in ax
        yl = cax.get_ylim()
        mn = min(mn, yl[1])
        mx = max(mx, yl[2])
    end
    r = (mx - mn) * pct
    return mn - r, mx + r
end
# ============================================================================ #
roundn(x::Real, n::Integer) = round(x / 10.0^n) * 10.0^n
# ============================================================================ #
function descriptive_stats_jsd(d::Dict, metric::String; io::IO=stdout)
    println(io, "*"^40)
    for typ in keys(d)
        for modl in ["isi", "ff", "fb"]
            p = getindex.(d[typ][metric][modl], 2)
            n = count(<(0.05), p)
            println(io, uppercase(typ), " - ", uppercase(modl))
            println("\t$(n) / $(length(p)) pairs are significanly > chance")

            data = getindex.(d[typ][metric][modl], 1)
            bs = bmedian(data)

            val, lo, hi = confint(bs, BCaConfInt(0.95), 1)

            println("\tMedian $(uppercase(metric)): ", roundn(val, -3), ", MAD: ", roundn(mad(data), -3), " [$(roundn(lo,-3)), $(roundn(hi,-3))]")

        end
        println(io, "*"^40)
    end
end
# ============================================================================ #
function descriptive_stats(d::Dict, metric::String; io::IO=stdout)
    exclude = Int[]#[108, 210]
    println(io, "*"^40)
    for typ in keys(d)
        ids = d[typ]["ids"]
        if typ == "msequence"
            kuse = findall(!in(exclude), ids)
        else
            kuse = 1:length(ids)
        end
        for modl in ["isi", "ff", "fr"]
            println(io, uppercase(typ), " - ", uppercase(modl))
            data = convert(Vector{Float64}, d[typ][metric][modl][kuse])
            bs = bmedian(data)
            val, lo, hi = confint(bs, BCaConfInt(0.95), 1)

            @printf(io, "\tMedian %s: %.3f bits/event (MAD %.3f 95%% CI [%.3f, %.3f])\n", uppercase(metric), val, mad(data)[1], lo, hi)
        end
        println(io, "*"^40)
    end
end
# ============================================================================ #
end
