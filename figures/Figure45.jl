module Figure45

using GAPlot, Plot, UCDColors, SimpleStats, DatabaseWrapper, RelayGLM, Progress
using RelayGLM.RelayISI, PaperUtils, HyperParameters
using LinearAlgebra, Statistics, PyPlot, ColorTypes, Bootstrap, Printf

const Strmbol = Union{String,Symbol}

# ============================================================================ #
function collate_data(::Type{T}) where T <: RelayGLM.PerformanceMetric

    bin_size = 0.001
    mxisi = roundn.(10.0 .^ range(log10(0.03), log10(0.5), length=8), -3)

    d = Dict{String, Any}()
    tmp = Dict{String,Strmbol}("grating" => "(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)
    par = HyperParameters.load_parameters()

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
function make_figure(d::Dict{String,Any}, metric::String="rri"; io::IO=stdout)

    names = [("isi", "ISI model"), ("ff", "Retinal\nmodel"), ("fb", "Combined\nmodel")]
    colors = [GOLD, GREEN, PURPLE]
    idx = [(1,2), (1,3), (2,3)]

    denom = metric == "jsd" ? log(2.0) : 1.0
    km = metric == "jsd" ? 1 : 0

    # ------------------------------------------------------------------------ #
    # Figure 4

    h2 = figure()
    h2.set_size_inches((7,7))

    rh = [1.0, 0.3]
    rs = [0.08, 0.08, 0.05]
    cw = [1.0, 1.0, 1.0]
    cs = [0.13, 0.08, 0.08, 0.04]

    tmp = Plot.axes_layout(h2, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)
    ax2 = reshape(tmp[[1,4,2,5,3,6]], 2, 3)
    foreach(default_axes, ax2)

    exclude = [108, 210]

    println(io, "*"^80)
    println(io, "MSEQUENCE:")
    for k in eachindex(idx)
        k1, k2 = idx[k]
        d1 = data(d["msequence"], metric, names[k1][1], km, exclude)
        d2 = data(d["msequence"], metric, names[k2][1], km, exclude)
        v, lo, hi = GAPlot.cumming_plot(d1 ./ denom, d2 ./ denom, ax=ax2[:,k], colors=colors[[k1,k2]], dcolor=BLUE)
        _, p = SimpleStats.paired_permutation_test(median, d1 ./ denom, d2 ./ denom)

        er = mad((d1 ./ denom) .- (d2 ./ denom))

        ax2[1,k].set_xticklabels([names[k1][2], names[k2][2]], fontsize=12)

        println(io, "\t$(replace(names[k1][2], "\n"=>" ")) vs. $(replace(names[k2][2], "\n"=>" "))")
        @printf(io, "\t\tmedian %s difference: %.3f bits/event (MAD %.3f 95%% CI [%.3f, %.3f] p = %.4f)\n", uppercase(metric), v, er, lo, hi, p)
        println(io)
    end

    stp = 0.05
    label = false
    for cax in ax2[1,:]
        if metric == "roca"
            cax.set_ylim(0.48, 1.0)
            cax.set_ylabel(L"ROC_{area}", fontsize=14)
            stp = 0.03
        elseif metric == "pra"
            cax.set_ylim(0.0, 1.0)
            cax.set_ylabel("Precision-recall", fontsize=14)
        elseif metric == "jsd"
            cax.set_ylim(-0.02, 0.75)
            cax.set_ylabel("JS divergence", fontsize=14)
        elseif metric == "rbili"
            stp = 0.03
            cax.set_ylim(-0.02, 0.55)
            cax.set_ylabel("Relative Likelihood", fontsize=14)
        elseif metric == "rri"
            stp = 0.02
            cax.set_ylim(-0.01, 0.5)
            if !label
                cax.set_ylabel(L"\mathcal{I}_{Bernoulli}\ \mathrm{(bits/event)}", fontsize=14)
                label = true
            end
        else
            error("Invalid metric: $(metric)")
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
        d1 = data(d["grating"], metric, names[k1][1], km)
        d2 = data(d["grating"], metric, names[k2][1], km)
        v, lo, hi = GAPlot.cumming_plot(d1./ denom, d2 ./ denom, ax=ax3[:,k], colors=colors[[k1,k2]], dcolor=RED)
        _, p = SimpleStats.paired_permutation_test(median, d1 ./ denom, d2 ./ denom)

        er = mad((d1 ./ denom) .- (d2 ./ denom))

        ax3[1,k].set_xticklabels([names[k1][2], names[k2][2]], fontsize=12)

        println(io, "\t$(replace(names[k1][2], "\n"=>" ")) vs. $(replace(names[k2][2], "\n"=>" "))")
        @printf(io, "\t\tmedian %s difference: %.3f bits/event (MAD %.3f 95%% CI [%.3f, %.3f] p = %.4f)\n", uppercase(metric), v, er, lo, hi, p)
        println(io)
    end

    stp = 0.05

    label = false
    for cax in ax3[1,:]
        if metric == "roca"
            cax.set_ylim(0.48, 1.0)
            cax.set_ylabel(L"ROC_{area}", fontsize=14)
        elseif metric == "pra"
            cax.set_ylim(0.0, 1.0)
            cax.set_ylabel("Precision-recall", fontsize=14)
        elseif metric == "jsd"
            cax.set_ylim(-0.02, 0.75)
            cax.set_ylabel("JS divergence", fontsize=14)
        elseif metric == "rbili"
            cax.set_ylim(-0.02, 0.5)
            cax.set_ylabel("Relative Likelihood", fontsize=14)
        elseif metric == "rri"
            stp = 0.05
            cax.set_ylim(-0.01, 0.5)
            if !label
                cax.set_ylabel(L"\mathcal{I}_{Bernoulli}\ \mathrm{(bits/event)}", fontsize=14)
                label = true
            end
        else
            error("Invalid metric: $(metric)")
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
function data(d::Dict, metric::String, typ::String, idx::Integer, exc::Vector{Int}=Int[])

    if idx < 1
        out = d[metric][typ]
    else
        out = getfield.(d[metric][typ], idx)
    end

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
        for modl in ["isi", "ff", "fb"]
            println(io, uppercase(typ), " - ", uppercase(modl))
            data = d[typ][metric][modl][kuse]
            bs = bmedian(data)
            val, lo, hi = confint(bs, BCaConfInt(0.95), 1)

            @printf(io, "\tMedian %s: %.3f bits/event (MAD %.3f 95%% CI [%.3f, %.3f])\n", uppercase(metric), val, mad(data), lo, hi)
        end
        println(io, "*"^40)
    end
end
# ============================================================================ #
end
