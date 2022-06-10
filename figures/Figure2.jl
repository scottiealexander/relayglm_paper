module Figure2

using PaperUtils
using Plot, UCDColors, SimpleStats, DatabaseWrapper, RelayGLM, Progress

using Statistics, PyPlot, ColorTypes

import RelayGLM.RelayISI

const Strmbol = Union{String,Symbol}

# ============================================================================ #
function collate_data(;exclude::Dict{String,Vector{Int}}=PaperUtils.EXCLUDE)

    span = 200
    bin_size = 0.001
    isimax = span * bin_size
    sigma = 0.002

    d = Dict{String, Any}()
    tmp = Dict{String,Strmbol}("grating" => "(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)

    for (type, ptrn) in tmp

        db = get_database(ptrn, id -> !in(id, exclude[type]))
        d[type] = Dict{String, Any}()
        d[type]["ids"] = get_ids(db)
        d[type]["isi"] = Matrix{Float64}(undef, span-2, length(db))
        d[type]["xf"] = Matrix{Float64}(undef, span, length(db))
        d[type]["xse"] = Matrix{Float64}(undef, span, length(db))
        d[type]["efficacy"] = Vector{Float64}(undef, length(db))

        for k in 1:length(db)

            id = get_id(db[k])
            ret, lgn, _, _ = get_data(db, k)

            isi, status = RelayISI.spike_status(ret, lgn)
            edges, eff = RelayISI.get_eff(isi, status, 1:length(isi), sigma, bin_size, span * bin_size)

            d[type]["isi"][:, k] .= eff

            if !haskey(d[type], "labels")
                d[type]["labels"] = edges[1:end-1] .+ step(edges)/2
            end

            result = run_one(ret, lgn, span, bin_size)

            if !result.converged
                @warn("Pair $(id) [$(type)] failed to converge")
            end

            d[type]["xf"][:, k] .= get_coef(result, :ff)
            d[type]["xse"][:, k] .= get_error(result, :ff)

            d[type]["efficacy"][k] = sum(status) / length(isi)

            # clear_lines_above(1)
            show_progress(k/length(db), 0, "$(type): ", "($(k) of $(length(db)))")
        end
    end
    return d
end
# ============================================================================ #
function make_figure(d::Dict{String,Any}, ex_id::Integer=208)

    h, ax = subplots(3, 2)
    h.set_size_inches((9,9.5))
    foreach(default_axes, ax)

    # swap upper-right and left-middle axes so that we have all
    # ISI-efficacy plots in the leff column and GLM filter in the right column
    # yeah... I know, cringe away
    ax[2,1], ax[1,2] = ax[1,2], ax[2,1]

    colors = Dict("msequence"=>BLUE, "grating"=>RED, "awake"=>GOLD)
    labels = Dict("msequence"=>"Binary white noise", "grating"=>"Gratings", "awake"=>"Awake")

    t_xf = range(-0.2, -0.001, length=200)

    for type in ["msequence", "grating"]

        N = length(d[type]["ids"])

        k = findfirst(isequal(ex_id), d[type]["ids"])

        id = string(d[type]["ids"][k])

        x = d[type]["labels"]
        y = d[type]["isi"][:, k]

        ax[1,1].plot(x, y, linewidth=3, color=colors[type], label=labels[type])

        y, lo, hi = filter_ci(d[type]["isi"] ./ reshape(d[type]["efficacy"], 1, :))
        plot_with_error(x, y, lo, hi, RGB(colors[type]...), ax[1,2], linewidth=3, label=labels[type] * " (n=$(N))")

        y = d[type]["xf"][:, k]
        se = d[type]["xse"][:, k]
        plot_with_error(t_xf, y, se, RGB(colors[type]...), ax[2,1], linewidth=3, label=labels[type])

        y, lo, hi = filter_ci(normalize(d[type]["xf"]))
        plot_with_error(t_xf, y, lo, hi, RGB(colors[type]...), ax[2,2], linewidth=3, label=labels[type] * " (n=$(N))")
    end

    N = length(d["awake"]["ids"])
    tmp = d["awake"]["isi"] ./ reshape(d["awake"]["efficacy"], 1, :)
    ax[3,1].plot(d["awake"]["labels"], tmp, color="gray", linewidth=1, alpha=1.0)
    ax[3,1].plot(d["awake"]["labels"], mean(tmp, dims=2), color=GOLD, linewidth=3, label="Population mean (n=$(N))")

    # ax[3,1].plot(d["awake"]["labels"], d["awake"]["isi"], color="gray", linewidth=1, alpha=1.0)
    # ax[3,1].plot(d["awake"]["labels"], mean(d["awake"]["isi"], dims=2), color=GOLD, linewidth=3, label="Population mean (n=$(N))")

    ax[3,2].plot([t_xf[1], t_xf[end]], [0,0], "--", linewidth=2, color="black")
    ax[3,2].plot(t_xf, PaperUtils.normalize(d["awake"]["xf"]), color="gray", linewidth=1, alpha=1.0)
    ax[3,2].plot(t_xf, mean(PaperUtils.normalize(d["awake"]["xf"]), dims=2), color=GOLD, linewidth=3, label="Population mean (n=$(N))")

    format_axes(ax[1,1], "Inter-spike interval (seconds)", "Efficacy", true; title="ISI-efficacy: Pair $(ex_id)")
    format_axes(ax[1,2], "Inter-spike interval (seconds)", "Normalized efficacy\n(A.U.)", false; title="ISI-efficacy: Population")

    ax[1,2].legend(frameon=false, fontsize=14, loc="upper right", bbox_to_anchor=(1.1, 1.0))

    format_axes(ax[2,1], "Time before spike (seconds)", "Filter weight (A.U.)", false; title="RH-retinal: Pair $(ex_id)", zero=true, xlim=(-0.2, 0.0))
    format_axes(ax[2,2], "Time before spike (seconds)", "Filter weight (A.U.)", false; title="RH-retinal: Population", zero=true, xlim=(-0.2, 0.0))

    format_axes(ax[3,1], "Inter-spike interval (seconds)", "Normalized efficacy\n(A.U.)", true; title="ISI-efficacy: Awake")
    format_axes(ax[3,2], "Time before spike (seconds)", "Filter weight (A.U.)", false; title="RH-retinal: Awake", zero=true, xlim=(-0.2, 0.0))

    foreach(x->x.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05)), ax)

    ax[1,2].set_ylim(0, ax[1,2].get_ylim()[2])
    ax[1,2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))

    ax[3,1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))

    foreach((x,l) -> Plot.axes_label(h, x, l), ax, ["A","B","E","C","D","F"])

    h.tight_layout()

    return h
end
# ============================================================================ #
function run_one(ret::Vector{Float64}, lgn::Vector{Float64}, span::Integer, bin_size::Real=0.001)

    response = wasrelayed(ret, lgn)

    lm = 2.0 .^ range(1, 12, length=8)

    ps = PredictorSet()
    ps[:ff] = Predictor(ret, ret, DefaultBasis(length=span, offset=2, bin_size=0.001))
    glm = GLM(ps, response, SmoothingPrior, [lm])

    return cross_validate(RRI, Binomial, Logistic, glm, nfold=10, shuffle_design=true)
end
# ============================================================================ #
function format_axes(ax, xlab::String, ylab::String, leg::Bool;
     xlim=ax.get_xlim(), ylim=ax.get_ylim(), title::String="", zero::Bool=false)

     ax.set_xlabel(xlab, fontsize=14)
     ax.set_ylabel(ylab, fontsize=14)
     if leg
         ax.legend(frameon=false, fontsize=14)
     end
     ax.set_xlim(xlim)
     ax.set_ylim(ylim)

     if !isempty(title)
         ax.set_title(title, fontsize=16)
     end

     if zero
         ax.plot(xlim, [0, 0], "--", linewidth=2, color="black", zorder=0)
     end

     return ax
 end
# ============================================================================ #
end
