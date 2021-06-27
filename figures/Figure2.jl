module Figure2

using PaperUtils
using Plot, UCDColors, SimpleStats, DatabaseWrapper, RelayGLM, Progress

using Statistics, PyPlot, ColorTypes

import RelayGLM.RelayISI

const Strmbol = Union{String,Symbol}

# ============================================================================ #
function collate_data()

    span = 200
    bin_size = 0.001
    isimax = span * bin_size
    sigma = 0.002

    d = Dict{String, Any}()
    tmp = Dict{String,Strmbol}("grating" => "(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)

    for (type, ptrn) in tmp

        db = get_database(ptrn, id -> !in(id, PaperUtils.EXCLUDE[type]))
        d[type] = Dict{String, Any}()
        d[type]["ids"] = get_ids(db)
        d[type]["isi"] = Matrix{Float64}(undef, span-2, length(db))
        d[type]["xf"] = Matrix{Float64}(undef, span, length(db))
        d[type]["xse"] = Matrix{Float64}(undef, span, length(db))


        for k in 1:length(db)

            id = get_id(db[k])
            ret, lgn, _, _ = get_data(db, k)

            isi, status = RelayISI.spike_status(ret, lgn)
            edges, eff = RelayISI.get_eff(isi, status, 1:length(isi), sigma, bin_size, span * bin_size)

            d[type]["isi"][:, k] = eff

            if !haskey(d[type], "labels")
                d[type]["labels"] = edges[1:end-1] .+ step(edges)/2
            end

            result = run_one(ret, lgn, span, bin_size)

            if !result.converged
                @warn("Pair $(id) [$(type)] failed to converge")
            end

            d[type]["xf"][:, k] = get_coef(result, :ff)
            d[type]["xse"][:, k] = get_error(result, :ff)

            # clear_lines_above(1)
            show_progress(k/length(db), 0, "$(type): ", "($(k) of $(length(db)))")
        end
    end
    return d
end
# ============================================================================ #
function make_figure(d::Dict{String,Any}, ex_id::Integer=208, awake_id::String="02MAY270")

    h, ax = subplots(2, 2)
    h.set_size_inches((10,8))
    foreach(default_axes, ax)

    colors = Dict("msequence"=>BLUE, "grating"=>RED, "awake"=>GOLD)
    labels = Dict("msequence"=>"Binary white noise", "grating"=>"Gratings", "awake"=>"Awake")

    for type in ["msequence", "grating", "awake"]

        N = length(d[type]["ids"])

        if type == "awake"
            k = findfirst(isequal(awake_id), d[type]["ids"])
        else
            k = findfirst(isequal(ex_id), d[type]["ids"])
        end

        id = string(d[type]["ids"][k])

        x = d[type]["labels"]
        y = d[type]["isi"][:, k]

        ax[1,1].plot(x, y, linewidth=3, color=colors[type], label=labels[type] * " ($(id))")

        y, lo, hi = filter_ci(mean_norm(d[type]["isi"], dims=1))
        plot_with_error(x, y, lo, hi, RGB(colors[type]...), ax[1,2], linewidth=3, label=labels[type] * " (n=$(N))")


        x = range(-0.2, -0.001, length=200)
        y = d[type]["xf"][:, k]
        se = d[type]["xse"][:, k]
        plot_with_error(x, y, se, RGB(colors[type]...), ax[2,1], linewidth=3, label=labels[type] * " ($(id))")

        y, lo, hi = filter_ci(normalize(d[type]["xf"]))
        plot_with_error(x, y, lo, hi, RGB(colors[type]...), ax[2,2], linewidth=3, label=labels[type] * " (n=$(N))")
    end

    ax[1,1].legend(frameon=false, fontsize=14)
    ax[1,1].set_xlabel("Inter-spike interval (seconds)", fontsize=14)
    ax[1,1].set_ylabel("Efficacy", fontsize=14)
    ax[1,1].set_title("ISI-efficacy: examples", fontsize=16)

    ax[1,2].set_ylim(0, ax[1,2].get_ylim()[2])
    ax[1,2].set_ylabel("Normalized efficacy (A.U.)", fontsize=14)
    ax[1,2].legend(frameon=false, fontsize=14, loc="upper right", bbox_to_anchor=(1.05, 1.0))
    ax[1,2].set_xlabel("Inter-spike interval (seconds)", fontsize=14)
    ax[1,2].set_title("ISI-efficacy: population", fontsize=16)

    ax[2,1].set_xlabel("Time before spike (seconds)", fontsize=14)
    ax[2,1].set_ylabel("Filter weight (A.U.)", fontsize=14)
    ax[2,1].plot(ax[2,1].get_xlim(), [0,0], "--", linewidth=2, color="gray")
    ax[2,1].legend(frameon=false, fontsize=14)
    ax[2,1].set_xlim(-0.2, 0.0)
    ax[2,1].set_title("RH-retinal: examples", fontsize=16)

    ax[2,2].set_xlabel("Time before spike (seconds)", fontsize=14)
    ax[2,2].plot(ax[2,2].get_xlim(), [0,0], "--", linewidth=2, color="gray")
    ax[2,2].legend(frameon=false, fontsize=14)
    ax[2,2].set_xlim(-0.2, 0.0)
    ax[2,2].set_title("RH-retinal: population", fontsize=16)

    foreach(x -> Plot.axes_label(h, x[1], x[2]), zip(ax, ["A","C","B","D"]))

    foreach(x->x.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05)), ax)

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
function mean_norm(x::Matrix{<:Real}; dims::Integer=2)
    mn = mean(x, dims=dims)
    id = [2,1][dims]
    y = copy(x)
    for (k, sl) in enumerate(eachslice(y, dims=id))
        sl ./= mn[k]
    end
    return y
end
# ============================================================================ #
end
