module Figure3

using PaperUtils, DatabaseWrapper, RelayGLM, Progress
using Plot, UCDColors, SimpleStats

using Statistics, PyPlot, ColorTypes

const Strmbol = Union{String,Symbol}

# ============================================================================ #
function collate_data(;rmbursts::Bool=false, burst_isi::Real=0.04, burst_deadtime::Real=0.1, exclude::Dict{String,Vector{Int}}=PaperUtils.EXCLUDE)

    d = Dict{String, Any}()
    tmp = Dict{String,Strmbol}("grating" => "(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)

    ffspan = 200
    fbspan = 200
    bin_size = 0.001

    for (type, ptrn) in tmp

        db = get_database(ptrn, id -> !in(id, exclude[type]))

        d[type] = Dict{String, Any}()
        d[type]["ids"] = get_ids(db)
        d[type]["xf"] = Matrix{Float64}(undef, ffspan, length(db))
        d[type]["xse"] = Matrix{Float64}(undef, ffspan, length(db))
        d[type]["hf"] = Matrix{Float64}(undef, fbspan, length(db))
        d[type]["hse"] = Matrix{Float64}(undef, fbspan, length(db))

        for k in 1:length(db)

            ret, lgn, _, _ = get_data(db, k)
            id = get_id(db[k])

            res = run_one(ret, lgn,
                ffspan, # ff span
                fbspan, # fb span
                24,     # fb nbasis
                bin_size,
                rmbursts,
                burst_isi,
                burst_deadtime
            )

            if !res.converged
                @warn("Pair $(id) [$(type)] failed to converge")
            end

            d[type]["xf"][:,k] = get_coef(res, :ff)
            d[type]["xse"][:,k] = get_error(res, :ff)

            d[type]["hf"][:,k] = get_coef(res, :fb)
            d[type]["hse"][:,k] = get_error(res, :fb)

            show_progress(k/length(db), 0, "$(type): ", "($(k) of $(length(db)))")
        end
        println()
    end

    return d
end
# ============================================================================ #
function make_figure(d::Dict{String,Any}, ex_id::Integer=208; io::IO=stdout)

    h = figure()
    h.set_size_inches((9,9.5))

    rh = [1.0, 1.0, 1.0]
    rs = [0.06, 0.11, 0.11, 0.06]
    cw = [1.0, 1.0]
    cs = [0.10, 0.07, 0.03]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)

    foreach(default_axes, ax)

    names = Dict("msequence"=>"Binary white noise", "grating"=>"Gratings", "awake"=>"Awake")
    colors = Dict("msequence"=>BLUE, "grating"=>RED, "awake"=>GOLD)

    t_xf = range(-0.2, -0.001, length=200)
    t_hf = range(-0.2, -0.001, length=200)

    foreach((x,t) -> x.plot([t, -.001], [0,0], ":", color="black", linewidth=2), ax, fill(t_xf[1], length(ax)))

    for type in ["msequence", "grating"]

        k = findfirst(isequal(ex_id), d[type]["ids"])

        id = string(d[type]["ids"][k])

        # example pair CH-retina plot
        y = PaperUtils.normalize(d[type]["xf"][:,k])
        se = d[type]["xse"][:,k]
        ax[1].plot(t_xf, y, linewidth=3, color=colors[type], label=names[type])


        # example pair CH-LGN plot
        y = PaperUtils.normalize(d[type]["hf"][:,k])
        se = d[type]["hse"][:,k]
        ax[2].plot(t_hf, y, linewidth=3, color=colors[type], label=names[type])

        N = size(d[type]["xf"], 2)

        # population CH-retina plot
        y, lo, hi = filter_ci(PaperUtils.normalize(d[type]["xf"]))
        plot_with_error(t_xf, y, lo, hi, RGB(colors[type]...), ax[3], linewidth=3, label=names[type] * " (n=$(N))")

        # population CH-LGN plot
        y, lo, hi = filter_ci(PaperUtils.normalize(d[type]["hf"]))
        plot_with_error(t_hf, y, lo, hi, RGB(colors[type]...), ax[4], linewidth=3, label=names[type] * " (n=$(N))")
    end

    N = size(d["awake"]["xf"], 2)
    ax[5].plot(t_xf, PaperUtils.normalize(d["awake"]["xf"]), color="gray", linewidth=1, alpha=1.0)
    ax[5].plot(t_xf, mean(PaperUtils.normalize(d["awake"]["xf"]), dims=2), color=GOLD, linewidth=3, label="Population mean (n=$(N))")

    ax[6].plot(t_xf, PaperUtils.normalize(d["awake"]["hf"]), color="gray", linewidth=1, alpha=1.0)
    ax[6].plot(t_xf, mean(PaperUtils.normalize(d["awake"]["hf"]), dims=2), color=GOLD, linewidth=3, label="Population mean (n=$(N))")

    ax[1].set_title("CH-retinal: Pair $(ex_id)", fontsize=16)
    ax[3].set_title("CH-retinal: Population", fontsize=16)
    ax[5].set_title("CH-retinal: Awake", fontsize=16)

    ax[1].set_ylim(-0.22, 0.55)
    ax[2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax[2].set_title("CH-LGN: Pair $(ex_id)", fontsize=16)

    ax[3].set_ylim(-0.15, 0.4)
    ax[4].set_title("CH-LGN: Population", fontsize=16)
    ax[6].set_title("CH-LGN: Awake", fontsize=16)


    labels = ["A","B","C","D","E","F"]
    for k in 1:length(ax)
        if mod(k, 2) == 0
            Plot.axes_label(h, ax[k], labels[k], -0.23)
        else
            ax[k].legend(frameon=false, fontsize=14)
            ax[k].set_ylabel("Filter weight (A.U.)", fontsize=14)
            Plot.axes_label(h, ax[k], labels[k])
        end
        if k > 4
            ax[k].set_xlabel("Time before spike (seconds)", fontsize=14)
        end
        ax[k].set_xlim(t_xf[1]-0.004, t_xf[end]+0.004)
    end

    foreach(x->x.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04)), ax)

    return h
end
# ============================================================================ #
"remove all non-cardinal burst spikes from a spike train"
function rm_bursts(lgn::Vector{Float64}, burst_isi::Real, burst_deadtime::Real)
    kb = PaperUtils.burst_spikes(lgn, burst_isi, burst_deadtime)
    krm = Vector{Int}()
    for k in eachindex(kb)
        append!(krm, kb[k][2:end])
    end
    keep = setdiff(1:length(lgn), krm)
    return sort(lgn[keep])
end
# ============================================================================ #
function run_one(ret::AbstractVector{<:Real}, lgnd::AbstractVector{<:Real},
    ffspan::Integer, fbspan::Integer, fbnb::Integer, bin_size::Real,
    rmbursts::Bool=false, burst_isi::Real=0.04, burst_deadtime::Real=0.1)

    if rmbursts
        lgn = rm_bursts(lgnd, burst_isi, burst_deadtime)
    else
        lgn = lgnd
    end
    response = wasrelayed(ret, lgn)

    lm = 2.0 .^ range(-3.5, 3, length=5)

    ps = PredictorSet()
    ps[:ff] = Predictor(ret, ret, CosineBasis(length=Int(ffspan), offset=2, nbasis=16, b=10, ortho=false, bin_size=bin_size))
    ps[:fb] = Predictor(lgn, ret, CosineBasis(length=Int(fbspan), offset=2, nbasis=Int(fbnb), b=8, ortho=false, bin_size=bin_size))

    glm = GLM(ps, response, RidgePrior, [lm])
    result = cross_validate(RRI, Binomial, Logistic, glm, nfold=10, shuffle_design=true)

    return result
end
# ============================================================================ #
end
