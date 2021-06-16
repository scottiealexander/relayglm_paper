module Figure3

using PaperUtils, PairsDB, RelayGLM, Progress
using Plot, UCDColors

using Statistics, PyPlot, ColorTypes

# ============================================================================ #
function collate_data(rmbursts::Bool=false, burst_isi::Real=0.04, burst_deadtime::Real=0.1)

    d = Dict{String, Any}()
    tmp = Dict{String,String}("grating" => "(?:contrast|area|grating)", "msequence"=>"msequence")

    ffspan = 200
    fbspan = 200
    bin_size = 0.001

    for (type, ptrn) in tmp

        db = get_database(ptrn, id -> !in(id, PaperUtils.EXCLUDE[type]))

        d[type] = Dict{String, Any}()
        d[type]["ids"] = first.(db)
        d[type]["xf"] = Matrix{Float64}(undef, ffspan, length(db))
        d[type]["xse"] = Matrix{Float64}(undef, ffspan, length(db))
        d[type]["hf"] = Matrix{Float64}(undef, fbspan, length(db))
        d[type]["hse"] = Matrix{Float64}(undef, fbspan, length(db))

        for k in 1:length(db)

            ret, lgn, _, _ = get_data(db, k)
            id = first(db[k])

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
                @warn("Pair $(first(db[k])) [$(type)] failed to converge")
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
    h.set_size_inches((10,7.5))

    rh = [1.0, 1.0]
    rs = [0.08, 0.13, 0.08]
    cw = [1.0, 1.0]
    cs = [0.10, 0.05, 0.03]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)

    foreach(default_axes, ax)

    names = Dict("msequence"=>"Binary white noise", "grating"=>"Gratings")
    colors = Dict("msequence"=>BLUE, "grating"=>RED)

    t_xf = range(-0.2, -0.001, length=200)
    t_hf = range(-0.2, -0.001, length=200)

    for type in ["msequence", "grating"]
        k = findfirst(isequal(ex_id), d[type]["ids"])

        # example pair CH-retina plot
        y = PaperUtils.normalize(d[type]["xf"][:,k])
        se = d[type]["xse"][:,k]
        ax[1].plot(t_xf, y, linewidth=3, color=colors[type], label=names[type])
        ax[1].plot([t_xf[1], -.001], [0,0], ":", color="gray", linewidth=2)

        # example pair CH-LGN plot
        y = PaperUtils.normalize(d[type]["hf"][:,k])
        se = d[type]["hse"][:,k]
        ax[2].plot(t_hf, y, linewidth=3, color=colors[type], label=names[type])
        ax[2].plot([t_hf[1], -.001], [0,0], ":", color="gray", linewidth=2)

        N = size(d[type]["xf"], 2)

        # population CH-retina plot
        y, lo, hi = filter_ci(PaperUtils.normalize(d[type]["xf"]))
        plot_with_error(t_xf, y, lo, hi, RGB(colors[type]...), ax[3], linewidth=3, label=names[type] * "(n=$(N))")

        ax[3].plot([t_xf[1], -.001], [0,0], ":", color="gray", linewidth=2)

        # population CH-LGN plot
        y, lo, hi = filter_ci(PaperUtils.normalize(d[type]["hf"]))
        plot_with_error(t_hf, y, lo, hi, RGB(colors[type]...), ax[4], linewidth=3, label=names[type] * " (n=$(N))")

        ax[4].plot([t_hf[1], -.001], [0,0], ":", color="gray", linewidth=2)
    end

    ax[1].set_title("CH-retinal: Pair ID $(ex_id)", fontsize=16)

    ax[3].set_title("CH-retinal: Population", fontsize=16)

    ax[1].set_ylim(-0.22, 0.55)
    ax[2].set_ylim(-0.22, 0.55)
    ax[2].set_yticklabels([])
    ax[2].set_title("CH-LGN: Pair ID $(ex_id)", fontsize=16)

    yt = ax[3].get_yticks()

    ax[3].set_ylim(-0.15, 0.4)
    ax[4].set_ylim(-0.15, 0.4)
    ax[4].set_yticklabels([])
    ax[4].set_title("CH-LGN: Population", fontsize=16)

    for k in 1:4
        if mod(k, 2) == 0
            ax[k].legend(frameon=false, fontsize=14)
        end
        if k == 1 || k == 3
            ax[k].set_ylabel("Filter weight (A.U.)", fontsize=14)
        end
        if 2 < k < 5
            ax[k].set_xlabel("Time before spike (seconds)", fontsize=14)
        end
    end

    Plot.axes_label(h, ax[1], "A")
    Plot.axes_label(h, ax[2], "B", -0.23)
    Plot.axes_label(h, ax[3], "C")
    Plot.axes_label(h, ax[4], "D", -0.23)

    ax[1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))
    ax[3].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))

    ax[2].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))
    ax[4].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))

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
