module FigureS6

# reviewer response figure 1

using PyPlot, StatsBase, RelayGLM, RelayGLM.RelayUtils, PaperUtils
using DatabaseWrapper, RejectionSampling, Plot, UCDColors, Progress

function collate_data()

    tmp = ["grating" => "(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand]
    all_ids = Dict("grating" => [207, 208, 201], "msequence"=>[207, 208, 201], "awake"=>[200009210, 200106030, 200002040])

    d = Dict{String,Any}()

    for (type, ptrn) in tmp
        db = get_database(ptrn, id -> !in(id, PaperUtils.EXCLUDE[type]))

        # eff = get_efficacy(db)
        # ks = sortperm(eff, rev=true)
        # ids = get_ids(db)[ks[1:3]]

        ids = all_ids[type]

        d[type] = Dict{String, Any}()

        d[type]["ids"] = ids
        d[type]["coef"] = zeros(200, length(ids))
        d[type]["coef_err"] = zeros(200, length(ids))
        d[type]["sim"] = zeros(200, length(ids))
        d[type]["sim_err"] = zeros(200, length(ids))

        show_progress(0.0, 0, "$(type): ", "(0 of $(length(ids)))")

        for (k, id) in enumerate(ids)
            t1 = time()
            ret, lgn, _, _ = get_data(db, id=id)
            coef, coef_err, sim, sim_err = run_one(ret, lgn, 200, 20)

            d[type]["coef"][:,k] .= coef
            d[type]["coef_err"][:,k] .= coef_err
            d[type]["sim"][:,k] .= sim
            d[type]["sim_err"][:,k] .= sim_err

            elap = time() - t1
            show_progress(k/length(ids), 0, "$(type): ", "($(k) of $(length(ids)) @ $(elap))")
        end
    end

    return d
end

function make_figure(d)

    h = figure()
    h.set_size_inches((10,8))

    rh = [1.0, 1.0, 1.0]
    rs = [0.08, 0.13, 0.13, 0.08]
    cw = [1.0, 1.0, 1.0]
    cs = [0.08, 0.06, 0.06, 0.03]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)
    ax = permutedims(reshape(ax, 3, 3), (2,1))

    foreach(default_axes, ax)

    len = size(d["grating"]["coef"], 1)

    t = -0.201:0.001:-0.002

    for (k, stim) in enumerate(["grating", "msequence", "awake"])
        for j in 1:size(d[stim]["coef"], 2)

            # plot_with_error(t, d[stim]["coef"][:,j], d[stim]["coef_err"][:,j], PURPLE, ax[k,j], linewidth=2)
            ax[k,j].plot(t, d[stim]["coef"][:,j], color=PURPLE, linewidth=2, label="generating filter")
            plot_with_error(t, d[stim]["sim"][:,j], d[stim]["sim_err"][:,j], GREEN, ax[k,j], linewidth=2, label="learned filter")
            ax[k,j].plot([t[1], t[end]], [0,0], "--", color="grey", linewidth=1.5, zorder=-10)
            ax[k,j].set_title("Pair " * string(d[stim]["ids"][j]), fontsize=16)
        end
    end

    foreach(ax[3,:]) do cax
        cax.set_xlabel("Time before spike (seconds)", fontsize=14)
    end

    foreach(ax[:,1]) do cax
        cax.set_ylabel("Filter weight (A.U.)", fontsize=14)
    end

    ax[1,1].legend(frameon=false, fontsize=14)

    h.text(0.5, 0.99, "Gratings", fontsize=18, color=RED, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.67, "Binary white noise", fontsize=18, color=BLUE, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.34, "Awake", fontsize=18, color=GOLD, horizontalalignment="center", verticalalignment="top")

    labels = ["A","B","C","D","E","F","G","H","I"]
    foreach((cax,lab)->Plot.axes_label(h, cax, lab), permutedims(ax, (2,1)), labels)

end

function get_efficacy(db)
    eff = zeros(length(db))
    for k in 1:length(db)
        ret, lgn, _, _ = get_data(db, k)
        status = wasrelayed(ret, lgn)
        eff[k] = sum(status) / length(ret)
    end
    return eff
end

function run_one(ret::AbstractVector{<:Real}, lgn::AbstractVector{<:Real}, len::Integer=200, niter::Integer=10)

    lm = 2.0 .^ range(1, 12, length=8)

    ps = PredictorSet();
    ps[:retina] = Predictor(ret, ret, DefaultBasis(length=len, offset=2, bin_size=0.001))
    response = wasrelayed(ret, lgn)
    glm = GLM(ps, response, SmoothingPrior, [lm])

    result = cross_validate(RRI, Binomial, Logistic, glm, nfold=10, shuffle_design=true)

    obs_coef = get_coef(result, :retina)
    obs_error = get_error(result, :retina)

    # sampler object for sampling from the observed RGC's ISI distribution
    edges = 0.001:0.001:0.2
    h = fit(Histogram, diff(ret), edges)
    s = get_sampler(centers(edges), h.weights ./ trapz(h.weights))

    status = Vector{Bool}(undef, length(ret))

    sim_coef = zeros(len, niter)

    for k in 1:niter

        # get simulated RGC spike train
        isi = RejectionSampling.sample(s, length(ret))
        sim_ts = cumsum(isi) .+ ret[1]

        ps2 = PredictorSet()
        ps2[:retina] = Predictor(sim_ts, sim_ts, DefaultBasis(length=len, offset=2, bin_size=0.001))
        dm = RelayGLM.generate(ps2)

        pred = RelayUtils.logistic_turbo!(dm * result.coef)

        @inbounds for k in eachindex(pred)
            status[k] = rand() <= pred[k] ? true : false
        end

        glm2 = GLM(ps2, status, SmoothingPrior, [lm])
        result2 = cross_validate(RRI, Binomial, Logistic, glm2, nfold=10, shuffle_design=true)
        sim_coef[:,k] .= get_coef(result2, :retina)
    end

    val, lo, hi = filter_ci(sim_coef)

    return obs_coef, obs_error, val, hi .- lo
end

function main()
    db = get_database("(?:contrast|area|grating)")
    ret, lgn, _, _ = get_data(db, id=208)

    ps = PredictorSet();
    ps[:retina] = Predictor(ret, ret, DefaultBasis(length=60, offset=2, bin_size=0.001))
    response = wasrelayed(ret, lgn)

    @info("Average efficacy [data]: $(sum(response) / length(response))")

    glm = GLM(ps, response)
    result = cross_validate(RRI, Binomial, Logistic, glm, nfold=10, shuffle_design=true)

    tmp = vec(sum(RelayGLM.generate(ps)[:,2:end], dims=1))

    sim_ts = get_ts(ret)

    ps2 = PredictorSet()
    ps2[:retina] = Predictor(sim_ts, sim_ts, DefaultBasis(length=60, offset=2, bin_size=0.001))
    dm = RelayGLM.generate(ps2)

    t = -0.061:0.001:-0.002

    h, ax = subplots(1, 2)
    h.set_size_inches((10,5))
    foreach(default_axes, ax)

    ax[1].plot(t, tmp, linewidth=2, label="data")
    ax[1].plot(t, vec(sum(dm[:, 2:end], dims=1)), linewidth=2, label="simulation")
    ax[1].set_title("Auto-correlations (from design matrix)", fontsize=16)

    pred = RelayUtils.logistic_turbo!(dm * result.coef)

    status = Vector{Bool}(undef, length(pred))
    @inbounds for k in eachindex(pred)
        status[k] = rand() <= pred[k] ? true : false
    end

    @info("Average efficacy [sim]: $(sum(status) / length(status))")

    glm2 = GLM(ps2, status)
    result2 = cross_validate(RRI, Binomial, Logistic, glm2, nfold=10, shuffle_design=true)

    ax[2].plot([t[1], t[end]], [0, 0], "--", linewidth=1, color="gray")
    ax[2].plot(t, get_coef(result, :retina), linewidth=2, label="fit to data")
    ax[2].plot(t, get_coef(result2, :retina), linewidth=2, label="fit to simulation")
    ax[2].set_title("GLM filters", fontsize=16)

    for cax in ax
        cax.legend(frameon=false, fontsize=14)
        cax.set_xlabel("Time before spike (seconds)", fontsize=14)
    end

    ax[1].set_ylabel("Spike count", fontsize=14)
    ax[2].set_ylabel("Filter magnitude (A.U.)", fontsize=14)

    ax[1].set_ylim(0, ax[1].get_ylim()[2])

    h.tight_layout()

    return sim_ts, status
end

function get_ts(ts::AbstractVector{<:Real})
    edges = 0.001:0.001:0.2
    h = fit(Histogram, diff(ts), edges)

    s = get_sampler(centers(edges), h.weights ./ trapz(h.weights))

    isi = RejectionSampling.sample(s, length(ts))

    return cumsum(isi) .+ ts[1]
end

centers(x::AbstractRange) = x[1:end-1] .+ step(x) / 2

end
