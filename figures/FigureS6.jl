module FigureS6

using PyPlot, StatsBase, RelayGLM, RelayGLM.RelayUtils
using PairsDB, RejectionSampling, Plot

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
    ax[2].plot(t, get_coef(result2, :retina), linewidth=2, label="fit to stimulation")
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
