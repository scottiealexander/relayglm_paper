module HyperParameters

using PaperUtils
using PairsDB, RelayGLM, Progress

using Statistics, Optim, Dates, JSON

export FFSpan, FBSpan, FBBasis

const DBFILE = joinpath(@__DIR__, "hyper_parameters.json")
const NSPAN = 10
const NBASIS = 10
# ============================================================================ #
logistic(x::AbstractVector{<:Real}, p::Vector) = (p[1] ./ (1.0 .+ exp.(-p[2] .* (x .- p[3])))) .+ p[4]
decay(x::AbstractVector{<:Real}, p::Vector) = p[1] .* exp.(-p[2] .* x) .+ p[3]
# ---------------------------------------------------------------------------- #
function findthr(x::Vector{Float64}, c::Real, op::Function=Base.:>=)
   mn, mx = extrema(x)
   return findfirst(a -> op(a, mn + c * (mx-mn)), x)
end
# ---------------------------------------------------------------------------- #
function optimal_span(span::AbstractVector{Float64}, y::Vector{Float64}, fdecay::Bool=true)

    if any(!isfinite, y)
        ku = findall(isfinite , y)
        return optimal_span_impl(span[ku], y[ku], fdecay)
    end
    return optimal_span_impl(span, y, fdecay)
end
# ---------------------------------------------------------------------------- #
function optimal_span_impl(span::AbstractVector{Float64}, y::Vector{Float64}, fdecay::Bool=true)

    if isempty(y)
        @warn("Input to optimal_span is empty!")
        return NaN
    end

    # if y has a "well defined" minimum that is not an endpoint, just return that
    _, kmn = findmin(y)
    if kmn > 2 && kmn < (length(y) - 2)
        return span[kmn]
    end

    mn, mx = extrema(y)
    if fdecay
        kt = findthr(y, 1.0 / exp(1), <=)
        if kt == nothing
            @show(span, y)
            @warn("Failed to estimate initial value for tau")
            return NaN
        end
        tau = 1.0 / span[kt]
        p0 = [mx-mn, tau, mn]
        op = <=
        model = decay
        c = 0.01
    else
        kc = findthr(y, 0.5, >=)
        if kc == nothing
            @show(span, y)
            @warn("Failed to estimate initial value for c50")
            return NaN
        end
        c50 = span[kc]
        p0 = [mx, 0.05, c50, mn]
        op = >=
        model = logistic
        c = 0.99
    end

    obj(p::Vector) = sum(abs2, y .- model(span, p))
    opt = optimize(obj, p0; autodiff=:forward)

    xx = range(Float64(span[1]), Float64(span[end]), length=128)
    ks = findthr(model(xx, opt.minimizer), c, op)
    if ks == nothing
        @show(opt.minimizer, fdecay)
        @warn("Failed to estimate asymptotic value")
        return NaN
    end
    return xx[ks]
end
# ============================================================================ #
function temporal_span(ret::Vector{Float64}, lgn::Vector{Float64}, id::Integer, bin_size::Real=0.001, io::IO=stdout)

    response = wasrelayed(ret, lgn)

    ps = PredictorSet()
    ps[:ff] = Predictor(ret, ret, CosineBasis(length=60, offset=2, nbasis=20, b=10, ortho=false, bin_size=bin_size))
    glm = GLM(ps, response)

    spans = round.(Int, 10.0 .^ range(1, 2.85, length=NSPAN))
    lis = zeros(length(spans))
    roca = zeros(length(spans))
    jsd = zeros(length(spans))

    li = +Inf
    kmin = 1

    result = RelayGLM.GLMResult(zeros(ncol(ps)), 0.0, 0.0, +Inf, false)

    for k in 1:length(spans)
        t1 = time()
        set_parameter!(get_basis(glm, :ff), length=spans[k])

        res = cross_validate(Binomial, Logistic, glm, nfold=10, shuffle_design=true)

        if !res.converged
            println(io, "[WARN]: Failure to converge for pair $(id), span = $(spans[k])")
        end

        lis[k] = res.nlli
        roca[k] = res.roca
        jsd[k] = res.accuracy

        if res.nlli < li
            kmin = k
            li = res.nlli
            result = res
        end
        # show_progress(k/length(spans), 1, "Span: ", "($(k) of $(length(spans)))")
    end

    return result, lis, roca, jsd, Vector{Float64}(spans)
end
# ============================================================================ #
function temporal_span_fb(ret::Vector{Float64}, lgn::Vector{Float64}, id::Integer, ff_span::Integer, bin_size::Real=0.001, io::IO=stdout)

    response = wasrelayed(ret, lgn)

    # ff_span = max(60, ff_span)

    ps = PredictorSet()
    ps[:ff] = Predictor(ret, ret, CosineBasis(length=Int(ff_span), offset=2, nbasis=20, b=10, ortho=false, bin_size=bin_size))
    ps[:fb] = Predictor(lgn, ret, CosineBasis(length=100, offset=2, nbasis=20, b=6, ortho=false, bin_size=bin_size))
    glm = GLM(ps, response)

    spans = round.(Int, 10.0 .^ range(1.8, 2.9, length=NSPAN))
    lis = zeros(length(spans))
    roca = zeros(length(spans))
    jsd = zeros(length(spans))

    li = +Inf
    kmin = 1

    result = RelayGLM.GLMResult(zeros(ncol(ps)), 0.0, 0.0, +Inf, false)

    for k in 1:length(spans)
        set_parameter!(get_basis(glm, :fb), length=spans[k])

        res = cross_validate(Binomial, Logistic, glm, nfold=10, shuffle_design=true)

        if !res.converged
            println(io, "[WARN]: Failure to converge for pair $(id), span = $(spans[k])")
        end

        lis[k] = res.nlli
        roca[k] = res.roca
        jsd[k] = res.accuracy

        if res.nlli < li
            kmin = k
            li = res.nlli
            result = res
        end
        show_progress(k/length(spans), 1, "Span: ", "($(k) of $(length(spans)))")
    end

    return result, lis, roca, jsd, Vector{Float64}(spans)
end
# ============================================================================ #
function nbasis_fb(ret::Vector{Float64}, lgn::Vector{Float64}, id::Integer, ff_span::Integer, fb_span::Integer, bin_size::Real=0.001, io::IO=stdout)

    response = wasrelayed(ret, lgn)

    # ff_span = max(60, ff_span)
    # fb_span = max(60, fb_span)

    ps = PredictorSet()
    ps[:ff] = Predictor(ret, ret, CosineBasis(length=Int(ff_span), offset=2, nbasis=20, b=10, ortho=false, bin_size=bin_size))
    ps[:fb] = Predictor(lgn, ret, CosineBasis(length=Int(fb_span), offset=2, nbasis=20, b=6, ortho=false, bin_size=bin_size))
    glm = GLM(ps, response)

    nbasis = 4:4:40 #round.(Int, range(8, 56, length=NBASIS))
    lis = zeros(length(nbasis))
    roca = zeros(length(nbasis))
    jsd = zeros(length(nbasis))

    li = +Inf
    kmin = 1

    result = RelayGLM.GLMResult(zeros(ncol(ps)), 0.0, 0.0, +Inf, false)

    for k in 1:length(nbasis)
        set_parameter!(get_basis(glm, :fb), nbasis=nbasis[k])

        res = cross_validate(Binomial, Logistic, glm, nfold=10, shuffle_design=true)

        if !res.converged
            println(io, "[WARN]: Failure to converge for pair $(id), nbasis = $(nbasis[k])")
        end

        lis[k] = res.nlli
        roca[k] = res.roca
        jsd[k] = res.accuracy

        if res.nlli < li
            kmin = k
            li = res.nlli
            result = res
        end
        show_progress(k/length(nbasis), 1, "N-basis: ", "($(k) of $(length(nbasis)))")
    end

    return result, lis, roca, jsd, Vector{Float64}(nbasis)
end
# ============================================================================ #
@enum ParamType FFSpan FBSpan FBBasis
function collate_data(typ::ParamType, bin_size::Real=0.001)

    suffix = typ == FBBasis ? "_fbbasis-hp.log" : (typ == FBSpan ? "_fb-hp.log" : "_hp.log")
    logfile = Dates.format(Dates.now(), "YYYYmmdd-HHMM") * suffix
    io = open(logfile, "w")

    len = typ == FBBasis ? NBASIS : NSPAN

    param = load_parameters()

    d = Dict{String, Any}()
    tmp = Dict{String,String}("grating" => "(?:contrast|area|grating)", "msequence"=>"msequence")

    for (type, ptrn) in tmp

        db = get_database(ptrn, id -> !in(id, EXCLUDE[type]))
        d[type] = Dict{String, Any}()
        d[type]["ids"] = first.(db)
        d[type]["roca"] = zeros(len, length(db))
        d[type]["jsd"] = zeros(len, length(db))
        d[type]["nlli"] = zeros(len, length(db))

        if typ == FBBasis
            d[type]["aic"] = zeros(len, length(db))
        end

        d[type]["span"] = zeros(len)
        d[type]["roca_span"] = zeros(length(db))
        d[type]["nlli_span"] = zeros(length(db))
        d[type]["jsd_span"] = zeros(length(db))

        if typ == FBBasis
            d[type]["aic_span"] = zeros(length(db))
        end

        d[type]["converged"] = fill(true, length(db))
        have_span = false

        for k in 1:length(db)
            show_progress((k-1)/length(db), 0, "$(type): ", "($(k) of $(length(db)))\n")

            println(io, "*"^60)
            println(io, "[INFO]: Fitting pair $(first(db[k])) [$(type)]")

            ret, lgn, _, _ = get_data(db, k)
            id = first(db[k])

            if typ == FBBasis
                result, li, roca, jsd, spans = nbasis_fb(ret, lgn, id, param[id][type]["ff_temporal_span"], param[id][type]["fb_temporal_span"], bin_size, io)
            elseif typ == FBSpan
                result, li, roca, jsd, spans = temporal_span_fb(ret, lgn, id, param[id][type]["ff_temporal_span"], bin_size, io)
            else
                result, li, roca, jsd, spans = temporal_span(ret, lgn, id, bin_size, io)
            end

            if !result.converged
                d[type]["converged"][k] = false
                println(io, "[WARN]: Pair $(first(db[k])) [$(type)] failed to converge")
            end

            if !have_span
                d[type]["span"] .= spans
                have_span = true
            end

            d[type]["roca"][:,k] = roca
            d[type]["nlli"][:,k] = li
            d[type]["jsd"][:,k] = jsd

            if typ == FBBasis
                # AIC(x) = -2*ln(L(x)) + 2k where k = number of parameters and L(x) is
                # the likelihood of the model (parameters) <x>, in the below li
                # is the negative log likelihood, thus we multiply by -1 to get:
                d[type]["aic"][:,k] = 2.0 .* li .+ 2.0 .* (spans .+ 20) # + 20 for the FF terms
            end

            d[type]["jsd_span"][k] = optimal_span(spans, jsd, false)
            d[type]["roca_span"][k] = optimal_span(spans, roca, false)
            d[type]["nlli_span"][k] = optimal_span(spans, li, true)

            if typ == FBBasis
                _, kmn = findmin(d[type]["aic"][:,k])
                if kmn != nothing
                    d[type]["aic_span"][k] = spans[kmn]
                end
            end

            clear_lines_above(1)
        end
    end
    close(io)
    return d
end
# ============================================================================ #
function parameter_dict(d::Dict, key::String, metric::String="jsd")
    out = Dict{Int,Any}()
    return parameter_dict!(out, d, key, metric)
end
# ============================================================================ #
function parameter_dict!(out::Dict{Int,<:Any}, d::Dict, key::String, metric::String="jsd")
    ids = union(d["grating"]["ids"], d["msequence"]["ids"])
    spans = d["grating"]["span"]
    for id in ids
        if !haskey(out, id)
            out[id] = Dict{String,Any}()
        end
        for typ in ["grating","msequence"]
            k = findfirst(isequal(id), d[typ]["ids"])
            k == nothing && continue
            if !haskey(out[id], typ)
                out[id][typ] = Dict{String,Any}()
            end
            # t = optimal_span(spans, d[typ][metric][:,k], false)
            t = d[typ][metric * "_span"][k]
            out[id][typ][key] = round(Int, t)
        end
    end
    return out
end
# ============================================================================ #
function write_parameters(d::Dict, key::String; metric::String="jsd", ofile::String=DBFILE)
    out = parameter_dict(d, key, metric)
    save_parameters(out, ofile)
end
# ============================================================================ #
function save_parameters(d::Dict{Int,<:Any}, ofile::String=DBFILE)
    tmp = Dict{String,Dict{String,Any}}((string(k) => v for (k,v) in d))
    open(ofile, "w") do io
        JSON.print(io, tmp, 4)
    end
end
# ============================================================================ #
function load_parameters(ifile::String=DBFILE)
    d = open(JSON.parse, ifile, "r")
    return Dict((parse(Int, k) => v for (k,v) in d))
end
# ============================================================================ #
end
