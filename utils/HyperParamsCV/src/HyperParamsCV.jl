module HyperParamsCV

using RelayGLM, DatabaseWrapper, PaperUtils, Progress

import RelayGLM.RelayISI
import RelayGLM.GLMMetrics
import RelayGLM.GLMFit
import RelayGLM.GLMFit.GLMModels
import RelayGLM.GLMFit.Partitions

using LinearAlgebra, Statistics
import JSON

const Strmbol = Union{String,Symbol}
const DBFILE = joinpath(@__DIR__, "hyper_parameters.json")

# ============================================================================ #
function train_isi_model(isi::AbstractVector{<:Real}, status::AbstractVector{<:Real},
    idx::AbstractVector{<:Integer})

    sigma = roundn.(vcat(0.0, 10 .^ range(log10(0.002), log10(0.03), length=7)), -3)
    isimax = roundn.(10.0 .^ range(log10(0.03), log10(0.5), length=8), -3)

    best_sigma = -1.0
    best_isimax = -1.0
    rri = -Inf

    # find the best set of hyperparameters (only using training data @ <idx>)
    for row in eachrow([repeat(sigma, inner=length(isimax)) repeat(isimax, outer=length(sigma))])

        res = RelayISI.isi_model(RRI, isi[idx], status[idx], row[1], 0.001, row[2], 10, true)
        rri_k = mean(res)

        if rri_k > rri
            best_sigma = row[1]
            best_isimax = row[2]
            rri = rri_k
        end
    end

    # build the model using the best hyperparams (note, we are still *ONLY*
    # using the training data specified by <idx>)
    edges, eff = RelayISI.get_eff(isi, status, idx, best_sigma, 0.001, best_isimax)
    eff = RelayISI.scale_ef(edges, eff, isi[idx], status[idx], 0.001)

    return best_sigma, best_isimax, rri, edges, eff
end
# ============================================================================ #
function test_isi_model(edges::AbstractVector{<:Real}, eff::Vector{Float64},
    isi_test::Vector{Float64})

    return RelayISI.logistic!(RelayISI.predict(edges, eff, isi_test))
end
# ============================================================================ #
function glm_bin_size(glm::RelayGLM.RegGLM)
    bs = map(x -> getfield(get_basis(x), :bin_size), RelayGLM.predictors(glm))
    @assert(all(x -> x==bs[1], bs), "Bin sizes *MUST* be consistent across predictors")
    return bs[1]
end
# ============================================================================ #
function glm_data(glm::RelayGLM.RegGLM, idx::AbstractVector{<:Integer})
    bs = glm_bin_size(glm)

    # NOTE: we *ONLY* use the rows of the predictor maxtrix, and the entries of
    # the response vector, indicated by the index vector <idx>
    return GLMFit.GLM(Binomial, Logistic, RelayGLM.generate(RelayGLM.predictors(glm))[idx,:], RelayGLM.response(glm)[idx], bs)
end
# ---------------------------------------------------------------------------- #
function glm_data(glm::RelayGLM.RegGLM, pr::AbstractMatrix, idx::AbstractVector{<:Integer})
    bs = glm_bin_size(glm)
    # see NOTE in glm_data() above...
    return GLMFit.RegularizedGLM(Binomial, Logistic, RelayGLM.generate(RelayGLM.predictors(glm))[idx,:], RelayGLM.response(glm)[idx], bs, pr)
end
# ============================================================================ #
function init_ff_model(ret::AbstractVector{<:Real}, lgn::AbstractVector{<:Real},
    status::AbstractVector{<:Real}, span_ff::Integer, lm::AbstractVector{<:Real})

    nb_ff = 16
    b_ff = 10

    # status of first spike cannot be predicted...
    if length(status) == (length(ret)-1)
        evt = ret[2:end]
    else
        evt = ret
    end

    train = PredictorSet()
    train[:ff] = Predictor(ret, evt, DefaultBasis(length=span_ff, offset=2, bin_size=0.001))
    return GLM(train, status, SmoothingPrior, [lm])
end
# ============================================================================ #
function train_ff_model(ret::AbstractVector{<:Real}, lgn::AbstractVector{<:Real},
    status::AbstractVector{<:Real}, idx::AbstractVector{<:Integer})

    nbasis = 16

    # lengths = round.(Int, 10.0 .^ range(1, 2.505, length=6)) #[60, 100]
    lengths = round.(Int, 10.0 .^ range(log10(30), log10(500), length=8))
    lm = 2.0 .^ range(2, 12, length=5)

    # --- initialize the model --- #
    # I believe that this *HAS* to use all data to maintain temporal continuity
    glm = init_ff_model(ret, lgn, status, lengths[1], lm)

    rri = -Inf
    best_length = -1
    best_lm = Vector{Vector{Float64}}(undef, 1)
    coef = Vector{Vector{Float64}}(undef, 1)

    show_progress(0.0, 3, "FF model training: ", "(0 of $(length(lengths)))")

    for k in eachindex(lengths)
        t1 = time()

        # set the span for this iteration
        set_parameter!(get_basis(glm, :ff), length=lengths[k])

        prior = RelayGLM.generate_prior(glm)
        pr = GLMFit.GLMPriors.allocate_prior(prior)

        # NOW we extract only the training data to train the model
        train = glm_data(glm, pr, idx)

        # initial guess is STA
        x0 = GLMFit.predictors(train)' * GLMFit.response(train) ./ sum(GLMFit.response(train))

        # NOTE: the prior has *NO* dependency on the number of rows of the
        # predictor matrix, only the number of columns, so we are not violating
        # our promise to only us the data indicated by <idx> in the following:
        res = GLMFit.cross_validate(RRI, train, prior, x0, 10, true)

        rri_k = mean(res.metric)
        if rri_k > rri
            rri = rri_k
            coef[1] = GLMFit.coefficients(res)
            best_length = lengths[k]
            best_lm[1] = GLMFit.lambda(res)
        end
        show_progress(k/length(lengths), 3, "FF model training: ", "($(k) of $(length(lengths))) @ $(time() - t1)")
    end

    return best_length, best_lm[1], rri, coef[1]
end
# ============================================================================ #
function init_fr_model(ret::AbstractVector{<:Real}, lgn::AbstractVector{<:Real},
    status::AbstractVector{<:Real}, span_ff::Integer, span_fr::Integer, nb_fr::Integer, lm::AbstractVector{<:Real})

    nb_ff = 16
    b_ff = 10
    b_fr = 8

    # status of first spike cannot be predicted...
    if length(status) == (length(ret)-1)
        evt = ret[2:end]
    else
        evt = ret
    end

    ps = PredictorSet()
    ps[:ff] = Predictor(ret, evt, CosineBasis(length=span_ff, offset=2, nbasis=nb_ff, b=b_ff, ortho=false, bin_size=0.001))
    ps[:fr] = Predictor(lgn, evt, CosineBasis(length=span_fr, offset=2, nbasis=nb_fr, b=b_fr, ortho=false, bin_size=0.001))

    return GLM(ps, status, RidgePrior, [lm])
end
# ============================================================================ #
function train_fr_model(ret::AbstractVector{<:Real}, lgn::AbstractVector{<:Real},
    status::AbstractVector{<:Real}, idx::AbstractVector{<:Integer}, span_ff::Integer)

    # lengths = round.(Int, 10.0 .^ range(1.5, 2.7, length=6)) # [80, 120]
    # nbasis_fr = [6, 8, 12, 20] # [6, 12]
    # lm = 2.0 .^ range(-3, 5, length=5) # [0.1, 1.0]

    lengths = round.(Int, 10.0 .^ range(log10(40), log10(600), length=8))
    nbasis_fr = [8, 12, 18, 24, 32]
    lm = 2.0 .^ range(-3.5, 3, length=5)

    # --- initialize the model --- #
    # I believe that this *HAS* to use all data to maintain temporal
    # continuity with the change-of-basis
    glm = init_fr_model(ret, lgn, status, span_ff, lengths[1], nbasis_fr[1], lm)

    rri = -Inf
    best_length = -1
    best_nb = -1
    best_lm = Vector{Vector{Float64}}(undef, 1)
    coef = Vector{Vector{Float64}}(undef, 1)

    N = length(lengths) * length(nbasis_fr)
    k = 1
    show_progress(0.0, 3, "FR model training: ", "(0 of $(N))")

    for row in eachrow([repeat(lengths, inner=length(nbasis_fr)) repeat(nbasis_fr, outer=length(lengths))])
        t1 = time()

        # set the parameters for this iteration
        set_parameter!(get_basis(glm, :fr), length=row[1])
        set_parameter!(get_basis(glm, :fr), nbasis=row[2])

        prior = RelayGLM.generate_prior(glm)
        pr = GLMFit.GLMPriors.allocate_prior(prior)

        # NOW we extract only the training data to train the model
        train = glm_data(glm, pr, idx)

        # initial guess is STA
        x0 = GLMFit.predictors(train)' * GLMFit.response(train) ./ sum(GLMFit.response(train))

        # NOTE: see note in train_ff_model() above...
        res = GLMFit.cross_validate(RRI, train, prior, x0, 10, true)

        rri_k = mean(res.metric)
        if rri_k > rri
            rri = rri_k
            coef[1] = GLMFit.coefficients(res)
            best_length = row[1]
            best_nb = row[2]
            best_lm[1] = GLMFit.lambda(res)
        end

        show_progress(k/N, 3, "FR model training: ", "($(k) of $(N)) @ $(time() - t1)")
        k += 1
    end

    return best_length, best_nb, best_lm[1], rri, coef[1]
end
# ============================================================================ #
function test_glm_model(::Type{A}, test::GLMFit.GLM, coef::Vector{Float64}) where {A<:RelayGLM.GLMModels.ActivationFunction}

    mul!(test.r, test.x, coef)
    pred, _ = GLMModels.activate!(A, test.r, test.dr, test.bin_size)
    return pred
end
# ============================================================================ #
function nested_cv(ret::AbstractVector{<:Real}, lgn::AbstractVector{<:Real}, id::Integer, type::String, nfold::Integer=10, shlf::Bool=true, io::IO=devnull)

    isi, relay_status = RelayISI.spike_status(ret, lgn)

    res_isi = RRI(nfold, length(ret))
    res_ff = RRI(nfold, length(ret))
    res_fr = RRI(nfold, length(ret))

    sigmas = zeros(nfold)
    isimaxes = zeros(nfold)
    spans_ff = zeros(nfold)
    lms_ff = zeros(nfold)
    spans_fr = zeros(nfold)
    nbs_fr = zeros(nfold)
    lms_fr = zeros(nfold)

    k = 1

    show_progress(0.0, 1, "Cross-val folds [$(id)]: ", "(0 of $(nfold))\n")

    for p in Partitions.ballanced_partition(Partitions.IndexPartitioner, isi, relay_status, nfold, shlf)

        t1 = time()

        idxtrain = Partitions.training_set(p)
        idxtest = Partitions.testing_set(p)

        ef_tr = sum(relay_status[idxtrain]) / length(idxtrain)
        ef_te = sum(relay_status[idxtest]) / length(idxtest)

        # isi model:
        sigma, isimax, rii_isi, edges, eff = train_isi_model(isi, relay_status, idxtrain)
        pred_isi = test_isi_model(edges, eff, isi[idxtest])
        GLMMetrics.eval_and_store!(res_isi, relay_status[idxtest], pred_isi, k, idxtest)

        sigmas[k] = sigma
        isimaxes[k] = isimax

        print(io, "Partition ", k, ":\n\t")
        print(io, "eff = (", ef_tr, ", ", ef_te, ")\n\tsigma = ", sigma, "\n\tlength = ", isimax, "\n\t")

        show_progress(1/3, 2, "ISI model ", "(1 of 3) @ $(time() - t1)\n")

        # ff model: train and return the best set of parameters
        span_ff, lm_ff, rii_ff, coef_ff = train_ff_model(ret, lgn, relay_status, idxtrain)

        print(io, "span_ff = ", span_ff, "\n\tlm_ff = ", lm_ff, "\n\t")

        # test data for ff model
        glm_ff = init_ff_model(ret, lgn, relay_status, span_ff, lm_ff)
        test_ff = glm_data(glm_ff, idxtest)

        # test!
        pred_ff = test_glm_model(Logistic, test_ff, coef_ff)
        GLMMetrics.eval_and_store!(res_ff, relay_status[idxtest], pred_ff, k, idxtest)

        spans_ff[k] = span_ff
        lms_ff[k] = lm_ff[1]

        clear_lines_above(1)
        show_progress(2/3, 2, "FF model ", "(2 of 3) @ $(time() - t1)\n")

        # fr model
        span_fr, nb_fr, lm_fr, rii_fr, coef_fr = train_fr_model(ret, lgn, relay_status, idxtrain, span_ff)

        print(io, "span_fr = ", span_fr, "\n\tnb_fr = ", nb_fr, "\n\tlm_fr = ", lm_fr, "\n\n")

        # test data for fr model
        glm_fr = init_fr_model(ret, lgn, relay_status, span_ff, span_fr, nb_fr, lm_fr)
        test_fr = glm_data(glm_fr, idxtest)

        pred_fr = test_glm_model(Logistic, test_fr, coef_fr)
        GLMMetrics.eval_and_store!(res_fr, relay_status[idxtest], pred_fr, k, idxtest)

        spans_fr[k] = span_fr
        nbs_fr[k] = nb_fr
        lms_fr[k] = lm_fr[1]

        clear_lines_above(1)
        show_progress(3/3, 2, "FR model ", "(3 of 3) @ $(time() - t1)\n")

        elap = time() - t1

        clear_lines_above(2)
        # clear_lines_above(1)
        show_progress(k/nfold, 1, "Cross-val folds [$(id)]: ", "($(k) of $(nfold)) @ $(elap))\n")

        k += 1
    end

    return res_isi, res_ff, res_fr, sigmas, isimaxes, spans_ff, lms_ff, spans_fr, nbs_fr, lms_fr
end
# ============================================================================ #
function get_usrey_ids(grp::Integer, ngroups::Integer=3)
    grp > 4 && error("Invalid group: must be in [1,2,3]")

    tmp = Dict{String,String}("grating" => "(?:contrast|area|grating)", "msequence"=>"msequence")
    all_ids = Int[]
    for (type, ptrn) in tmp
        db = get_database(ptrn, id -> !in(id, PaperUtils.EXCLUDE[type]))
        append!(all_ids, get_ids(db))
    end
    ids = sort(unique(all_ids))

    if ngroups == 3
        ks = (grp-1)*15+1
        ke = ks + 14
        ke = grp == 3 ? length(ids) : ke
    elseif ngroups == 4
        ks = (grp-1)*11+1
        ke = ks + 10
        ke = grp == 4 ? length(ids) : ke
    else
        error("Invalid number of groups $(ngroups)")
    end

    return ids[ks:ke]
end
# ============================================================================ #
get_weyand_ids(::Integer) = get_ids(get_database(:weyand))
# ============================================================================ #
function collate_data(iids::AbstractVector{<:Integer}=Int[], logdir::AbstractString="")

    d = Dict{String, Any}()
    tmp = Dict{String,Strmbol}("grating" => "(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)

    nfold = 10

    for (type, ptrn) in tmp

        db = get_database(ptrn, id -> !in(id, PaperUtils.EXCLUDE[type]))
        d[type] = Dict{Int,Dict}()

        dbids = get_ids(db)
        if isempty(iids)
            ids = dbids
        else
            ids = filter(in(dbids), iids)
        end

        foreach(ids) do id
            d[type][id] = Dict{String,Dict{String,Vector{Float64}}}(
                "isi" => Dict{String,Vector{Float64}}(),
                "ff" => Dict{String,Vector{Float64}}(),
                "fr" => Dict{String,Vector{Float64}}()
            )
        end

        N = length(ids)

        show_progress(0.0, 0, "$(type): ", "(0 of $(N))\n")

        for k in 1:N
            t1 = time()

            id = ids[k]

            ret, lgn, _, _ = get_data(db, id=id)

            io = isdir(logdir) ? open(joinpath(logdir, "$(id)_$(type)-hp-cv.log"), "w") : devnull

            res_isi, res_ff, res_fr, sigma, isimaxes, span_ff, lm_ff, span_fr, nb_fr, lm_fr = nested_cv(ret, lgn, id, type, nfold, true, io)

            close(io)

            d[type][id]["isi"]["sigma"] = Vector{Float64}(sigma)
            d[type][id]["isi"]["isimax"] = Vector{Float64}(isimaxes)
            d[type][id]["isi"]["rri"] = res_isi.x
            d[type][id]["isi"]["prediction"] = res_isi.pred

            d[type][id]["ff"]["span"] = Vector{Float64}(span_ff)
            d[type][id]["ff"]["lm"] = lm_ff
            d[type][id]["ff"]["rri"] = res_ff.x
            d[type][id]["ff"]["prediction"] = res_ff.pred

            d[type][id]["fr"]["span"] = Vector{Float64}(span_fr)
            d[type][id]["fr"]["nbasis"] = Vector{Float64}(nb_fr)
            d[type][id]["fr"]["lm"] = lm_fr
            d[type][id]["fr"]["rri"] = res_fr.x
            d[type][id]["fr"]["prediction"] = res_fr.pred

            elap = time() - t1
            clear_lines_above(4)
            # clear_lines_above(1)
            show_progress(k/N, 0, "$(type): ", "($(k) of $(N) @ $(elap))\n")
        end
    end

    return d
end
# ============================================================================ #
function merge_dicts(data::Vector{<:Dict}, fields::Vector{String}=["isi","ff","fr"]; force::Bool=false)
    return merge_dicts!(Dict{String,Any}(), data, fields; force=force)
end
# ============================================================================ #
function merge_dicts!(out::Dict{String,<:Any}, data::Vector{<:Dict}, fields::Vector{String}=["isi","ff","fr"]; force::Bool=false)
    for typ in sort(collect(keys(data[1])))
        ids = Vector{Int}(undef, 0)
        for d in data
            append!(ids, keys(d[typ]))
        end
        if force || !haskey(out, typ)
            out[typ] = Dict{String,Any}()
        end
        if force || !haskey(out[typ], "ids")
            out[typ]["ids"] = sort(ids)
        end
        if force || !haskey(out[typ], "rri")
            out[typ]["rri"] = Dict{String,Any}()
        end

        for field in fields
            if force || !haskey(out[typ]["rri"], field)
                out[typ]["rri"][field] = zeros(length(ids))
            end
        end

        for d in data
            for id in keys(d[typ])
                k = findfirst(isequal(id), out[typ]["ids"])
                k == nothing && error("Failded to locate id $(id) in type $(typ)")
                for field in fields
                    out[typ]["rri"][field][k] = mean(d[typ][id][field]["rri"])
                end
            end
        end
    end
    return out
end
# ============================================================================ #
function merge_all(data::Vector{<:Dict})

    out = Dict{String,Any}()

    for d in data
        for typ in keys(d)
            if !haskey(out, typ)
                out[typ] = Dict{Int,Any}()
            end
            for id in keys(d[typ])
                out[typ][id] = d[typ][id]
            end
        end
    end

    return out
end
# ============================================================================ #
function parameter_dict(data::Vector{Dict{String,Any}})
    out = Dict{Int,Any}()
    return parameter_dict!(out, data)
end
# ============================================================================ #
function parameter_dict!(out::Dict{Int,Any}, data::Vector{Dict{String,Any}})
    all_ids = Vector{Int}(undef, 0)
    for d in data
        for k in keys(d)
            append!(all_ids, keys(d[k]))
        end
    end
    ids = sort(unique(all_ids))

    for d in data
        for typ in keys(d)
            for id in keys(d[typ])
                if !haskey(out, id)
                    out[id] = Dict{String,Any}()
                end
                if !haskey(out[id], typ)
                    out[id][typ] = Dict{String,Any}()
                end

                out[id][typ]["ff_temporal_span"] = median(d[typ][id]["ff"]["span"])
                out[id][typ]["fb_temporal_span"] = median(d[typ][id]["fr"]["span"])
                out[id][typ]["fb_nbasis"] = median(d[typ][id]["fr"]["nbasis"])
            end
        end
    end
    return out
end
# ============================================================================ #
function write_parameters(data::Vector{Dict{String,Any}}, ofile::String=DBFILE)
    save_parameters(parameter_dict(data), ofile)
end
# ============================================================================ #
function save_parameters(d::Dict{Int,Any}, ofile::String=DBFILE)
    tmp = Dict{String,Dict{String,Any}}((string(k) => v for (k,v) in d))
    open(ofile, "w") do io
        JSON.print(io, tmp, 4)
    end
    return ofile
end
# ============================================================================ #
function load_parameters(ifile::String=DBFILE)
    d = open(JSON.parse, ifile, "r")
    return Dict((parse(Int, k) => v for (k,v) in d))
end
# ============================================================================ #
end
