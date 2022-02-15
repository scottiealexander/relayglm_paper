module FigureS5

using JLD, PyPlot, Statistics, Random
using MSequence, MSequenceUtils, PaperUtils, PairsDB, Figure1

function collate_data(data::Dict{String,Any}=Dict{String,Any}(), ids::Vector{Int}=Int[])

    if isempty(data)
        data = load("../20211216_hp_cv_pred_all.jld")
    end

    db = get_database("msequence", id -> !in(id, PaperUtils.EXCLUDE["msequence"]))

    if isempty(ids)
        ids = filter(>(99), sort(collect(keys(data["msequence"]))))
    end

    d = Dict{Union{String,Int},Any}()
    d["isi"] = zeros(2, length(ids))
    d["ff"] = zeros(2, length(ids))
    d["fr"] = zeros(2, length(ids))

    for k in eachindex(ids)

        ret, lgn, evt, lab = get_data(db, id=ids[k])

        fpt = get_uniform_param(db[id=ids[k]], "frames_per_term")
        fps = get_uniform_param(db[id=ids[k]], "refresh_rate")

        ifi = fpt / fps

        ret_rf = sta(ret, evt, lab, ifi)

        ktrg = PaperUtils.contribution(ret, lgn)
        lgn_rf = sta(lgn[ktrg], evt, lab, ifi)

        rk = peakframe(ret_rf)
        lk = peakframe(lgn_rf)

        if !(0 < rk < 16) || !(0 < lk < 16)
            @warn("ID $(ids[k]): rk = $(rk), lk = $(lk)")
            # return ret_rf, lgn_rf
            rk = 4
            lk = 4
        end

        ret_rf = Figure1.mean_squeeze(ret_rf[:,:,rk-1:rk+1])
        lgn_rf = Figure1.mean_squeeze(lgn_rf[:,:,lk-1:lk+1])

        d[ids[k]] = Dict{String,Matrix{Float64}}()
        d[ids[k]]["ret_rf"] = ret_rf
        d[ids[k]]["lgn_rf"] = lgn_rf

        for model in ["isi", "ff", "fr"]
            pred = data["msequence"][ids[k]][model]["prediction"][1:end-1]
            # pred = fill(mean(data["msequence"][id][model]["prediction"][1:end-1]), length(ret)-1)

            tmp = sta(ret[2:end] .+ 0.0028, pred, evt, lab, ifi)
            pred_rf = Figure1.mean_squeeze(tmp[:,:,lk-1:lk+1])

            d[model][:,k] .= [cor(vec(pred_rf), vec(lgn_rf)), mean(foo(ret[2:end] .+ 0.0028, pred, evt, lab, ifi, lgn_rf, lk, 100))]
            # d[model][:,k] .= [cor(vec(pred_rf), vec(lgn_rf)), cor(vec(ret_rf), vec(lgn_rf))]



            d[ids[k]][model * "_rf"] = pred_rf

        end
    end

    d["ids"] = ids

    return d
end

function foo(ret, pred, evt, lab, ifi, lgn_rf, lk, niter=1000)

    tmp = copy(pred)
    out = zeros(niter)
    tmp_rf = zeros(16,16,16)
    for k in 1:niter
        shuffle!(tmp)
        tmp_rf .= sta(ret, tmp, evt, lab, ifi)
        pred_rf = Figure1.mean_squeeze(tmp_rf[:,:,lk-1:lk+1])
        out[k] = cor(vec(pred_rf), vec(lgn_rf))
    end
    return out
end

function make_figure(d::Dict{String,<:Any}, model::String="ff")

    h, ax = subplots(1,3)
    # foreach(default_axes, ax)
    h.set_size_inches((12,5))

    cmap = matplotlib.colors.ListedColormap(clay_color())
    smth = [0,0]

    im1 = Figure1.show_rf(d["ret_rf"], ax[1], cmap, smth)
    im2 = Figure1.show_rf(d["lgn_rf"], ax[2], cmap, smth)
    im3 = Figure1.show_rf(d[model * "_rf"], ax[3], cmap, smth)

    foreach((im1,im2,im3)) do x
        cl = x.get_clim()
        mx = maximum(abs.(cl))
        x.set_clim(-mx, mx)
    end

    for cax in ax
        for spine in cax."spines".values()
            spine.set_visible(false)
        end
        cax.set_xticks([])
        cax.set_yticks([])
    end

    ax[1].set_title("Retina", fontsize=18)
    ax[2].set_title("LGN", fontsize=18)
    ax[3].set_title("Predicted LGN", fontsize=18)


    h.tight_layout()

    return h, ax
end

end
