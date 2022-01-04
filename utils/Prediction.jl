module Prediction

using Statistics, PyPlot, JLD

using RelayGLM, DatabaseWrapper, PaperUtils, Plot
import RelayGLM.RelayISI
import RelayGLM.RelayUtils

using SimpleStats, UCDColors

const Strmbol = Union{String,Symbol}

# Prediction
# d = Prediction.collate_data();
# Prediction.make_figure(d);

# ============================================================================ #
function collate_data(data::Dict{String,Any}=Dict{String,Any}())

    if isempty(data)
        data = load("../20211216_hp_cv_pred_all.jld")
    end

    tmp = Dict{String,Strmbol}("grating"=>"(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)

    pred_bins = range(0, 1, length=11)
    isi_bins = 0.002:0.002:0.124
    # isi_bins2 = range(0.002, 0.06, length=10)

    out = Dict{String,Any}()

    for (typ, ptrn) in tmp

        db = get_database(ptrn, id -> !in(id, PaperUtils.EXCLUDE[typ]))

        out[typ] = Dict{String,Any}()

        npair = length(db)
        kpair = 1

        for model in ["isi","ff","fr"]
            out[typ][model] = Dict{String,Matrix{Float64}}()
            out[typ][model]["pred_efficacy"] = fill(NaN, length(pred_bins)-1, npair)
            out[typ][model]["obs_isi_efficacy"] = fill(NaN, length(isi_bins)-1, npair)
            out[typ][model]["pred_isi_efficacy"] = fill(NaN, length(isi_bins)-1, npair)
            # out[typ][model]["rri_by_isi"] = fill(NaN, length(isi_bins2)-1, npair)
        end

        ids = sort(collect(keys(data[typ])))

        out[typ]["ids"] = ids

        for id in ids

            ret, lgn, _, _ = get_data(db, id=id)

            isi, status = RelayISI.spike_status(ret, lgn)

            mean_eff = sum(status) / length(status)

            for model in keys(data[typ][id])
                pred = data[typ][id][model]["prediction"]
                kuse = findall(!isnan, pred)

                ef2, ef3 = calc_isi_efficacy(pred[kuse], isi[kuse], status[kuse], isi_bins)

                out[typ][model]["obs_isi_efficacy"][:,kpair] .= ef2 ./ mean_eff
                out[typ][model]["pred_isi_efficacy"][:,kpair] .= ef3 ./ mean(pred[kuse])

                out[typ][model]["pred_efficacy"][:,kpair] .= calc_pred_efficacy(pred[kuse], isi[kuse], status[kuse], pred_bins)
                # out[typ][model]["rri_by_isi"][:,kpair] .= calc_rri_by_isi(pred[kuse], isi[kuse], status[kuse], isi_bins2)

            end

            kpair += 1

        end

    end

    return out
end
# ============================================================================ #
function make_figure(d::Dict{String,Any})

    pred_bins = centers(range(0, 1, length=11))
    isi_bins = centers(0.002:0.002:0.124)

    # isi_bins2 = centers(range(0.002, 0.06, length=10))

    BLACK = [0.,0.,0.]

    h = figure()
    h.set_size_inches((8.5,9.5))

    rh = [1.0, 1.0, 1.0]
    rs = [0.06, 0.1, 0.1, 0.06]
    cw = [1.0, 1.0]
    cs = [0.09, 0.1, 0.02]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)

    foreach(default_axes, ax)

    # make axes order / shape consistent w/ PyPlot.subplots()
    ax = permutedims(reshape(ax, 2, 3), (2,1))

    # ------------------------------------------------------------------------ #

    foreach(ax[:,1]) do cax
        cax.plot([0,1], [0,1], "--", color="gray")
    end

    ax[1,1].plot(pred_bins, nanmean(d["grating"]["isi"]["pred_efficacy"], dims=2), ".-", color=BLACK, label="ISI", linewidth=2.5, markersize=10)
    ax[1,1].plot(pred_bins, nanmean(d["grating"]["ff"]["pred_efficacy"], dims=2), ".-", color=GREEN, label="RH", linewidth=2.5, markersize=10)
    ax[1,1].plot(pred_bins, nanmean(d["grating"]["fr"]["pred_efficacy"], dims=2), ".-", color=PURPLE, label="CH", linewidth=2.5, markersize=10)

    ax[2,1].plot(pred_bins, nanmean(d["msequence"]["isi"]["pred_efficacy"], dims=2), ".-", color=BLACK, linewidth=2.5, markersize=10)
    ax[2,1].plot(pred_bins, nanmean(d["msequence"]["ff"]["pred_efficacy"], dims=2), ".-", color=GREEN, linewidth=2.5, markersize=10)
    ax[2,1].plot(pred_bins, nanmean(d["msequence"]["fr"]["pred_efficacy"], dims=2), ".-", color=PURPLE, linewidth=2.5, markersize=10)

    ax[3,1].plot(pred_bins, nanmean(d["awake"]["isi"]["pred_efficacy"], dims=2), ".-", color=BLACK, linewidth=2.5, markersize=10)
    ax[3,1].plot(pred_bins, nanmean(d["awake"]["ff"]["pred_efficacy"], dims=2), ".-", color=GREEN, linewidth=2.5, markersize=10)
    ax[3,1].plot(pred_bins, nanmean(d["awake"]["fr"]["pred_efficacy"], dims=2), ".-", color=PURPLE, linewidth=2.5, markersize=10)

    ax[1,1].legend(frameon=false, fontsize=14)

    ax[3,1].set_xlabel("Predicted efficacy (binned)", fontsize=14)

    foreach(ax[:,1]) do cax
        cax.set_ylabel("Observed efficacy", fontsize=14)
    end

    # ------------------------------------------------------------------------ #

    # ax[1,2].plot(isi_bins, nanmean(d["grating"]["isi"]["obs_isi_efficacy"], dims=2), ".", color="gray", linewidth=2)
    ax[1,2].bar(isi_bins, vec(nanmean(d["grating"]["isi"]["obs_isi_efficacy"], dims=2)), width=0.002*0.9, color="gray", label="data")
    ax[1,2].plot(isi_bins, nanmean(d["grating"]["isi"]["pred_isi_efficacy"], dims=2), "-", color=BLACK, linewidth=2.5, label="ISI")
    ax[1,2].plot(isi_bins, nanmean(d["grating"]["ff"]["pred_isi_efficacy"], dims=2), "-", color=GREEN, linewidth=2.5, label="RH")
    ax[1,2].plot(isi_bins, nanmean(d["grating"]["fr"]["pred_isi_efficacy"], dims=2), "-", color=PURPLE, linewidth=2.5, label="CH")

    # ax[2,2].plot(isi_bins, nanmean(d["msequence"]["isi"]["obs_isi_efficacy"], dims=2), ".", color="gray", linewidth=2.5)
    ax[2,2].bar(isi_bins, vec(nanmean(d["msequence"]["isi"]["obs_isi_efficacy"], dims=2)), width=0.002*0.9, color="gray")
    ax[2,2].plot(isi_bins, nanmean(d["msequence"]["isi"]["pred_isi_efficacy"], dims=2), "-", color=BLACK, linewidth=2.5)
    ax[2,2].plot(isi_bins, nanmean(d["msequence"]["ff"]["pred_isi_efficacy"], dims=2), "-", color=GREEN, linewidth=2.5)
    ax[2,2].plot(isi_bins, nanmean(d["msequence"]["fr"]["pred_isi_efficacy"], dims=2), "-", color=PURPLE, linewidth=2.5)

    # ax[3,2].plot(isi_bins, nanmean(d["awake"]["isi"]["obs_isi_efficacy"], dims=2), ".", color="gray", linewidth=2.5)
    ax[3,2].bar(isi_bins, vec(nanmean(d["awake"]["isi"]["obs_isi_efficacy"], dims=2)), width=0.002*0.9, color="gray")
    ax[3,2].plot(isi_bins, nanmean(d["awake"]["isi"]["pred_isi_efficacy"], dims=2), "-", color=BLACK, linewidth=2.5)
    ax[3,2].plot(isi_bins, nanmean(d["awake"]["ff"]["pred_isi_efficacy"], dims=2), "-", color=GREEN, linewidth=2.5)
    ax[3,2].plot(isi_bins, nanmean(d["awake"]["fr"]["pred_isi_efficacy"], dims=2), "-", color=PURPLE, linewidth=2.5)

    ax[1,2].legend(frameon=false, fontsize=14)

    ax[3,2].set_xlabel("Inter-spike interval (seconds)", fontsize=14)
    # ax[1,2].set_ylabel("Normalized efficacy", fontsize=14)

    foreach(ax[:,2]) do cax
        yl = cax.get_ylim()
        cax.set_ylim([min(yl[1], 0), yl[2]])
        cax.set_xlim(0, 0.13)
        cax.set_ylabel("Normalized efficacy", fontsize=14)
    end

    # ------------------------------------------------------------------------ #

    h.text(0.5, 0.99, "Gratings", fontsize=24, color=RED, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.66, "Binary white noise", fontsize=24, color=BLUE, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.33, "Awake", fontsize=24, color=GOLD, horizontalalignment="center", verticalalignment="top")

    foreach((x,l) -> Plot.axes_label(h, x, l), ax[:,1], ["A","B","C"])

    return h, ax

end
# ============================================================================ #
function calc_pred_efficacy(yp::Vector{<:Real}, isi::Vector{<:Real}, status, edges=range(0, 1, length=11))

    ki = hist_indicies(yp, edges)
    ef = fill(NaN, length(ki))
    for k in eachindex(ki)
        ef[k] = sum(status[ki[k]]) / length(ki[k])
    end

    return ef
end
# ============================================================================ #
function calc_isi_efficacy(yp::Vector{<:Real}, isi::Vector{<:Real}, status, edges=0.002:0.001:0.125)

    ki = hist_indicies(isi, edges)
    obs_ef = fill(NaN, length(ki))
    pred_ef = copy(obs_ef)

    for k in eachindex(ki)
        obs_ef[k] = sum(status[ki[k]]) / length(ki[k])
        pred_ef[k] = mean(yp[ki[k]])
    end

    return obs_ef, pred_ef
end
# ============================================================================ #
function calc_rri_by_isi(yp::Vector{<:Real}, isi::Vector{<:Real}, status, edges=range(0.002, 0.08, length=12), verbose::Bool=false)

    ki = hist_indicies(isi, edges)

    rri = fill(NaN, length(ki))
    for k in eachindex(ki)
        li = RelayUtils.binomial_lli_turbo(yp[ki[k]], status[ki[k]])
        null_li = RelayUtils.binomial_lli(status[ki[k]])
        rri[k] = (li - null_li) / length(ki[k]) / log(2)

        if verbose
            li > null_li ? print("1: ") : print("0: ")
            println("li = $(li / length(ki[k])), null_li = $(null_li / length(ki[k]))")
        end
    end
    return rri
end
# ============================================================================ #
centers(x::AbstractRange) = x[1:end-1] .+ (step(x)/2)
# ============================================================================ #
function hist_indicies(x::AbstractVector{<:Real}, edges::AbstractVector{<:Real})
    ki = [Vector{Int}() for k in 1:(length(edges)-1)]
    for k in eachindex(x)
        kb = searchsortedlast(edges, x[k])
        if 0 < kb < length(edges)
            push!(ki[kb], k)
        end
    end
    return ki
end
# ============================================================================ #
end


# foreach(ax[:,3]) do cax
#     cax.plot([0, isi_bins2[end]], [0,0], "--", color="gray")
# end
#
# ax[1,3].plot(isi_bins2, nanmean(d["grating"]["isi"]["rri_by_isi"], dims=2), ".-", color=BLACK, label="ISI", linewidth=2)
# ax[1,3].plot(isi_bins2, nanmean(d["grating"]["ff"]["rri_by_isi"], dims=2), ".-", color=GREEN, label="RH", linewidth=2)
# ax[1,3].plot(isi_bins2, nanmean(d["grating"]["fr"]["rri_by_isi"], dims=2), ".-", color=PURPLE, label="CH", linewidth=2)
#
# ax[2,3].plot(isi_bins2, nanmean(d["msequence"]["isi"]["rri_by_isi"], dims=2), ".-", color=BLACK, linewidth=2)
# ax[2,3].plot(isi_bins2, nanmean(d["msequence"]["ff"]["rri_by_isi"], dims=2), ".-", color=GREEN, linewidth=2)
# ax[2,3].plot(isi_bins2, nanmean(d["msequence"]["fr"]["rri_by_isi"], dims=2), ".-", color=PURPLE, linewidth=2)
#
# ax[3,3].plot(isi_bins2, nanmean(d["awake"]["isi"]["rri_by_isi"], dims=2), ".-", color=BLACK, linewidth=2)
# ax[3,3].plot(isi_bins2, nanmean(d["awake"]["ff"]["rri_by_isi"], dims=2), ".-", color=GREEN, linewidth=2)
# ax[3,3].plot(isi_bins2, nanmean(d["awake"]["fr"]["rri_by_isi"], dims=2), ".-", color=PURPLE, linewidth=2)
#
# ax[1,3].legend(frameon=false, fontsize=14)
#
# ax[3,3].set_xlabel("Inter-spike interval (seconds)", fontsize=14)
# # ax[1,3].set_ylabel(L"\mathcal{I}_{Bernoulli}", fontsize=16)
#
# foreach(ax[:,3]) do cax
#     cax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
#     cax.set_ylabel(L"\mathcal{I}_{Bernoulli}", fontsize=16)
# end
