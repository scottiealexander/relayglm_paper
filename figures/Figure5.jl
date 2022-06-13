module Figure5

using Statistics, PyPlot, JLD, Printf

using RelayGLM, DatabaseWrapper, PaperUtils, Plot
import RelayGLM.RelayISI
import RelayGLM.RelayUtils

using SimpleStats, UCDColors

const Strmbol = Union{String,Symbol}

# ============================================================================ #
function collate_data(data::Dict=Dict(); exclude::Dict{String,Vector{Int}}=PaperUtils.EXCLUDE)

    if isempty(data)
        data = load("../20211216_hp_cv_pred_all.jld")
    end

    tmp = Dict{String,Strmbol}("grating"=>"(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)

    pred_bins = get_prediction_bins()
    # isi_bins = get_isi_bins()
    model_compare_bins = get_model_compare_bins()

    qtls = [0.5]

    out = Dict{String,Any}()

    for (typ, ptrn) in tmp

        db = get_database(ptrn, id -> !in(id, exclude[typ]))

        out[typ] = Dict{String,Any}()

        npair = length(db)
        kpair = 1

        out[typ]["mean_efficacy"] = fill(NaN, npair)
        out[typ]["rh_vs_isi"] = fill(NaN, length(model_compare_bins)-1, npair)
        out[typ]["ch_vs_isi"] = fill(NaN, length(model_compare_bins)-1, npair)

        for model in ["isi","ff","fr"]
            out[typ][model] = Dict{String,VecOrMat{Float64}}()
            out[typ][model]["pred_efficacy"] = fill(NaN, length(pred_bins)-1, npair)
            out[typ][model]["mean_efficacy"] = fill(NaN, npair)

            # out[typ][model]["obs_isi_efficacy"] = fill(NaN, length(isi_bins)-1, npair)
            # out[typ][model]["pred_isi_efficacy"] = fill(NaN, length(isi_bins)-1, npair)
        end

        ids = get_ids(db) #sort(collect(keys(data[typ])))

        out[typ]["ids"] = ids

        for id in ids

            ret, lgn, _, _ = get_data(db, id=id)

            isi, status = RelayISI.spike_status(ret, lgn)

            mean_eff = sum(status) / length(status)
            out[typ]["mean_efficacy"][kpair] = mean_eff

            rh, ch = model_compare_isi(data[typ][id], isi, status, model_compare_bins)

            out[typ]["rh_vs_isi"][:,kpair] .= rh
            out[typ]["ch_vs_isi"][:,kpair] .= ch

            for model in keys(data[typ][id])

                # last element is always NaN (due to a bug in HyperParamsCV), but pred[1] corresponds to isi[1] (NOT ret[1])
                pred = data[typ][id][model]["prediction"][1:end-1]

                @assert(length(pred) == length(isi))

                out[typ][model]["pred_efficacy"][:,kpair] .= calc_pred_efficacy(pred, isi, status, pred_bins)
                out[typ][model]["mean_efficacy"][kpair] = mean(pred)

                # ef2, ef3 = calc_isi_efficacy(pred, isi, status, isi_bins)
                # out[typ][model]["obs_isi_efficacy"][:,kpair] .= ef2 ./ mean_eff
                # out[typ][model]["pred_isi_efficacy"][:,kpair] .= ef3 ./ mean(pred)

            end

            kpair += 1

        end

    end

    return out, data
end
# ============================================================================ #
@inline get_prediction_bins() = range(0.05, 2.6, length=11)
@inline get_model_compare_bins() = range(0.002, 0.124, length=14) # 10 .^ range(log10(0.002), log10(0.124), length=10)#
# @inline get_isi_bins() = 0.002:0.004:0.124
# ============================================================================ #
function make_figure(d::Dict{String,Any})

    pred_bins = centers(get_prediction_bins())
    model_compare_bins = centers(get_model_compare_bins())

    BLACK = [0.,0.,0.]

    h = figure()
    h.set_size_inches((8.5,9.5))

    rh = [1.0, 1.0, 1.0]
    rs = [0.06, 0.1, 0.1, 0.08]
    cw = [1.0, 1.0]
    cs = [0.11, 0.1, 0.02]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)

    foreach(default_axes, ax)

    # make axes order / shape consistent w/ PyPlot.subplots()
    ax = permutedims(reshape(ax, 2, 3), (2,1))

    # ------------------------------------------------------------------------ #
    foreach(ax[:,1]) do cax
        cax.plot([0., 2.6], [0., 2.6], "--", color="gray", linewidth=2, zorder=-10)
    end

    plot_pred_efficacy(pred_bins, d["grating"]["isi"]["pred_efficacy"], BLACK, "ISI", ax[1,1])
    plot_pred_efficacy(pred_bins, d["grating"]["ff"]["pred_efficacy"], GREEN, "RH", ax[1,1])
    plot_pred_efficacy(pred_bins, d["grating"]["fr"]["pred_efficacy"], PURPLE, "CH", ax[1,1])

    plot_pred_efficacy(pred_bins, d["msequence"]["isi"]["pred_efficacy"], BLACK, "ISI", ax[2,1])
    plot_pred_efficacy(pred_bins, d["msequence"]["ff"]["pred_efficacy"], GREEN, "RH", ax[2,1])
    plot_pred_efficacy(pred_bins, d["msequence"]["fr"]["pred_efficacy"], PURPLE, "CH", ax[2,1])

    plot_pred_efficacy(pred_bins, d["awake"]["isi"]["pred_efficacy"], BLACK, "ISI", ax[3,1])
    plot_pred_efficacy(pred_bins, d["awake"]["ff"]["pred_efficacy"], GREEN, "RH", ax[3,1])
    plot_pred_efficacy(pred_bins, d["awake"]["fr"]["pred_efficacy"], PURPLE, "CH", ax[3,1])

    ax[1,1].legend(frameon=false, fontsize=14)

    ax[3,1].set_xlabel("Predicted efficacy\n(normalized, binned)", fontsize=14)

    foreach(ax[:,1]) do cax
        cax.set_ylabel("Observed efficacy\n(normalized)", fontsize=14)
        cax.set_xlim(0, 2.6)
        cax.set_ylim(0, 3)
    end

    # ------------------------------------------------------------------------ #

    x = model_compare_bins

    use_log = is_log(x)

    if use_log
        x .= log10.(x)
    end

    foreach(ax) do cax
        cax.plot([x[1], x[end]], [0,0], "--", color="gray", linewidth=2)
    end

    plot_with_error(x, vec(nanmedian(d["grating"]["rh_vs_isi"], dims=2)),
        vec(nanmad(d["grating"]["rh_vs_isi"], dims=2)), GREEN, ax[1,2], linewidth=2.5, label="RH - ISI")
    plot_with_error(x, vec(nanmedian(d["grating"]["ch_vs_isi"], dims=2)),
        vec(nanmad(d["grating"]["ch_vs_isi"], dims=2)), PURPLE, ax[1,2], linewidth=2.5, label="CH - ISI")

    plot_with_error(x, vec(nanmedian(d["msequence"]["rh_vs_isi"], dims=2)),
        vec(nanmad(d["msequence"]["rh_vs_isi"], dims=2)), GREEN, ax[2,2], linewidth=2.5, label="RH - ISI")
    plot_with_error(x, vec(nanmedian(d["msequence"]["ch_vs_isi"], dims=2)),
        vec(nanmad(d["msequence"]["ch_vs_isi"], dims=2)), PURPLE, ax[2,2], linewidth=2.5, label="CH - ISI")

    plot_with_error(x, vec(nanmedian(d["awake"]["rh_vs_isi"], dims=2)),
        vec(nanmad(d["awake"]["rh_vs_isi"], dims=2)), GREEN, ax[3,2], linewidth=2.5, label="RH - ISI")
    plot_with_error(x, vec(nanmedian(d["awake"]["ch_vs_isi"], dims=2)),
        vec(nanmad(d["awake"]["ch_vs_isi"], dims=2)), PURPLE, ax[3,2], linewidth=2.5, label="CH - ISI")


    foreach(ax[:,2]) do cax
        cax.set_ylabel(L"\Delta \mathcal{I}_{Bernoulli}", fontsize=14)
        if use_log
            xtl = map(x -> @sprintf("%.3f", 10^x), cax.get_xticks())
            cax.set_xticklabels(xtl)
        else
            cax.set_xlim(0, x[end] * 1.02)
        end
    end


    ax[1,2].legend(frameon=false, fontsize=14)
    ax[3,2].set_xlabel("Inter-spike interval (seconds)", fontsize=14)

    ax[1,2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax[2,2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))

    # ------------------------------------------------------------------------ #

    h.text(0.5, 0.99, "Gratings", fontsize=24, color=RED, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.66, "Binary white noise", fontsize=24, color=BLUE, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.34, "Awake", fontsize=24, color=GOLD, horizontalalignment="center", verticalalignment="top")

    foreach((x,l) -> Plot.axes_label(h, x, l), ax[:,1], ["A","B","C"])

    return h, ax

end
# ============================================================================ #
function is_log(x)
    d = diff(x)
    return d[end] > (d[1]*2)
end
# ============================================================================ #
function plot_pred_efficacy(x::AbstractVector{<:Real}, y::AbstractMatrix{<:Real}, color, label, ax)

    use = map(eachrow(y)) do row
        sum(.!isnan.(row)) > 1
    end

    tmp1 = nanmedian(y[use,:], dims=2)
    tmp2 = nanmad(y[use,:], dims=2)
    ax.errorbar(x[use], tmp1, yerr=tmp2, fmt=".-",
        color=color, label=label, linewidth=2.5, markersize=10, capsize=3, capthick=2.5)

    return nothing
end
# ============================================================================ #
function calc_pred_efficacy(yp::Vector{<:Real}, isi::Vector{<:Real}, status, edges=range(0, 1, length=11))

    mean_ef = sum(status) / length(status)

    ki = hist_indicies(yp ./ mean(yp), edges)
    ef = fill(NaN, length(ki))
    for k in eachindex(ki)
        ef[k] = (sum(status[ki[k]]) / length(ki[k])) / mean_ef
    end

    return ef
end
# ============================================================================ #
function model_compare_isi(d::Dict{String,<:Any}, isi::Vector{<:Real}, status, edges)

    ki = hist_indicies(isi, edges)

    isi_pred = d["isi"]["prediction"][1:end-1]
    rh_pred = d["ff"]["prediction"][1:end-1]
    ch_pred = d["fr"]["prediction"][1:end-1]

    rh = fill(NaN, length(ki))
    ch = fill(NaN, length(ki))

    for k in eachindex(ki)
        isi_li = RelayUtils.binomial_lli_turbo(isi_pred[ki[k]], status[ki[k]])

        rh_li = RelayUtils.binomial_lli_turbo(rh_pred[ki[k]], status[ki[k]])
        rh[k] = (rh_li - isi_li) / length(ki[k]) / log(2)

        ch_li = RelayUtils.binomial_lli_turbo(ch_pred[ki[k]], status[ki[k]])
        ch[k] = (ch_li - isi_li) / length(ki[k]) / log(2)
    end
    return rh, ch

end
# ============================================================================ #
centers(x::AbstractRange) = x[1:end-1] .+ (step(x)/2)
centers(x::Vector{<:Real}) = x[1:end-1]
# ============================================================================ #
function hist_indicies(x::AbstractVector{<:Real}, edges::AbstractVector{<:Real})
    ki = [Vector{Int}() for k in 1:(length(edges)-1)]
    for k in eachindex(x)
        # returns the index of the last value in `a` less than or equal to `x`
        kb = searchsortedlast(edges, x[k])
        if 0 < kb < length(edges)
            push!(ki[kb], k)
        end
    end
    return ki
end
# ============================================================================ #
end
