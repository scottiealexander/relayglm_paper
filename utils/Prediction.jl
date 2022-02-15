module Prediction

using Statistics, PyPlot, JLD, Printf, StatsBase

using RelayGLM, DatabaseWrapper, PaperUtils, Plot
import RelayGLM.RelayISI
import RelayGLM.RelayUtils

using SimpleStats, UCDColors

const Strmbol = Union{String,Symbol}

# Prediction
# d = Prediction.collate_data();
# Prediction.make_figure(d);

# ============================================================================ #
@inline get_prediction_bins() = range(0.05, 2.6, length=11)#10 .^ range(log10(0.05), log10(2.6), length=11) #range(0, 1, length=11)
@inline get_isi_bins() = 0.002:0.004:0.124 #10 .^ range(log10(0.002), log10(0.124), length=20) #
@inline get_isi_bins2() = range(0.002, 0.124, length=16) #10 .^ range(log10(0.002), log10(0.1), length=16)
function is_log(x)
    d = diff(x)
    return d[end] > (d[1]*2)
end
# ============================================================================ #
function dist_plot(x::AbstractVector{<:Real}, length::Integer=floor(Int, length(x) / 10))
    mn, mx = extrema(x)
    bins = range(mn - eps(), mx + eps(), length=length)
    return dist_plot(x, bins)
end
function dist_plot(x::AbstractVector{<:Real}, bins::AbstractVector{<:Real})
    h = fit(Histogram, x, bins)
    return centers(bins), h.weights
end
# ============================================================================ #
function scratch(data::Dict{String,Any}; type="grating",cmp=<=, thr=0.03, max_diff=0.4)
    # TODO WARNING FIXME
    #   compare distrubutions of predicted efficacies between models

    bins = range(-max_diff, max_diff, length=51)

    dbnames = Dict{String,Strmbol}("grating"=>"(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)
    db = get_database(dbnames[type], id -> !in(id, PaperUtils.EXCLUDE[type]))

    out = Dict{String,Any}()
    k = 1
    ids = sort(collect(keys(data[type])))
    out["rh_vs_isi"] = fill(NaN, length(bins)-1, length(ids))
    out["ch_vs_isi"] = fill(NaN, length(bins)-1, length(ids))
    for id in ids

        ret, lgn, _, _ = get_data(db, id=id)

        isi, status = RelayISI.spike_status(ret, lgn)

        kuse = findall(cmp(0.03), isi)

        # out["rh_vs_isi"][k] = median(data[type][id]["ff"]["prediction"][1:end-1] .- data[type][id]["isi"]["prediction"][1:end-1])
        # out["ch_vs_isi"][k] = median(data[type][id]["fr"]["prediction"][1:end-1] .- data[type][id]["isi"]["prediction"][1:end-1])

        tmp = fit(Histogram, data[type][id]["ff"]["prediction"][1:end-1][kuse] .- data[type][id]["isi"]["prediction"][1:end-1][kuse], bins)
        out["rh_vs_isi"][:,k] .= (tmp.weights ./ length(kuse))

        tmp = fit(Histogram, data[type][id]["fr"]["prediction"][1:end-1][kuse] .- data[type][id]["isi"]["prediction"][1:end-1][kuse], bins)
        out["ch_vs_isi"][:,k] .= (tmp.weights ./ length(kuse))

        # for model in keys(data[type][id])
        #     if !haskey(out, model)
        #         out[model] = Dict{String,Any}()
        #         out[model] = fill(NaN, length(bins)-1, length(ids))
        #         # out[model]["mean"] = zeros(length(ids))
        #         # out[model]["median"] = zeros(length(ids))
        #     end
        #     tmp = fit(Histogram, data[type][id][model]["prediction"][1:end-1], bins)
        #     out[model][:, k] .= tmp.weights ./ (length(data[type][id][model]["prediction"]) - 1)
        #
        #     if model == "isi"
        #         # ISI model predictions are quantized due to the cross-validated
        #         # look-up table being quantized, so smooth a little to
        #         # compensate for that fact
        #         tmp2 = RelayGLM.RelayISI.smooth_ef(out[model][:, k], step(bins) * 2, step(bins))
        #         out[model][:, k] .= (tmp2 ./ sum(tmp2, dims=1))
        #     end
        #
        #     # mn = nanmean(data[type][id][model]["prediction"])
        #     # md = nanmedian(data[type][id][model]["prediction"])
        #     # out[model]["mean"][k] = mn
        #     # out[model]["median"][k] = md
        # end

        k += 1
    end
    return centers(bins), out
end
# ============================================================================ #
function collate_data(data::Dict{String,Any}=Dict{String,Any}())

    if isempty(data)
        data = load("../20211216_hp_cv_pred_all.jld")
    end

    tmp = Dict{String,Strmbol}("grating"=>"(?:contrast|area|grating)", "msequence"=>"msequence", "awake"=>:weyand)

    pred_bins = get_prediction_bins()
    isi_bins = get_isi_bins()
    isi_bins2 = get_isi_bins2()

    qtls = [0.5]

    out = Dict{String,Any}()

    for (typ, ptrn) in tmp

        db = get_database(ptrn, id -> !in(id, PaperUtils.EXCLUDE[typ]))

        out[typ] = Dict{String,Any}()

        npair = length(db)
        kpair = 1

        out[typ]["mean_efficacy"] = fill(NaN, npair)
        out[typ]["rh_vs_isi"] = fill(NaN, length(isi_bins2)-1, npair)
        out[typ]["ch_vs_isi"] = fill(NaN, length(isi_bins2)-1, npair)

        for model in ["isi","ff","fr"]
            out[typ][model] = Dict{String,VecOrMat{Float64}}()
            out[typ][model]["pred_efficacy"] = fill(NaN, length(pred_bins)-1, npair)
            out[typ][model]["obs_isi_efficacy"] = fill(NaN, length(isi_bins)-1, npair)
            out[typ][model]["pred_isi_efficacy"] = fill(NaN, length(isi_bins)-1, npair)
            out[typ][model]["rri_q"] = fill(NaN, length(qtls) + 1, npair)
            out[typ][model]["mean_efficacy"] = fill(NaN, npair)
            out[typ][model]["rri"] = fill(NaN, npair)
        end

        ids = sort(collect(keys(data[typ])))

        out[typ]["ids"] = ids

        for id in ids

            ret, lgn, _, _ = get_data(db, id=id)

            isi, status = RelayISI.spike_status(ret, lgn)

            # klo = findall(<(0.04), isi)
            # khi = findall(>=(0.04), isi)
            # klo, khi = median_split(isi)
            ki = quantile_groups(isi, qtls)

            mean_eff = sum(status) / length(status)
            out[typ]["mean_efficacy"][kpair] = mean_eff

            rh, ch = model_compare_isi(data[typ][id], isi, status, isi_bins2)

            out[typ]["rh_vs_isi"][:,kpair] .= rh
            out[typ]["ch_vs_isi"][:,kpair] .= ch

            # out[typ]["rh_vs_isi"][:,kpair] .= calc_isi_efficacy(data[typ][id]["ff"]["prediction"][1:end-1] .- data[typ][id]["isi"]["prediction"][1:end-1],
            #     isi, status, isi_bins)[2]
            # out[typ]["ch_vs_isi"][:,kpair] .= calc_isi_efficacy(data[typ][id]["fr"]["prediction"][1:end-1] .- data[typ][id]["isi"]["prediction"][1:end-1],
            #     isi, status, isi_bins)[2]

            for model in keys(data[typ][id])

                # last element is always NaN (due to a bug), but pred[1] corresponds to isi[1] (not ret[1])
                pred = data[typ][id][model]["prediction"][1:end-1]

                @assert(length(pred) == length(isi))

                # kuse = findall(!isnan, pred)

                ef2, ef3 = calc_isi_efficacy(pred, isi, status, isi_bins)

                out[typ][model]["obs_isi_efficacy"][:,kpair] .= ef2 ./ mean_eff
                out[typ][model]["pred_isi_efficacy"][:,kpair] .= ef3 ./ mean(pred)

                out[typ][model]["pred_efficacy"][:,kpair] .= calc_pred_efficacy(pred, isi, status, pred_bins)

                # out[typ][model]["rri_by_isi"][:,kpair] .= calc_rri_by_isi(pred, isi, status, isi_bins2)

                out[typ][model]["mean_efficacy"][kpair] = mean(pred)

                out[typ][model]["rri"][kpair] = rri(pred, status)

                for k in eachindex(ki)
                    out[typ][model]["rri_q"][k, kpair] = rri(pred[ki[k]], status[ki[k]])
                end

            end

            kpair += 1

        end

    end

    return out, data
end
# ============================================================================ #
function rri(yp, y)
    li = RelayUtils.binomial_lli_turbo(yp, y)
    null_li = RelayUtils.binomial_lli(y)
    return (li - null_li) / (length(y) * log(2))
end
# ============================================================================ #
function make_figure(d::Dict{String,Any})

    pred_bins = centers(get_prediction_bins())
    isi_bins = centers(get_isi_bins())

    # isi_bins2 = centers(range(0.002, 0.06, length=10))

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

    # x = range(pred_bins[1], pred_bins[end], length=2)
    foreach(ax[:,1]) do cax
        cax.plot([0., 2.6], [0., 2.6], "--", color="gray", linewidth=2, zorder=-10)
    end

    # val, lo, hi = filter_ci(d["grating"]["isi"]["pred_efficacy"])

    # ax[1,1].errorbar(pred_bins, val,
    #     yerr=hi .- lo, fmt=".-", color=BLACK, label="ISI", linewidth=2.5, markersize=10, capsize=3, capthick=2.5)

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

    use_log = is_log(isi_bins)

    if use_log
        isi_bins .= log10.(isi_bins)
    end

    # ax[1,2].bar(isi_bins, vec(nanmedian(d["grating"]["isi"]["obs_isi_efficacy"], dims=2)), width=isi_bar_width(use_log), color="gray", label="data")
    obs = d["grating"]["isi"]["obs_isi_efficacy"]
    plot_isi_efficacy(isi_bins, d["grating"]["isi"]["pred_isi_efficacy"] .- obs, BLACK, "ISI", ax[1,2])
    plot_isi_efficacy(isi_bins, d["grating"]["ff"]["pred_isi_efficacy"] .- obs, GREEN, "RH", ax[1,2])
    plot_isi_efficacy(isi_bins, d["grating"]["fr"]["pred_isi_efficacy"] .- obs, PURPLE, "CH", ax[1,2])

    # ax[2,2].bar(isi_bins, vec(nanmedian(d["msequence"]["isi"]["obs_isi_efficacy"], dims=2)), width=isi_bar_width(use_log), color="gray")
    obs = d["msequence"]["isi"]["obs_isi_efficacy"]
    plot_isi_efficacy(isi_bins, d["msequence"]["isi"]["pred_isi_efficacy"] .- obs, BLACK, "ISI", ax[2,2])
    plot_isi_efficacy(isi_bins, d["msequence"]["ff"]["pred_isi_efficacy"] .- obs, GREEN, "RH", ax[2,2])
    plot_isi_efficacy(isi_bins, d["msequence"]["fr"]["pred_isi_efficacy"] .- obs, PURPLE, "CH", ax[2,2])

    # ax[3,2].bar(isi_bins, vec(nanmedian(d["awake"]["isi"]["obs_isi_efficacy"], dims=2)), width=isi_bar_width(use_log), color="gray")
    obs = d["awake"]["isi"]["obs_isi_efficacy"]
    plot_isi_efficacy(isi_bins, d["awake"]["isi"]["pred_isi_efficacy"] .- obs, BLACK, "ISI", ax[3,2])
    plot_isi_efficacy(isi_bins, d["awake"]["ff"]["pred_isi_efficacy"] .- obs, GREEN, "RH", ax[3,2])
    plot_isi_efficacy(isi_bins, d["awake"]["fr"]["pred_isi_efficacy"] .- obs, PURPLE, "CH", ax[3,2])


    ax[1,2].legend(frameon=false, fontsize=14)

    ax[3,2].set_xlabel("Inter-spike interval (seconds)", fontsize=14)
    # ax[1,2].set_ylabel("Normalized efficacy", fontsize=14)

    foreach(ax[:,2]) do cax
        yl = cax.get_ylim()
        cax.set_ylim([min(yl[1], 0), yl[2]])
        if use_log
            xtl = map(x -> @sprintf("%.3f", 10^x), cax.get_xticks())
            cax.set_xticklabels(xtl)
        else
            cax.set_xlim(0, isi_bins[end] * 1.05)
        end
        cax.set_ylabel("Efficacy difference", fontsize=14)
        cax.plot([0,isi_bins[end]],[0,0],"--",linewidth=2.5,color="gray",zorder=-10)
    end

    # ------------------------------------------------------------------------ #

    h.text(0.5, 0.99, "Gratings", fontsize=24, color=RED, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.66, "Binary white noise", fontsize=24, color=BLUE, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.34, "Awake", fontsize=24, color=GOLD, horizontalalignment="center", verticalalignment="top")

    foreach((x,l) -> Plot.axes_label(h, x, l), ax[:,1], ["A","B","C"])

    return h, ax

end
# ============================================================================ #
function make_figure2(d)

    isi_bins2 = centers(get_isi_bins2())
    BLACK = [0.,0.,0.]

    h, ax = subplots(3,1)
    h.set_size_inches((5,9))
    foreach(default_axes, ax)
    foreach(ax) do cax
        cax.plot([0, isi_bins2[end]], [0,0], "--", color="gray")
    end

    ax[1].plot(isi_bins2, nanmedian(d["grating"]["isi"]["rri_by_isi"], dims=2), ".-", color=BLACK, label="ISI", linewidth=2)
    ax[1].plot(isi_bins2, nanmedian(d["grating"]["ff"]["rri_by_isi"], dims=2), ".-", color=GREEN, label="RH", linewidth=2)
    ax[1].plot(isi_bins2, nanmedian(d["grating"]["fr"]["rri_by_isi"], dims=2), ".-", color=PURPLE, label="CH", linewidth=2)

    ax[2].plot(isi_bins2, nanmedian(d["msequence"]["isi"]["rri_by_isi"], dims=2), ".-", color=BLACK, linewidth=2)
    ax[2].plot(isi_bins2, nanmedian(d["msequence"]["ff"]["rri_by_isi"], dims=2), ".-", color=GREEN, linewidth=2)
    ax[2].plot(isi_bins2, nanmedian(d["msequence"]["fr"]["rri_by_isi"], dims=2), ".-", color=PURPLE, linewidth=2)

    ax[3].plot(isi_bins2, nanmedian(d["awake"]["isi"]["rri_by_isi"], dims=2), ".-", color=BLACK, linewidth=2)
    ax[3].plot(isi_bins2, nanmedian(d["awake"]["ff"]["rri_by_isi"], dims=2), ".-", color=GREEN, linewidth=2)
    ax[3].plot(isi_bins2, nanmedian(d["awake"]["fr"]["rri_by_isi"], dims=2), ".-", color=PURPLE, linewidth=2)

    ax[1].legend(frameon=false, fontsize=14)

    ax[3].set_xlabel("Inter-spike interval (seconds)", fontsize=14)
    # ax[1,3].set_ylabel(L"\mathcal{I}_{Bernoulli}", fontsize=16)

    foreach(ax) do cax
        cax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
        cax.set_ylabel(L"\mathcal{I}_{Bernoulli}", fontsize=16)
    end
    h.tight_layout()
end
# ============================================================================ #
function make_figure3(d)
    BLACK = [0.,0.,0.]

    h, ax = subplots(3,1)
    h.set_size_inches((5,9))
    foreach(default_axes, ax)

    x = 1:size(d["awake"]["isi"]["rri_q"],1)

    ax[1].plot(x, d["grating"]["isi"]["rri_q"], ".-", color=BLACK, linewidth=2, markersize=8)
    ax[1].plot(x .+ (x[end] - .5), d["grating"]["ff"]["rri_q"], ".-", color=GREEN, linewidth=2, markersize=8)
    ax[1].plot(x .+ (x[end] - .5) * 2, d["grating"]["fr"]["rri_q"], ".-", color=PURPLE, linewidth=2, markersize=8)

    ax[2].plot(x, d["msequence"]["isi"]["rri_q"], ".-", color=BLACK, linewidth=2, markersize=8)
    ax[2].plot(x .+ (x[end] - .5), d["msequence"]["ff"]["rri_q"], ".-", color=GREEN, linewidth=2, markersize=8)
    ax[2].plot(x .+ (x[end] - .5) * 2, d["msequence"]["fr"]["rri_q"], ".-", color=PURPLE, linewidth=2, markersize=8)

    ax[3].plot(x, d["awake"]["isi"]["rri_q"], ".-", color=BLACK, linewidth=2, markersize=8)
    ax[3].plot(x .+ (x[end] - .5), d["awake"]["ff"]["rri_q"], ".-", color=GREEN, linewidth=2, markersize=8)
    ax[3].plot(x .+ (x[end] - .5) * 2, d["awake"]["fr"]["rri_q"], ".-", color=PURPLE, linewidth=2, markersize=8)


    h.tight_layout()
end
# ============================================================================ #
function make_figure4(d)

    BLACK = [0.,0.,0.]

    h = figure()
    h.set_size_inches((4.5,9.5))

    rh = [1.0, 1.0, 1.0]
    rs = [0.07, 0.11, 0.11, 0.06]
    cw = [1.0]
    cs = [0.19, 0.05]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)

    foreach(default_axes, ax)

    x = centers(get_isi_bins2())
    use_log = is_log(x)
    if use_log
        x .= log10.(x)
    end

    foreach(ax) do cax
        cax.plot([x[1], x[end]], [0,0], "--", color="gray", linewidth=2)
    end

    plot_with_error(x, vec(nanmedian(d["grating"]["rh_vs_isi"], dims=2)),
        vec(nanmad(d["grating"]["rh_vs_isi"], dims=2)), GREEN, ax[1], linewidth=2.5, label="RH - ISI")
    plot_with_error(x, vec(nanmedian(d["grating"]["ch_vs_isi"], dims=2)),
        vec(nanmad(d["grating"]["ch_vs_isi"], dims=2)), PURPLE, ax[1], linewidth=2.5, label="CH - ISI")

    plot_with_error(x, vec(nanmedian(d["msequence"]["rh_vs_isi"], dims=2)),
        vec(nanmad(d["msequence"]["rh_vs_isi"], dims=2)), GREEN, ax[2], linewidth=2.5, label="RH - ISI")
    plot_with_error(x, vec(nanmedian(d["msequence"]["ch_vs_isi"], dims=2)),
        vec(nanmad(d["msequence"]["ch_vs_isi"], dims=2)), PURPLE, ax[2], linewidth=2.5, label="CH - ISI")

    plot_with_error(x, vec(nanmedian(d["awake"]["rh_vs_isi"], dims=2)),
        vec(nanmad(d["awake"]["rh_vs_isi"], dims=2)), GREEN, ax[3], linewidth=2.5, label="RH - ISI")
    plot_with_error(x, vec(nanmedian(d["awake"]["ch_vs_isi"], dims=2)),
        vec(nanmad(d["awake"]["ch_vs_isi"], dims=2)), PURPLE, ax[3], linewidth=2.5, label="CH - ISI")

    foreach(ax) do cax
        # yl = cax.get_ylim()
        # cax.set_ylim([min(yl[1], 0), yl[2]])
        if use_log
            xtl = map(x -> @sprintf("%.3f", 10^x), cax.get_xticks())
            cax.set_xticklabels(xtl)
        else
            cax.set_xlim(0, 0.1)
        end
        cax.set_ylabel(L"\Delta \mathcal{I}_{Bernoulli}", fontsize=14)
    end


    ax[1].legend(frameon=false, fontsize=14)
    ax[3].set_xlabel("Inter-spike interval (seconds)", fontsize=14)

    ax[1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.03))
    ax[2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))

    h.text(0.5, 0.985, "Gratings", fontsize=24, color=RED, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.655, "Binary white noise", fontsize=24, color=BLUE, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.33, "Awake", fontsize=24, color=GOLD, horizontalalignment="center", verticalalignment="top")

    foreach((x,l) -> Plot.axes_label(h, x, l), ax, ["A","B","C"])

    return h, ax
end
# ============================================================================ #
function plot_pred_efficacy_diff(data)

    h = figure()
    h.set_size_inches((8.5,9.5))

    rh = [1.0, 1.0, 1.0]
    rs = [0.07, 0.12, 0.12, 0.08]
    cw = [1.0, 1.0]
    cs = [0.11, 0.1, 0.02]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)

    foreach(default_axes, ax)

    # make axes order / shape consistent w/ PyPlot.subplots()
    ax = permutedims(reshape(ax, 2, 3), (2, 1))

    k = 1

    for type in ["grating","msequence","awake"]

        if type == "msequence"
            max_diff = 0.15
        elseif type == "awake"
            max_diff = 0.45
        else
            max_diff = 0.35
        end

        bins, lo = scratch(data, type=type, cmp= <= , thr=0.03, max_diff=max_diff)
        _, hi = scratch(data, type=type, cmp= > , thr=0.03, max_diff=max_diff)

        plot_with_error(bins, vec(nanmedian(lo["rh_vs_isi"], dims=2)), vec(nanmad(lo["rh_vs_isi"], dims=2)), GREEN, ax[k,1], linewidth=3, label="RH - ISI")
        plot_with_error(bins, vec(nanmedian(lo["ch_vs_isi"], dims=2)), vec(nanmad(lo["ch_vs_isi"], dims=2)), PURPLE, ax[k,1], linewidth=3, label="CH - ISI")
        ax[k, 1].set_title("ISIs \$\\leq\$ 0.03 sec", fontsize=18)

        plot_with_error(bins, vec(nanmedian(hi["rh_vs_isi"], dims=2)), vec(nanmad(hi["rh_vs_isi"], dims=2)), GREEN, ax[k,2], linewidth=3, label="RH - ISI")
        plot_with_error(bins, vec(nanmedian(hi["ch_vs_isi"], dims=2)), vec(nanmad(hi["ch_vs_isi"], dims=2)), PURPLE, ax[k,2], linewidth=3, label="CH - ISI")
        ax[k, 2].set_title("ISIs > 0.03 sec", fontsize=18)

        k += 1
    end

    foreach(ax) do cax
        yl = cax.get_ylim()
        cax.plot([0,0],[0,yl[2] * 1.05], "--", color="gray", linewidth=2.5, zorder=-10)
    end

    foreach(ax[:,1]) do cax
        cax.set_ylabel("Average probability", fontsize=14)
    end

    ax[1,1].legend(frameon=false, fontsize=14)
    ax[1,2].legend(frameon=false, fontsize=14)

    foreach(ax[3,:]) do cax
        cax.set_xlabel("Difference in\npredicted efficacy", fontsize=14)
    end

    h.text(0.5, 0.99, "Gratings", fontsize=24, color=RED, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.68, "Binary white noise", fontsize=24, color=BLUE, horizontalalignment="center", verticalalignment="top")
    h.text(0.5, 0.35, "Awake", fontsize=24, color=GOLD, horizontalalignment="center", verticalalignment="top")

    foreach((x,l) -> Plot.axes_label(h, x, l), ax[:,1], ["A","B","C"])

    return h, ax
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
function plot_isi_efficacy(x::AbstractVector{<:Real}, y::AbstractMatrix{<:Real}, color, label, ax)

    plot_with_error(x, vec(nanmedian(y, dims=2)), vec(nanmad(y, dims=2)),
        color, ax, linewidth=2.5, label=label)

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
function isi_bar_width(use_log::Bool=false)
    x = get_isi_bins()
    if use_log
        x .= log10.(x)
    end
    return diff(x) .* 0.9
end
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
