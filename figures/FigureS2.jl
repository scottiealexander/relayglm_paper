module FigureS2

using RelayGLM, DatabaseWrapper, PaperUtils, Progress, SimpleStats
import RelayGLM.RelayISI, Figure8
using PyPlot, Plot, Statistics, ColorTypes, UCDColors
import JSON
# ============================================================================ #
function collate_data(; extended::Bool=false)

    span = 200
    bin_size = 0.001

    db = get_database(:weyand, id -> !in(id, PaperUtils.EXCLUDE["awake"]))

    d45 = open(JSON.parse, joinpath(@__DIR__, "..", "preprocessed_data", "figure6_7.json"), "r")

    d = Dict{String,Any}()

    d["rri"] = d45["awake"]

    d["ids"] = zeros(Int, length(db))

    if extended
        d["efficacy"] = zeros(length(db))
        d["isi"] = zeros(span-2, length(db))
        d["rh"] = zeros(span, length(db))
        d["ch_ret"] = zeros(span, length(db))
        d["ch_lgn"] = zeros(span, length(db))
    end

    d["rh_lo"] = zeros(span, length(db))
    d["rh_hi"] = zeros(span, length(db))

    show_progress(0.0, 0, "Awake: ", "(0 of $(length(db)))")

    for k in 1:length(db)

        id = get_id(db[k])
        ret, lgn, _, _ = get_data(db, k)

        if extended
            ed, ef = isi_efficacy(ret, lgn, span, 0.002, bin_size)

            rh, mean_ef = rh_filter(ret, lgn, span, bin_size)
            ch_ret, ch_lgn = ch_filters(ret, lgn, span, span, 24, bin_size)

            if !haskey(d, "isi_labels")
                d["isi_labels"] = ed[1:end-1] .+ (step(ed)/2)
            end
        end

        rh_lo, rh_hi = activity_state(ret, lgn, 0.1, span, bin_size)

        d["ids"][k] = id

        if extended
            d["efficacy"][k] = mean_ef
            d["isi"][:,k] .= ef
            d["rh"][:,k] .= rh
            d["ch_ret"][:,k] .= ch_ret
            d["ch_lgn"][:,k] .= ch_lgn
        end

        d["rh_lo"][:,k] .= rh_lo
        d["rh_hi"][:,k] .= rh_hi

        show_progress(k/length(db), 0, "Awake: ", "($(k) of $(length(db)))")
    end

    return d
end
# ============================================================================ #
function make_figure(d::Dict{String,Any})

    h = figure()
    h.set_size_inches((9,5.5))

    rh = [1.0]
    rs = [0.1, 0.1]
    cw = [0.67, 1.0]
    cs = [0.10, 0.1, 0.03]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)

    foreach(default_axes, ax)

    t_rh = -0.201:0.001:-0.002

    N = length(d["ids"])

    foreach(["isi","ff","fr"],["black",GREEN,PURPLE],0:2) do field, col, k
        m = median(d["rri"]["rri"][field])
        sd = mad(Vector{Float64}(d["rri"]["rri"][field]))

        ax[1].plot(k, m, ".", markersize=18, color=col)
        ax[1].plot([k, k], [m-sd, m+sd], "-", linewidth=5, color=col)

        ax[1].plot(fill(k, N), d["rri"]["rri"][field], ".", color="gray", markersize=12, alpha=0.6)
    end

    ax[1].plot(hcat(fill(0, N), fill(1, N))', hcat(d["rri"]["rri"]["isi"], d["rri"]["rri"]["ff"])', "-", color="gray", linewidth=2, alpha=0.6)
    ax[1].plot(hcat(fill(1, N), fill(2, N))', hcat(d["rri"]["rri"]["ff"], d["rri"]["rri"]["fr"])', "-", color="gray", linewidth=2, alpha=0.6)

    ax[1].set_ylim(0.0, 0.52)
    ax[1].set_xticks(0:2)
    ax[1].set_xticklabels(["ISI\nmodel", "Retinal\nmodel", "Combined\nmodel"])
    ax[1].set_xlim(-0.5, 2.5)
    ax[1].set_ylabel("\$\\mathcal{I}_{Bernoulli}\$", fontsize=14)
    ax[1].set_title("Model performance", fontsize=16)

    inset_length = 30

    sax = Figure8.add_subplot_axes(ax[2], [0.08, 0.35, 0.48, 0.6])
    default_axes(sax)
    sax.set_yticklabels([])
    ki = length(t_rh)-inset_length
    sax.plot([t_rh[ki], t_rh[end]], [0,0], "--", color="black", linewidth=1)
    sax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))

    ax[2].plot([t_rh[1]-0.005, t_rh[end]], [0,0], "--", color="black", linewidth=1)
    ax[2].set_xlim(t_rh[1] - 0.005, t_rh[end] + 0.008)
    ax[2].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))

    ki = findall(!isequal(200001250), d["ids"])

    mnl, mxl = filter_plot(t_rh, d["rh_lo"], ax[2], sax, GREEN, "Low", inset_length)
    mnh, mxh = filter_plot(t_rh, d["rh_hi"], ax[2], sax, PURPLE, "High", inset_length)
    mn = min(mnl, mnh)
    mx = max(mxl, mxh)

    Figure8.inset_box(t_rh, mn, mx, ax[2], inset_length)

    # k = findfirst(isequal(200001250), d["ids"])
    # kt = length(t_rh)-inset_length+1:length(t_rh)
    # sax.plot(t_rh[kt], PaperUtils.normalize(d["rh_lo"])[kt, k], linewidth=1.5, linestyle="--", color=GREEN)
    # sax.plot(t_rh[kt], PaperUtils.normalize(d["rh_hi"])[kt, k], linewidth=1.5, linestyle="--", color=PURPLE)

    # ax[2].plot(t_rh, PaperUtils.normalize(d["rh_lo"])[:, k], linewidth=1.5, linestyle="--", color=GREEN)
    # ax[2].plot(t_rh, PaperUtils.normalize(d["rh_hi"])[:, k], linewidth=1.5, linestyle="--", color=PURPLE)


    # draw two arrows pointing to the filters learned from pair 200001250, which
    # is the only pair (so far as I know) that actually had a controled stimulus
    # which happened to be a grating
    t = matplotlib.markers.MarkerStyle(marker="\$\\uparrow\$")

    t._transform = t.get_transform().rotate_deg(65)
    sax.scatter(-0.0085, -0.13, marker=t, s=100, c=reshape(RED, 1, :))

    t._transform = t.get_transform().rotate_deg(-65)
    sax.scatter(-0.014, -0.062, marker=t, s=100, c=reshape(RED, 1, :))

    ax[2].set_xlabel("Time before spike (seconds)", fontsize=14)
    ax[2].set_ylabel("Filter weight (A.U.)", fontsize=14)
    ax[2].legend(frameon=false, fontsize=14, loc="upper right", bbox_to_anchor=[0.83, 1.0])
    ax[2].set_title("High vs. low activity level", fontsize=16)

    labels = ["A","B"]
    foreach((k,l)->Plot.axes_label(h, ax[k], l), 1:2, labels)

    return h
end
# ============================================================================ #
function make_extended_figure(d::Dict{String,Any})

    h = figure()
    h.set_size_inches((9,9.5))

    rh = [1.0, 1.0, 1.4]
    rs = [0.06, 0.11, 0.13, 0.06]
    cw = [1.0, 1.0]
    cs = [0.10, 0.1, 0.03]

    ax = Plot.axes_layout(h, row_height=rh, row_spacing=rs, col_width=cw, col_spacing=cs)

    foreach(default_axes, ax)

    t_rh = -0.201:0.001:-0.002

    N = length(d["ids"])

    foreach(ax[[1,3,4]]) do cax
        cax.plot([t_rh[1], t_rh[end]], [0,0], "--", color="black", linewidth=1)
    end

    ax[1].plot(t_rh, PaperUtils.normalize(d["rh"]), color="gray", linewidth=1.5)
    ax[1].plot(t_rh, mean(PaperUtils.normalize(d["rh"]), dims=2), color=GOLD, linewidth=3, label="Population mean (n=$(N))")
    ax[1].set_xlabel("Time before spike (seconds)", fontsize=14)
    ax[1].set_ylabel("Filter weight (A.U.)", fontsize=14)
    ax[1].set_title("RH retinal filter", fontsize=16)

    ax[1].legend(frameon=false, fontsize=14)

    isi_norm = d["isi"] ./ reshape(d["efficacy"], 1, :)

    ax[2].plot(d["isi_labels"], isi_norm, color="gray", linewidth=1.5)
    ax[2].plot(d["isi_labels"], mean(isi_norm, dims=2), color=GOLD, linewidth=3, label="Population mean (n=$(N))")
    ax[2].set_xlabel("Inter-spike interval (seconds)", fontsize=14)
    ax[2].set_ylabel("Normalized efficacy\n(A.U.)", fontsize=14)
    ax[2].set_title("ISI-efficacy", fontsize=16)

    ax[3].plot(t_rh, PaperUtils.normalize(d["ch_ret"]), color="gray", linewidth=1.5)
    ax[3].plot(t_rh, mean(PaperUtils.normalize(d["ch_ret"]), dims=2), color=GOLD, linewidth=3, label="Population mean (n=$(N))")
    ax[3].set_xlabel("Time before spike (seconds)", fontsize=14)
    ax[3].set_ylabel("Filter weight (A.U.)", fontsize=14)
    ax[3].set_title("CH retinal filter", fontsize=16)

    ax[4].plot(t_rh, PaperUtils.normalize(d["ch_lgn"]), color="gray", linewidth=1.5)
    ax[4].plot(t_rh, mean(PaperUtils.normalize(d["ch_lgn"]), dims=2), color=GOLD, linewidth=3, label="Population mean (n=$(N))")
    ax[4].set_xlabel("Time before spike (seconds)", fontsize=14)
    ax[4].set_ylabel("Filter weight (A.U.)", fontsize=14)
    ax[4].set_title("CH LGN filter", fontsize=16)

    foreach(["isi","ff","fr"],["black",GREEN,PURPLE],0:2) do field, col, k
        m = median(d["rri"]["rri"][field])
        sd = mad(Vector{Float64}(d["rri"]["rri"][field]))

        ax[5].plot(k, m, ".", markersize=18, color=col)
        ax[5].plot([k, k], [m-sd, m+sd], "-", linewidth=5, color=col)

        ax[5].plot(fill(k, N), d["rri"]["rri"][field], ".", color="gray", markersize=12, alpha=0.6)
    end

    ax[5].plot(hcat(fill(0, N), fill(1, N))', hcat(d["rri"]["rri"]["isi"], d["rri"]["rri"]["ff"])', "-", color="gray", linewidth=2, alpha=0.6)
    ax[5].plot(hcat(fill(1, N), fill(2, N))', hcat(d["rri"]["rri"]["ff"], d["rri"]["rri"]["fr"])', "-", color="gray", linewidth=2, alpha=0.6)

    ax[5].set_ylim(0.0, 0.5)
    ax[5].set_xticks(0:2)
    ax[5].set_xticklabels(["ISI\nmodel", "Retinal\nmodel", "Combined\nmodel"])
    ax[5].set_xlim(-0.5, 2.5)
    ax[5].set_ylabel("\$\\mathcal{I}_{Bernoulli}\$", fontsize=14)
    ax[5].set_title("Model performance", fontsize=16)

    inset_length = 30

    sax = Figure8.add_subplot_axes(ax[6], [0.2, 0.4, 0.45, 0.6])
    default_axes(sax)
    sax.set_yticklabels([])
    ki = length(t_rh)-inset_length
    sax.plot([t_rh[ki], t_rh[end]], [0,0], "--", color="black", linewidth=1)
    sax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))

    ax[6].plot([t_rh[1], t_rh[end]], [0,0], "--", color="black", linewidth=1)

    mnl, mxl = filter_plot(t_rh, d["rh_lo"], ax[6], sax, GREEN, "Low", inset_length)
    mnh, mxh = filter_plot(t_rh, d["rh_hi"], ax[6], sax, PURPLE, "High", inset_length)
    mn = min(mnl, mnh)
    mx = max(mxl, mxh)

    Figure8.inset_box(t_rh, mn, mx, ax[6], inset_length)

    ax[6].set_xlabel("Time before spike (seconds)", fontsize=14)
    ax[6].set_ylabel("Filter weight (A.U.)", fontsize=14)
    ax[6].legend(frameon=false, fontsize=14)
    ax[6].set_title("High vs. low activity level", fontsize=16)

    resized_axes(ax[5], [0,0], [-.1, 0])
    resized_axes(ax[6], [-0.08,0], [0,0])

    labels = ["A","B","C","D","E","F"]
    foreach((k,l)->Plot.axes_label(h, ax[k], l), 1:6, labels)

end
# ============================================================================ #
function resized_axes(ax, d0, d1)
    bbox = ax.get_position()
    p0 = bbox.p0 .+ d0
    p1 = bbox.p1 .+ d1
    bbox.update_from_data_xy([p0, p1], ignore=true)
    ax.set_position(bbox)
    return ax
end
# ============================================================================ #
function filter_plot(t, xf, ax, sax, col, label, inset_length)

    light_col, _ = Plot.shading_color(col, 4.0)
    tmp = PaperUtils.normalize(xf)
    ax.plot(t, tmp, linewidth=1.5, color=light_col)
    ax.plot(t, mean(tmp, dims=2), linewidth=3.5, color=col, label=label, zorder=100)

    mn, mx = 0, 0

    if inset_length > 0
        ki = length(t)-inset_length+1:length(t)
        sax.plot(t[ki], tmp[ki,:], linewidth=1.5, color=light_col)
        sax.plot(t[ki], mean(tmp[ki,:], dims=2), linewidth=3.5, color=col, label=label, zorder=100)
        mn, mx = extrema(tmp[ki,:]) .* 1.05
    end

    return mn, mx
end
# ============================================================================ #
function isi_efficacy(ret, lgn, span, sigma, bin_size)

    isi, status = RelayISI.spike_status(ret, lgn)
    edges, eff = RelayISI.get_eff(isi, status, 1:length(isi), sigma, bin_size, span * bin_size)

    return edges, eff
end
# ============================================================================ #
function rh_filter(ret, lgn, span, bin_size)

    response = wasrelayed(ret, lgn)

    lm = 2.0 .^ range(1, 12, length=8)

    ps = PredictorSet()
    ps[:retina] = Predictor(ret, ret, DefaultBasis(length=span, offset=2, bin_size=bin_size))
    glm = GLM(ps, response, SmoothingPrior, [lm])

    res = cross_validate(RRI, Binomial, Logistic, glm, nfold=10, shuffle_design=true)

    return get_coef(res, :retina), sum(response) / length(ret)
end
# ============================================================================ #
function ch_filters(ret, lgn, ffspan, fbspan, fbnb, bin_size)

    response = wasrelayed(ret, lgn)

    lm = 2.0 .^ range(-3.5, 3, length=5)

    ps = PredictorSet()
    ps[:retina] = Predictor(ret, ret, CosineBasis(length=Int(ffspan), offset=2, nbasis=16, b=10, ortho=false, bin_size=bin_size))
    ps[:lgn] = Predictor(lgn, ret, CosineBasis(length=Int(fbspan), offset=2, nbasis=Int(fbnb), b=8, ortho=false, bin_size=bin_size))

    glm = GLM(ps, response, RidgePrior, [lm])
    res = cross_validate(RRI, Binomial, Logistic, glm, nfold=10, shuffle_design=true)

    return get_coef(res, :retina), get_coef(res, :lgn)
end
# ============================================================================ #
function activity_state(ret, lgn, twin, span, bin_size)

    klo, khi = Figure8.rate_split(ret, lgn, round(Int, twin / bin_size), 2)

    lo = run_one(klo, ret, lgn, span, bin_size)
    hi = run_one(khi, ret, lgn, span, bin_size)

    return get_coef(lo, :retina), get_coef(hi, :retina)
end
# ============================================================================ #
function run_one(kuse::AbstractVector{<:Integer}, ret::AbstractVector{<:Real}, lgn::AbstractVector{<:Real}, span, bin_size::Real)

    response = wasrelayed(ret[kuse], lgn)

    ps = PredictorSet()
    ps[:retina] = Predictor(ret, ret[kuse], CosineBasis(length=span, offset=2, nbasis=16, b=10, ortho=false, bin_size=bin_size))
    lm = 2.0 .^ range(-3.5, 3, length=5)
    glm = GLM(ps, response, RidgePrior, [lm])

    return cross_validate(RRI, Binomial, Logistic, glm, nfold=10, shuffle_design=true)
end
# ============================================================================ #
end
