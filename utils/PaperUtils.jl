module PaperUtils

using RelayGLM, SpkCore, SimpleStats
using Statistics, Printf, Distributions, Bootstrap

import RelayGLM.RelayUtils
import LinearAlgebra

export get_ef, get_asterix, normalize, rate_split, median_split, quantile_groups
export axes_position, axes_layout
export correlation, count_relayed, contribution, roundn, filter_ci

const EXCLUDE = Dict{String,Vector{Int}}("grating"=>[115], "msequence"=>Int[101,103])

# ---------------------------------------------------------------------------- #
function roundn(x::Real, n::Integer)
    f = 10.0^-n
    return round(x * f) / f
end
# ---------------------------------------------------------------------------- #
contribution(ret::Vector{Float64}, lgn::Vector{Float64}) = findall(>(0.0), RelayUtils.relay_status(lgn, ret))
# ---------------------------------------------------------------------------- #
count_relayed(ret::Vector{Float64}, lgn::Vector{Float64}) = sum(RelayUtils.relay_status(lgn, ret))
# ---------------------------------------------------------------------------- #
function correlation(x, y; method::Symbol=:cor)
    if method == :cor
        r = cor(x, y)
    elseif method == :spearman
        r = spearman(x, y)
    end
    p = 2.0 * ccdf(Normal(), atanh(abs(r)) * sqrt(length(x) - 3))
    return r, p
end
# ---------------------------------------------------------------------------- #
function get_asterix(p::Real)
    if p > 0.05
        return "n.s."
    elseif 0.05 > p >= 0.01
        return "*"
    elseif 0.01 > p >= 0.001
        return "**"
    elseif 0.001 > p
        return "***"
    end
end
# ---------------------------------------------------------------------------- #
normalize(x::AbstractVector{<:Real}) = x ./ LinearAlgebra.norm(x, 2)
function normalize(x::Matrix{<:Real})
    y = copy(x)
    for col in eachcol(y)
        col ./= LinearAlgebra.norm(col, 2)
    end
    return y
end
# ============================================================================ #
function filter_ci(x::Matrix{<:Real})
    val = zeros(size(x, 1))
    hi = copy(val)
    lo = copy(val)

    for k = 1:size(x, 1)
        bs = bmean(x[k,:])
        val[k], lo[k], hi[k] = confint(bs, BCaConfInt(0.95), 1)
    end
    return val, lo, hi
end
# ============================================================================ #
function median_split(x::Vector{<:Real})
    m = median(x)
    kl = Vector{Int}()
    ku = Vector{Int}()
    km = Vector{Int}()
    for k in eachindex(x)
        if x[k] == m
            push!(km, k)
        elseif x[k] < m
            push!(kl, k)
        else
            push!(ku, k)
        end
    end

    if !isempty(km)
        for k in eachindex(km)
            if length(kl) <= length(ku)
                push!(kl, km[k])
            else
                push!(ku, km[k])
            end
        end
    end
    return sort(kl), sort(ku)
end
# ============================================================================ #
function quantile_groups(x::AbstractVector{<:Real}, qt::AbstractVector{<:Real})
    ki = [Vector{Int}() for k in 1:(length(qt)+1)]
    edges = vcat(quantile(x, qt), maximum(x)*1.1)
    kedge = Vector{Int}()
    for k in eachindex(x)
        kb = searchsortedfirst(edges, x[k])
        if x[k] == edges[kb]
            push!(kedge, k)
        elseif kb > length(edges)
            push!(ki[end], k)
        else
            push!(ki[kb], k)
        end
    end

    for k in eachindex(kedge)
        idx = kedge[k]
        kb = searchsortedfirst(edges, x[idx])
        if kb < length(edges) && (length(ki[kb+1]) < length(ki[kb]))
            push!(ki[kb+1], idx)
        else
            push!(ki[kb], idx)
        end
    end

    foreach(sort!, ki)

    return ki
end
# ============================================================================ #
"""
returns a Vector{Vector{Int}} `x` of length N where N is the number of bursts
and the indicies of the spikes that comprise the k'th burst are accessed at
`x[k]` (i.e. `x[k][1]` is the index of the cardinal spike of the k'th burst)
"""
function burst_spikes(ts::Vector{Float64}, bisi::Real=0.004, deadtime::Float64=0.1)
    bs = Vector{Vector{Int}}()
    isi = [0.0; diff(ts)]
    kp = findall(>=(deadtime), isi)
    @inbounds for k in eachindex(kp)
        j = 1
        @inbounds while (kp[k]+j <= length(isi)) && (isi[kp[k]+j] <= bisi)
            if j == 1
                push!(bs, [kp[k]])
            end
            push!(bs[end], kp[k]+j)
            j += 1
        end
    end
    return bs
end
# ============================================================================ #
function basic_stats(x::AbstractVector{<:Real}, method::Symbol=:median; io::IO=stdout, name::String="", units::String="")
    if method == :mean
        val, lo, hi = confint(bmean(x), BCaConfInt(0.95), 1)
        sd = ste(x)
        s1 = "mean"
        s2 = "std"
    elseif method == :median
        val, lo, hi = confint(bmedian(x), BCaConfInt(0.95), 1)
        sd = mad(x)
        s1 = "median"
        s2 = "MAD"
    else
        error("Unsupported method: $(method)")
    end

    name = isempty(name) ? " " : " " * name * " "
    units = isempty(units) ? " " : " " * units * " "

    @printf(io, "%s%s%.3f%s(%s %.3f, 95%% CI [%.3f, %.3f])\n", s1, name, val, units, s2, sd, lo, hi)
end
# ============================================================================ #
end
