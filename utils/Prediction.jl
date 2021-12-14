module Prediction

using Statistics
using RelayGLM
import RelayGLM.RelayISI
import RelayGLM.RelayUtils

function main(ret::Vector{<:Real}, lgn::Vector{<:Real}, pred::Vector{<:Real})

    isi, status = RelayISI.spike_status(ret, lgn)

    e = range(0, 1, length=11)
    ki = hist_indicies(pred, e)
    ef = fill(NaN, length(ki))
    for k in eachindex(ki)
        ef[k] = sum(status[ki[k]]) / length(ki[k])
    end

    e2 = 0.002:0.001:0.125
    ki2 = hist_indicies(isi, e2)
    ef2 = fill(NaN, length(ki2))
    ef3 = copy(ef2)
    for k in eachindex(ki2)
        ef2[k] = sum(status[ki2[k]]) / length(ki2[k])
        ef3[k] = mean(pred[ki2[k]])
    end

    e3 = range(0.002, 0.08, length=12)
    ki3 = hist_indicies(isi, e3)
    rri = fill(NaN, length(ki3))

    for k in eachindex(ki3)
        rri[k] = (RelayUtils.binomial_lli_turbo(pred[ki3[k]], status[ki3[k]]) - RelayUtils.binomial_lli(status[ki3[k]])) / length(ki3[k]) / log(2)
    end

    return centers(e), ef, centers(e2), ef2, ef3, centers(e3), rri
end

centers(x::AbstractRange) = x[1:end-1] .+ (step(x)/2)

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

end
