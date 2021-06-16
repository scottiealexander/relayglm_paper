module MSequence

using Statistics

export sta

# ============================================================================ #
struct MSeq{T}
    seq::Vector{T}
    frame::Matrix{T}
end
# ---------------------------------------------------------------------------- #
MSeq(seq::Vector{T}) where T <: Real = MSeq(seq, zeros(T, 16, 16))
# ============================================================================ #
function get_frame(seq::MSeq, k::Integer) where T <: Real

    len = length(seq.seq)

    # shift from 1-based to 0-based so our index wrap (via mod) works smoothly
    # in the loop (THIS is (one reason) why indices *SHOULD* be 0-based)
    idx = mod(k-1, len)
    inc = 0
    for k in eachindex(seq.frame)
        seq.frame[k] = seq.seq[mod(idx + inc, len) + 1] * 2 - 1
        inc += 128
    end

    return seq.frame
end
# ============================================================================ #
"""
`sta(ts::Vector{Float64}, evt::Vector{Float64}, seq::Vector{<:Real}, ifi::AbstractFloat)`

Inputs:

* ts - vector of spike timestamps
* evt - vector of frame onset times (stimulus frames, not monitor frames)
* seq - binary m-sequence used (vector of UInt8 or Bool recommended)
* ifi - inter-frame-interval (or stimulus frame duration) in SECONDS

Output:

* rf - STRF as a 16x16x16 Array{Float64,3} (width x height x time)

"""
function sta(ts::Vector{Float64}, evt::Vector{Float64}, seq::Vector{<:Real}, ifi::AbstractFloat)

    mseq = MSeq(seq)
    rf = zeros(16,16,16)
    last = 1
    nframe = 0
    klast = lastindex(ts)
    for k in eachindex(evt)
        # index of first spike >= frame onset time
        ks = searchsortedfirst(view(ts, last:klast), evt[k]) + last - 1

        # index of last spike that is <= frame offset time
        kl = searchsortedlast(view(ts, ks:klast), evt[k] + ifi) + ks - 1

        # number of spikes (length(ks:kl))
        n = kl - ks + 1
        if n > 0
            for j = 0:min(15, k-1)
                rf[:,:,j+1] .+= (get_frame(mseq, k - j) .* n)
            end
            last = kl + 1
            nframe += 1
        end
    end

    # transform units to mean spikes-per-second
    return (rf ./ nframe) ./ ifi
end
# ============================================================================ #
end
