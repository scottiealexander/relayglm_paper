module RejectionSampling

export get_sampler, sample, trapz

abstract type AbstractSampler{I,O} end
# ============================================================================ #
struct ContinuousSampler{T} <: AbstractSampler{T,Float64}
    x1::T
    x2::T
    ymx::Float64
    f::Function
end

function generate_candidate(c::ContinuousSampler)
    x, y = rand(2)
    return (x * (c.x2 - c.x1)) + c.x1, y * c.ymx
end

# source: https://web.mit.edu/urban_or_book/www/book/chapter7/7.1.3.html
function sample(c::ContinuousSampler)
    done = false
    while !done
        x, y = generate_candidate(c)
        # maybe should be "<=" but source suggests "<" is correct
        if y < c.f(x)
            return x
        end
    end
end
# ============================================================================ #
struct DiscreteSampler{T<:AbstractVector, L<:AbstractVector{<:Real}} <: AbstractSampler{eltype(T),eltype(L)}
    x::T
    y::L
end

generate_candidate(d::DiscreteSampler) = rand(1:length(d.x)), rand()

function sample(d::DiscreteSampler)
    done = false
    while !done
        k, y = generate_candidate(d)
        if y < d.y[k]
            return d.x[k]
        end
    end
end
# ============================================================================ #
function trapz(y::AbstractVector{<:Number})
    out = 0.5 * (y[1] + y[end])
    @inbounds @fastmath for k in 2:(length(y)-1)
        out += y[k]
    end
    return out
end
# ---------------------------------------------------------------------------- #
function trapz(x::AbstractVector{<:Number}, y::AbstractVector{<:Number})
    (length(x) != length(y)) && error("Inputs *MUST* be the same length!")
    r = 0.0
    @inbounds @fastmath for k = 2:length(y)
        r += (x[k] - x[k-1]) * (y[k] + y[k-1])
    end
    return r/2.0
end
# ============================================================================ #
get_sampler(x1::Number, x2::Number, ymx::Float64, f::Function) = ContinuousSampler(x1, x2, ymx, f)
get_sampler(x::AbstractVector, y::AbstractVector) = DiscreteSampler(x, y)
# ---------------------------------------------------------------------------- #
function sample(d::AbstractSampler{T,L}, n::Integer) where {T,L}
    out = Vector{L}(undef, n)
    @inbounds for k in eachindex(out)
        out[k] = sample(d)
    end
    return out
end
# ============================================================================ #
end
