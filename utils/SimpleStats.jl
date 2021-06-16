module SimpleStats

using Statistics, Random, Bootstrap
import Statistics

export ste, mad, iqr, circular_var, spearman, cor_partial
export nanmean, nanstd, nanste, nanvar
export bmean, bmedian, bcor, bspearman, bcor_partial
export permutation_test, paired_permutation_test
export pvalue

export Both, Left, Right
# ============================================================================ #

nanfun(fun::Function, x::Vector{<:Real}) = fun(filter(!isnan, x))

function nanfun(fun::Function, x::Matrix{<:Real}; dims::Integer=1)
    return mapslices(x->nanfun(fun, x), x, dims=dims)
end

ste(x::AbstractVector{<:Real}) = Statistics.std(x) ./ sqrt(length(x))
ste(x::AbstractArray{<:Real}; dims::Integer=1) = Statistics.std(x, dims=dims) ./ sqrt(size(x, dims))

nanmean(x::VecOrMat{<:Real}) = nanfun(mean, x)
nanstd(x::VecOrMat{<:Real}) = nanfun(Statistics.std, x)
nanvar(x::VecOrMat{<:Real}) = nanfun(Statistics.var, x)
nanste(x::VecOrMat{<:Real}) = nanfun(ste, x)
# ============================================================================ #
"inter-quartile range"
iqr(x::AbstractVector{<:Real}) = diff(quantile(x, [0.25, 0.75]))[1]

"median absolute deviation"
mad(x::Vector{<:Real}) = median(abs.(x .- median(x)))
# ============================================================================ #
"circular variance"
circular_var(x::AbstractVector{<:Real}, w::AbstractVector{<:Real}=ones(length(x))) = 1.0 - resultant(x, w)

function resultant(x::AbstractVector{<:Real}, w::AbstractVector{<:Real}=ones(length(x)))
    r = sum(w .* exp.(1im .* x))
    return abs(r) / sum(w)
end
# ============================================================================ #
spearman(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = cor(tiedrank(x), tiedrank(y))
# ============================================================================ #
function cor_partial(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, z::AbstractVector{<:Real})
    p_xz = cor(x, z)
    p_yz = cor(y, z)
    return (cor(x, y) - (p_xz * p_yz)) / (sqrt(1 - p_xz^2) * sqrt(1 - p_yz^2))
end
# ============================================================================ #
bmean(x::AbstractVector{<:Real}, n::Integer=5000) = bmean(x, BasicSampling(n))
bmean(x::AbstractVector{<:Real}, s::Bootstrap.BootstrapSampling) = bootstrap(mean, x, s)
bmedian(x::AbstractVector{<:Real}, n::Integer=5000) = bmedian(x, BasicSampling(n))
bmedian(x::AbstractVector{<:Real}, s::Bootstrap.BootstrapSampling) = bootstrap(median, x, s)
# ============================================================================ #
bcor(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, n::Integer=5000) = bcor(x, y, BasicSampling(n))
bcor(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, s::Bootstrap.BootstrapSampling) = bootstrapn(cor, s, x, y)#bootstrap2(cor, x, y, s)#
# ============================================================================ #
bspearman(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, n::Integer=5000) = bspearman(x, y, BasicSampling(n))
bspearman(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, s::Bootstrap.BootstrapSampling) = bootstrapn(spearman, s, x, y)#bootstrap2(spearman, x, y, s)#
# ============================================================================ #
bcor_partial(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, z::AbstractVector{<:Real}, n::Integer=5000) = bcor_partial(x, y, z, BasicSampling(n))
bcor_partial(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, z::AbstractVector{<:Real}, s::Bootstrap.BootstrapSampling) = bootstrapn(cor_partial, s, x, y, z)
# ============================================================================ #
@enum TailType Both Left Right
# ---------------------------------------------------------------------------- #
pvalue(bs::Bootstrap.BootstrapSample, tail::TailType=Both) = pvalue(bs, 1, tail)
function pvalue(bs::Bootstrap.BootstrapSample, k::Integer, tail::TailType=Both)
    return calc_pvalue(bs.t0[k], bs.t1[k], tail)
end
Statistics.std(bs::Bootstrap.BootstrapSample, k::Integer=1) = Statistics.std(bs.t1[k])
# ============================================================================ #
# Bootstrap.bootstrap can handle functions with multiple outputs, but to make a
# multi-input function like cor work you need to pass the variables as columns
# of a matrix, this is a convienence wrapper for the N-argument case
function bootstrapn(f::Function, s::Bootstrap.BootstrapSampling, args::Vararg{T}) where T<:AbstractVector{<:Real}
    return bootstrap(hcat(args...), s) do data
        return f(eachcol(data)...)
    end
end
# ============================================================================ #
function tiedrank(x::AbstractVector)
    ks = sortperm(x)
    y = Vector{Float64}(x[ks])
    k = 1
    while k <= length(x)
        j = 1
        while j+k <= length(x) && y[k] == y[k+j]
            j += 1
        end
        j -= 1
        idx = k:k+j
        y[idx] .= mean(idx)
        k += j + 1
    end
    y[ks] = y
    return y
end
# ============================================================================ #
function permutation_test(f::Function, a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, n::Integer=5000; tail::TailType=Both)

    data = vcat(a, b)
    d = fill(NaN, n)
    ka = 1:length(a)
    kb = length(a)+1:length(data)
    for k = 1:n
        shuffle!(data)
        ta = view(data, ka)
        tb = view(data, kb)
        d[k] = f(ta) - f(tb)
    end

    stat = f(a) - f(b)

    return stat, calc_pvalue(stat, d, tail, true)
end
# ============================================================================ #
function paired_permutation_test(f::Function, a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, n::Integer=5000; tail::TailType=Both)
    np = length(a)
    @assert(np==length(b), "Lengths of inputs do NOT match!")

    x = [a b]
    dist = fill(NaN, n)
    for k in 1:n
        for j in 1:np
            # flip a coin, if heads swap column assignments for this row
            if rand() > 0.5
                x[j,1], x[j,2] = x[j,2], x[j,1]
            end
        end
        dist[k] = f(x[:,1] .- x[:,2])
    end

    stat = f(a .- b)

    return stat, calc_pvalue(stat, dist, tail, true)
end
# ============================================================================ #
function cor_permutation_test(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, n::Integer=5000; tail::TailType=Both, method::Symbol=:spearman)
    if method == :spearman
        fcor = spearman
    else
        fcor = cor
    end

    at = copy(a)

    dist = fill(NaN, n)
    for k in 1:n
        shuffle!(at)
        dist[k] = fcor(at, b)
    end

    stat = fcor(a, b)

    return stat, calc_pvalue(stat, dist, tail, true)
end
# ============================================================================ #
function cor_permutation_partial(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, z::AbstractVector{<:Real}, n::Integer=5000; tail::TailType=Both, method::Symbol=:spearman)

    at = copy(a)
    bt = copy(b)

    dist = fill(NaN, n)
    for k in 1:n
        shuffle!(at)
        shuffle!(bt)
        dist[k] = cor_partial(at, bt, z)
    end

    stat = cor_partial(a, b, z)

    return stat, calc_pvalue(stat, dist, tail, true)
end
# ============================================================================ #
# correction comes from: Phipson & Smyth 2010
function calc_pvalue(obs::Real, dist::Vector{<:Real}, tail::TailType, correction::Bool=false)
    n = length(dist) + correction
    if tail == Both
        p = (sum(x -> abs(x) >= abs(obs), dist) + correction) / n
    elseif tail == Left
        p = (sum(x -> obs >= x, dist) + correction) / n
    else
        p = (sum(x -> obs <= x, dist) + correction) / n
    end
    return p
end
# ============================================================================ #
end
