module MSequenceUtils

using ImageFiltering, Optim

export gaussian_fit, gaussian, ellipse, peakframe, clay_color

# ============================================================================ #
function peak(frame::AbstractMatrix{Float64})
    img = imfilter(frame, KernelFactors.IIRGaussian((2.0, 2.0)))
    mn, mx = extrema(img)
    return mx - mn
end
# ============================================================================ #
function peakframe(rf::Array{Float64,3})
    @assert(size(rf) == (16,16,16), "RF is not a valid size!")
    mx = 0.0
    kmx = 0
    for k in 1:size(rf, 3)
        tmx = peak(view(rf, :, :, k))
        if tmx > mx
            mx = tmx
            kmx = k
        end
    end
    return kmx
end
# ============================================================================ #
function clay_color()
    return open(joinpath(@__DIR__, "clay_color.bin"), "r") do io
        n = read(io, Int64)
        siz = Vector{Int64}(undef, n)
        read!(io, siz)
        data = Array{Float64,n}(undef, siz...)
        read!(io, data)
        return data
    end
end
# ============================================================================ #
function ellipse(p::Vector{<:Real}, length::Integer=100)
    t = range(0, 2pi, length=length)
    return p[4] .* cos.(t) .+ (p[2]-1), p[5] .* sin.(t) .+ (p[3]-1)
end
# ============================================================================ #
# p = [amplitude, center-x, center-y, sigma-x, sigma-y]
function gaussian(p::Vector{<:Real})
    yf = zeros(16, 16)
    for x = 1:16
        xt = (x - (p[2]))^2 / (p[4]^2)
        for y = 1:16
            yt = (y - (p[3]))^2 / (p[5]^2)
            yf[y,x] = p[1] * exp(-(xt + yt))
        end
    end
    return yf
end
# ============================================================================ #
function gaussian_fit(frame::Matrix{<:Real})

    mx, kmx = findmax(abs.(imfilter(frame, KernelFactors.IIRGaussian((2.0, 2.0)))))

    objective(p::Vector) = begin
        # p = [amplitude, center-x, center-y, sigma-x, sigma-y]
        return sum(abs2, frame .- gaussian(vcat(p, p[end])))
    end

    mn, mx = extrema(frame)
    if abs(mn) > mx
        amp = mn
    else
        amp = mx
    end
    p0 = [amp, kmx.I[2], kmx.I[1], 4.0]#, 4.0]

    res = optimize(objective, p0, LBFGS())

    return res.minimizer
end
# ============================================================================ #
end
