module SimpleFitting

using Optim

export line_fit, exp_fit

function line_fit(x::Vector{<:Real}, y::Vector{<:Real})
    sx = sum(x)
    sy = sum(y)

    m = length(x)

    sx2 = zero(sx)
    sy2 = zero(sy)
    sxy = zero(sx*sy)

    @inbounds for k = 1:m
        sx2 += x[k]*x[k]
        sy2 += y[k]*y[k]
        sxy += x[k]*y[k]
    end

    a0 = (sx2*sy - sxy*sx) / ( m*sx2 - sx*sx )
    a1 = (m*sxy - sx*sy) / (m*sx2 - sx*sx)

    return a0, a1
end

function exp_fit(x::Vector{<:Real}, y::Vector{<:Real}, tau::Real=-1.0)

    if isnan(tau) || tau < 0
        ooe = 1.0 / exp(1.0)
        mn, mx = extrema(y)
        kt = findfirst(x -> x <= (mx - mn) * ooe, y)
        if kt != nothing
            tau = x[kt] - x[1]
        else
            tau = (x[end] - x[1]) / 2.0
        end
    end

    p0 = [y[1], tau, y[end]]

    objective(p::Vector) = begin
        yf = p[1] .* exp.(-x ./ p[2]) .+ p[3]
        return sum(abs2, yf - y)
    end

    res = optimize(objective, p0, LBFGS(); autodiff = :forward)

    p = Optim.minimizer(res)

    #      max   tau   baseline
    return p[1], p[2], p[3]
end

end
