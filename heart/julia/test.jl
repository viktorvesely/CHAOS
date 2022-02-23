using LinearAlgebra
using Random, Distributions
using BenchmarkTools

function insert_one(vector)
    appended = ones(Float64, (length(vector) + 1, 1))
    appended[2:end] .= vector
    return appended
end


function append_insert_one(vectorA, vectorB)
    l1 = length(vectorA)
    l2 = length(vectorB)
    result = ones(Float64, (l1 + l2 + 1, 1))
    result[2:(l1 + 1)] .= l1
    result[(l1 + 2):end] .= l2 
    return result
end

function forward_passes(n_passes)

    n_input = 10
    n_output = 10
    n_reservior = 100
    n_readouts = n_input + n_reservior + 1
    w_in_scale = 0.8
    w_in_mu = 0.0
    w_scale = 0.8
    w_mu = 0.0
    spectral_radius = 0.95
    
    Random.seed!(150)

    w_in_d = Normal(w_in_mu, w_in_scale)
    w_d = Normal(w_mu, w_scale)

    w_in = rand(w_in_d, (n_reservior, n_input + 1))

    w = rand(w_d, (n_reservior, n_reservior))
    eigv = eigvals(w)
    sr = max(abs.(eigv)...)
    
    w = (w ./ sr) .* spectral_radius
    
    w_out = rand(Normal(0.0, 0.8), (n_output, n_readouts))

    x = zeros(Float64, (n_reservior, 1))
    train_state = zeros(Float64, (n_readouts, 1))

    us = rand(n_passes, n_input, 1) .- 0.5

    XX = zeros(Float64, (n_readouts, n_readouts))
    YX = zeros(Float64, (n_output, n_readouts))

    for i = 1:n_passes
        u = us[i,:,:]
        u_one = insert_one(u)

        x = tanh.(
            w_in * u_one .+ w * x
        )

        train_state = append_insert_one(u, x)
        train_state_T = transpose(train_state)

        yhat = w_out * train_state
        
        XX = XX .+ (train_state * train_state_T)
        YX = YX .+ (yhat * train_state_T)


    end

    return XX, YX

end


function test()
    @time XX, YX = forward_passes(10_000_000)
    print(XX[1])
    print("\n")
    print(YX[1])
    print("\n")
end

test()
