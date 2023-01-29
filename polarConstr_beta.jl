function polarConstr_beta(n::Int, k::Int)
    @assert (n & (n - 1)) == 0
    @assert n >= k
    m::Int = Int(log2(n))
    w = calc_weight(m)
    frz = zeros(Int,n)
    frz[sortperm(w)[1:n-k]] = ones(Int, n-k)
    return frz
end

function calc_weight(m::Int)
    N::Int = 1<<m
    w = zeros(Float64,N)
    beta = 2^0.25;
    for i = 1:N
        i_bit = bitstring(i - 1)
        b = i_bit[length(i_bit) : -1 : length(i_bit) - m + 1]
        for j = 1:m
            if b[j] == '1'
                w[i] = w[i] + beta^(j-1);
            end
        end
    end
    return w
end
