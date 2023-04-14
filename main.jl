n = 128
k = 64

Constr = "rm"

# maximum decoding complexity
λ_max = 20
maxN_nodes = λ_max*n #typemax(Int64)
eta = 10 # list size

#SNR in Es/N0

startSNRdB = 1.0
SNRstep = 0.25
target_BLER = 1e-4
min_errors = 500
max_blocks = Int(1e7 * min_errors)

if Constr == "beta"
	include("polarConstr_beta.jl")
	frz = polarConstr_beta(n, k)
elseif Constr == "rm"
	include("polarConstr_rm.jl")
	frz = polarConstr_rm(n, k)
end

# dymanic constraints
dfCons = Dict(  2=> [1]) # first two bits are usually frozen -> use if no dynamic frozen bits

@assert n - k == sum(frz)
num_frozen = n - k
frozen_pattern = Array{Bool}(undef, n)
@simd for j = 1:n
	if frz[j] == 1
		frozen_pattern[j] = true
	else
		frozen_pattern[j] = false
	end
end

# monte-carlo
using Random
using Printf
include("polar.jl")
include("DE.jl")

function sim_pc(startSNRdB::Float64, SNRstep::Float64, min_errors::Int, target_BLER::Float64, max_blocks::Int, n::Int, k::Int, maxN_nodes::Int, eta::Int, frozen_pattern::Array{Bool,1}, dfCons::Dict{Int64,Array{Int64,1}})
	BLER = 1.0
	SNRdB = startSNRdB - SNRstep

	while BLER > target_BLER
		@time begin
		SNRdB += SNRstep
		# Density Evolution
		levels = 255
		clipping = 25.0
		p = DE.Properties(n,levels,clipping)
		Pe = DE.biAWGN(SNRdB, p)
		#
		SNR = 10^(SNRdB/10)
		scale = sqrt(SNR)
		symbol_mapping = Dict(0x00 => scale, 0x01 => -scale)
		# pre-allocation
		p = polar.Properties(n, k, maxN_nodes, eta, frozen_pattern, dfCons, Pe) # polar coding functions
		w = Array{UInt8,1}(undef,k) # data bits
		c = Array{UInt8,1}(undef,n) # codeword
		z = Array{Float64,1}(undef,n) # noise
		y = Array{Float64,1}(undef,n) # received symbols
		w_hat = Array{UInt8,1}(undef,k) # estimate

		N_blocks = 0
		N_blockerrors = 0
		N_nodes = 0

		while N_blockerrors < min_errors && N_blocks < max_blocks
			N_blocks += 1
			# Source
			rand!(w, [0x00, 0x01])
			# Encoding
			polar.enc!(w, c, p)
			# biAWGN channel
			randn!(z)
			@simd for i = 1:n
				y[i] = symbol_mapping[c[i]] + z[i]
			end
			# Decoding
			polar.SCOS(2*scale*y, p)
			# Extract data bits
			j = 1
			for i = 1:n
				if p.frozen_pattern[i] == false
					w_hat[j] = p.u_hat[i]
					j += 1
				end
			end
			# Collect errors
			if w != w_hat
				N_blockerrors += 1
			end
			# complexity (number of node-visits)
			N_nodes += p.N_node
		end

		BLER = N_blockerrors/N_blocks
		normCompl = N_nodes/n/N_blocks

		@printf("\t%f\t%f\t%f", SNRdB, BLER, normCompl)
		@printf("\t");
		end
	end
end

@inbounds sim_pc(startSNRdB, SNRstep, min_errors, target_BLER, max_blocks, n, k, maxN_nodes, eta, frozen_pattern, dfCons)
