__precompile__(true)
module polar
	mutable struct Properties
		n::Int	# code length
		m::Int	# log2(n)
		k::Int	# code dimension
		maxN_nodes::Int
		frozen_pattern::Array{Bool,1}
		dfrozen_pattern::Array{Bool,1}
		dfCons::Dict{Int64,Array{Int64,1}}
		rho::Array{Float64,1}
		# Datastructures of SC-Decoder
		L::Array{Array{Float64, 1}, 1}
		C::Array{Array{UInt8, 1}, 1}
		perm_bitrev::Array{Int, 1}
		v::Array{UInt8, 1}
		u_hat::Array{UInt8, 1}
		M::Array{Float64, 1}
		M_cml::Float64
		MM::Array{Float64, 1}
		SS::Array{Float64, 1}
		flipList::Array{Array{Int, 1}, 1}
		flipM::Array{Float64, 1}
		flipS::Array{Float64, 1}
		N_node::Int
		last_flip_set::Array{Int, 1}
		eta::Int
		function Properties(n::Int, k::Int, maxN_nodes::Int, frozen_pattern::Array{Bool,1}, dfCons::Dict{Int64,Array{Int64,1}}, Pe::Array{Float64,1})
			m::Int = Int(log2(n))
			@assert n > 0
			@assert (n & (n - 1)) == 0 # power of 2
			@assert length(frozen_pattern) == n
			@assert k > 0
			@assert maxN_nodes >= n
			@assert n == k + sum(frozen_pattern)
			@assert length(Pe) == n
			#init datastructures
			L				= Array{Array{Float64, 1}, 1}(undef, m+1)
			C				= Array{Array{UInt8, 1}, 1}(undef, m+1)
			perm_bitrev 			= bitrev(collect(1:n))
			dfrozen_pattern 		= zeros(Bool, n)
			v	 					= Array{UInt8, 1}(undef, n)
			u_hat 					= Array{UInt8, 1}(undef, n)
			M						= Array{Float64, 1}(undef, n)
			rho			= Array{Float64, 1}(undef, n)
			flipList    			= Array{Array{Int, 1}, 1}(undef, 0)
			flipM  			= Array{Float64, 1}(undef, 0)
			flipS			= Array{Float64, 1}(undef, 0)
			MM  			= Array{Float64, 1}(undef, n)
			SS				= Array{Float64, 1}(undef, n)
			last_flip_set			= Array{Int, 1}(undef, 0)
			@simd for lambda = 1:m+1
				L[lambda] = zeros(Float64, n)	# float
				C[lambda] = Array{UInt8, 1}(undef, n)
			end
			for i in keys(dfCons)
				@assert i > maximum(dfCons[i]) # check dynamic constraints
				@assert frozen_pattern[i] == true
				dfrozen_pattern[i] = true
			end
			rho[1] = log(1 - Pe[1])
			for i = 2:n
				rho[i] = rho[i-1] + log(1 - Pe[i])
			end
			M_cml = Inf
			N_node = 0
			eta = Int(m) #list size = logN
			new(n, m, k, maxN_nodes, frozen_pattern, dfrozen_pattern, dfCons, rho, L, C, perm_bitrev, v, u_hat, M, M_cml, MM, SS, flipList, flipM, flipS, N_node, last_flip_set, eta)
		end
	end
	function recursively_calc_L(lambda::Int, phi::Int, p::Properties) # lambda from 1 to m+1
		if lambda == 1
			return
		end
		psi::Int = phi >> 1 #Int(floor(phi / 2))
		if phi % 2 == 0
			recursively_calc_L(lambda-1, psi, p) # recurse (values haven't been computed yet)
		end
		l1::Int = 1<<(lambda-1)
		l2::Int = 1<<(lambda-2)
		@simd for beta = 0:(1<<(p.m-lambda+1))-1
			if phi % 2 == 0 # case 1, equation (4) [TaVa15]
				p.L[lambda][phi + l1*beta + 1] = f_minus(p.L[lambda-1][psi + l2*2*beta + 1], p.L[lambda-1][psi + l2*(2*beta+1) + 1])
			else # case 2, equation (5) [TaVa15]
				p.L[lambda][phi + l1*beta + 1] = f_plus( p.L[lambda-1][psi + l2*2*beta + 1], p.L[lambda-1][psi + l2*(2*beta+1) + 1], p.C[lambda][phi+l1*beta])
			end
		end
		return nothing
	end
	function recursively_update_C(lambda::Int, phi::Int, p::Properties)
		psi::Int = phi >> 1 # Int(floor(phi / 2))
		l1::Int = 1<<(lambda-1)
		l2::Int = 1<<(lambda-2)
		@simd for beta = 0:(1<<(p.m-lambda+1)) - 1
			p.C[lambda-1][psi + l2*2*beta + 1] = p.C[lambda][phi + l1*beta] ⊻ p.C[lambda][phi + l1*beta + 1]
			p.C[lambda-1][psi + l2*(2*beta+1) + 1] = p.C[lambda][phi + l1*beta + 1]
		end
		if psi % 2 == 1
			recursively_update_C(lambda-1, psi, p)
		end
		return nothing
	end
	function HardDec(llr::Float64)
		if llr > 0.0
			return 0x00
		else
			return 0x01
		end
	end
	function f_plus(a::Float64, b::Float64, u::UInt8)::Float64
		if u == 0x00
			return a + b
		else
			return b - a
		end
	end
	function f_minus(a::Float64, b::Float64)::Float64
		return sign(a) * sign(b) * min( abs(a), abs(b) )
	end
	function calc_PM(PM::Float64, llr::Float64, v::UInt8)
		if v == HardDec(llr) #approximation
			return PM
		else
			return PM + abs(llr)
		end
	end
	function InitProperties!(p::Properties)
		p.N_node = 0
		p.M_cml = Inf
		while !isempty(p.flipList)
			pop!(p.flipList)
			pop!(p.flipM)
			pop!(p.flipS)
		end
		while !isempty(p.last_flip_set)
			pop!(p.last_flip_set)
		end
	end
	function find_startIndex(flip_set::Array{Int, 1}, p::Properties)
		for i = 1:p.n
			if (i in flip_set) ⊻ (i in p.last_flip_set) == true
				return i
			end
		end
		print(" Error! Same Flip Set ! \n")
	end
	function SCDec!(flip_set::Array{Int, 1}, startIndex::Int, p::Properties)::Int
		for phi = (startIndex-1):p.n-1
			p.N_node += 1
			recursively_calc_L(p.m+1, phi, p)
			if p.frozen_pattern[phi+1] # frozen
				if p.dfrozen_pattern[phi+1] # dynamic frozen
					p.v[phi+1] = 0x00
					for jj = 1:length(p.dfCons[phi+1])
						p.v[phi+1] ⊻= p.v[ p.dfCons[phi+1][jj] ]
					end
				else # static frozen
					p.v[phi+1] = 0x00
				end
			else # data
				if (phi+1) in flip_set
					p.v[phi+1] = HardDec(p.L[p.m+1][phi+1]) ⊻ 0x01
				else
					p.v[phi+1] = HardDec(p.L[p.m+1][phi+1])
				end
				if isempty(flip_set)
					p.MM[phi+1] = calc_PM(p.M[phi], p.L[p.m+1][phi+1], p.v[phi+1] ⊻ 0x01)
					p.SS[phi+1] = p.MM[phi+1] + p.rho[phi+1]
				elseif phi+1 > flip_set[end]
					p.MM[phi+1] = calc_PM(p.M[phi], p.L[p.m+1][phi+1], p.v[phi+1] ⊻ 0x01)
					p.SS[phi+1] = p.MM[phi+1] + p.rho[phi+1]
				end
			end

			if phi == 0
				p.M[1] = calc_PM(0.0, p.L[p.m+1][1], p.v[1])
			else
				p.M[phi+1] = calc_PM(p.M[phi], p.L[p.m+1][phi+1], p.v[phi+1])
			end

			if p.M[phi+1] > p.M_cml
				return phi+1
			end
			p.C[p.m+1][phi+1] = p.v[phi+1]
			if phi % 2 == 1
				recursively_update_C(p.m+1, phi, p)
			end

		end
		if p.M[p.n] < p.M_cml
			p.M_cml = p.M[p.n]
			@simd for i = 1:p.n
				p.u_hat[i] = p.v[i]
			end
			return p.n
		end
	end
	function SCOS(channel_LLRs::Array{Float64,1}, p::Properties)
		@assert length(channel_LLRs) == p.n
		InitProperties!(p)
		@simd for i = 1:p.n
			p.L[1][i] = channel_LLRs[ p.perm_bitrev[i] ];
		end
		SCDec!(Int[], 1, p)
		for i = 1:p.n
			if ( (!p.frozen_pattern[i]) && (p.MM[i] < p.M_cml) )
				InsertList([i], i, p)
			end
		end
		while( ( p.N_node < p.maxN_nodes ) && ( !isempty(p.flipM) ) )
			flip_set = popfirst!(p.flipList)
			popfirst!(p.flipS)
			pm = popfirst!(p.flipM)
			if pm < p.M_cml
				startIndex = find_startIndex(flip_set, p)
				endIndex = SCDec!(flip_set, startIndex, p)
				for i = (flip_set[end]+1):endIndex
					if ( (p.MM[i] < p.M_cml) && (!p.frozen_pattern[i]) )
						InsertList([flip_set;i], i, p)
					end
				end
				p.last_flip_set = flip_set
			end
		end
		return nothing
	end
	function InsertList(set::Array{Int,1}, index::Int64, p::Properties)
		if isempty(p.flipM)
			push!(p.flipList, set)
			push!(p.flipM, p.MM[index])
			push!(p.flipS, p.SS[index])
		else
			i::Int = length(p.flipM) + 1
			while ( (i > 1) && (p.SS[index] < p.flipS[i-1]) )
				i -= 1
			end
			splice!(p.flipList, i:(i-1), [set])
			splice!(p.flipM, i:(i-1), p.MM[index])
			splice!(p.flipS, i:(i-1), p.SS[index])
		end
		if length(p.flipM) > p.eta
			pop!(p.flipList)
			pop!(p.flipM)
			pop!(p.flipS)
		end
		return nothing
	end
	function enc!(w::Array{UInt8,1}, c::Array{UInt8,1}, p::Properties)
		@assert length(w) == p.k
		j = 1
		for i = 1:p.n
			if p.frozen_pattern[i] == false
				c[i] = w[j]
				j += 1
			else
				c[i] = 0x00
			end
		end
		@assert j == p.k + 1

		for i in keys(p.dfCons)
			for j = 1:length(p.dfCons[i])
				c[i] ⊻= c[ p.dfCons[i][j] ]
			end
		end
		transform!(c, p)
		return nothing
	end
	function transform!(c::Array{UInt8,1}, p::Properties)
		@assert length(c) == p.n
		for lambda = 0:p.m-1
			s::Int = 1 << lambda
			e::Int = 1 << (p.m-lambda-1)
			for j = 0:s-1
				@simd for i = 1:e
					c[j*e*2+i] ⊻= c[j*e*2+i+e]
				end
			end
		end
		return nothing
	end
	function bitrev(u)
		if length(u) == 2
			u_rev = u
		else
			u_rev = [bitrev(u[1:2:end]); bitrev(u[2:2:end])]
		end
		return u_rev
	end
end
