__precompile__(true)
module DE
using SpecialFunctions
using Base.Cartesian
	mutable struct Properties
		n::Int	# code length
		m::Int	# log2(n)
        q::Int
        x::Array{Float64, 1}
		qtable::Array{Array{Int, 1}, 1}
		function Properties(n::Int, q::Int, clipping::Float64)
			@assert n > 0
			@assert (n & (n - 1)) == 0 # power of 2
			@assert q % 2 == 1
			@assert q >= 3
			@assert clipping > 0
			m::Int = Int(log2(n))
			x::Array{Float64, 1} = -clipping : 2*clipping/(q-1) : clipping
			@assert length(x) == q
			qtable = Array{Array{Int, 1}, 1}(undef, q)
			@simd for i = 1:q
				qtable[i] = Array{Int, 1}(undef, i)
				@simd for j = 1:i
					qtable[i][j] = quantize(boxplus(x[i], x[j]), x)
				end
			end
			new(n, m, q, x, qtable)
		end
	end
	function JacLog(x::Float64)::Float64
    	if x > 10
        	return x;
    	elseif x < -10
        	return 0.0;
    	else
        	return log(1+exp(x));
		end
	end
	function boxplus(a::Float64, b::Float64)::Float64
		return sign(a) * sign(b) * min( abs(a), abs(b) ) + JacLog(-abs(a+b)) - JacLog(-abs(a-b))
	end
	function quantize(LLR::Float64, x::Array{Float64, 1})::Int
		distx = abs.(x .- LLR)
		ix = findall(distx .== minimum(distx))
		if (length(ix) > 1)
    		if (LLR > 0)
        		return maximum(ix);
    		else
        		return minimum(ix);
    		end
		else
			return ix[1]
		end
	end
	function biAWGN_LLRpdf(mu::Float64, sigma2::Float64, p::Properties)::Array{Float64, 1}
		chpdf = Array{Float64, 1}(undef, p.q)
		step::Float64 = p.x[2]-p.x[1];
		#d = Distributions.Normal(mu, sqrt(sigma2))
		@simd for i = 1:p.q
			l = p.x[i] - step/2
			r = p.x[i] + step/2
			if i == 1
				l = -Inf
			end
			if i == p.q
				r = Inf
			end
			chpdf[i] = abs( GausCDF(l,mu,sigma2) - GausCDF(r,mu,sigma2) )
		end
		return chpdf
	end
	function GausCDF(x::Float64, mu::Float64, sigma2::Float64)::Float64
		return 0.5*( 1 + erf( (x-mu)/sqrt(2*sigma2) ) )
	end
	function VN(pdf1::Array{Float64, 1}, pdf2::Array{Float64, 1}, p::Properties)::Array{Float64, 1}
		@assert length(pdf1) == p.q
		@assert length(pdf2) == p.q
		temp::Array{Float64, 1} = fastconv(pdf1, pdf2)
		l_tail::Int = (p.q - 1)>>1
		pdf_out::Array{Float64, 1} = temp[l_tail+1:l_tail+p.q]
		pdf_out[1] += sum(temp[1:l_tail])
		pdf_out[p.q] += sum(temp[l_tail+p.q+1:2*p.q-1])
		return pdf_out./sum(pdf_out)
	end
	function CN(pdf1::Array{Float64, 1}, pdf2::Array{Float64, 1}, p::Properties)::Array{Float64, 1}
		@assert length(pdf1) == p.q
		@assert length(pdf2) == p.q
		pdf_out = zeros(Float64, p.q)
		@simd for i = 1:p.q
			@simd for j = 1:p.q
				k = readCNLUT(i,j,p)
				pdf_out[k] += pdf1[i]*pdf2[j]
			end
		end
		return pdf_out./sum(pdf_out)
	end
	function readCNLUT(a::Int, b::Int, p::Properties)::Int
		if a < b
			return p.qtable[b][a]
		else
			return p.qtable[a][b]
		end
	end
	function performDE(chpdf::Array{Float64, 2}, p::Properties)::Array{Float64, 2}
		@assert size(chpdf, 1) == p.q
		@assert size(chpdf, 2) == p.n
		outpdf = copy(chpdf)
		for lambda = 0:p.m-1
			s::Int = 1 << lambda
			e::Int = 1 << (p.m-lambda-1)
			for j = 0:s-1
				@simd for i = 1:e
					outpdf[:, j*e*2+i] = CN(chpdf[:, j*e*2+i], chpdf[:, j*e*2+i+e], p)
					outpdf[:, j*e*2+i+e] = VN(chpdf[:, j*e*2+i], chpdf[:, j*e*2+i+e], p)
				end
			end
			copyto!(chpdf, outpdf)
		end
		return outpdf
	end
	@generated function fastconv(E::Array{T,N}, k::Array{T,N}) where {T,N}
	    quote
	        retsize = [size(E)...] + [size(k)...] .- 1
	        retsize = tuple(retsize...)
	        ret = zeros(T, retsize)
	        convn!(ret,E,k) #https://arxiv.org/abs/1612.08825
	        return ret
	    end
	end
	@generated function convn!(out::Array{T}, E::Array{T,N}, k::Array{T,N}) where {T,N}
	    quote
	        @inbounds begin
	            @nloops $N x E begin
	                @nloops $N i k begin
	                    (@nref $N out d->(x_d + i_d - 1)) += (@nref $N E x) * (@nref $N k i)
	                end
	            end
	        end
	        return out
	    end
	end
	function biAWGN(EsN0dB::Float64, p::Properties)::Array{Float64, 1}
		middle::Int = (p.q+1)>>1
		sigma2 = 10^(-EsN0dB/10)
		chpdfin = biAWGN_LLRpdf(2/sigma2, 4/sigma2, p)
		chpdf = Array{Float64, 2}(undef, p.q, p.n)
		BER = Array{Float64, 1}(undef, p.n)
		@simd for i = 1:p.n
			chpdf[:, i] = chpdfin
		end
		outpdf = performDE(chpdf, p)
		@simd for i = 1:p.n
			BER[i] = sum(outpdf[1:middle-1,i]) + 0.5*outpdf[middle,i]
		end
		return BER
	end
end
