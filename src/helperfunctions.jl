
function flip(M,wf,L,K,N)
    # Function that flips the wavelet coefficients
    idwt(reshape(reshape(dwt(M,wf,L),1,K)[:,end:-1:1],N,N),wf,L)
end


function huber(x::Float64,mu::Float64)
    return(abs(x) <= mu ? 1/2*x^2 : mu*(abs(x)-1/2*mu))
end



function T(x,μ::Float64)
    return(x./(max.(abs.(x),μ)))
end


function Nesta(y,z₀,mask,A,At,niter,μ::Float64=0.2, η::Float64=1.,c::Number=1.)
    # This is an implementation of the Nesta algortihm for l1 minimisation.
    # z₀ = At(y);
    # Ensure that A is normalised
    q₂ = copy(z₀);
    z = copy(z₀);
    for i = 1:niter 
        Tₙ = μ * A(T(At(z),μ));
        q = z - Tₙ;
        λ = max(0, η^(-1) * norm(y - √c * mask.*q) -1 );
        xₙ = ( λ / √c * mask.*y + q ) - λ/((λ + 1)*c)* (λ / √c * mask.*y+mask.*q ) ;
        q₂ .-= (i+1)/2 * Tₙ;
        λ = max(0, η^(-1) * norm(y - √c * mask.*q₂) -1 );
        vₙ = ( λ / √c * mask.*y + q₂ ) - λ/((λ + 1)*c)* (λ / √c * mask.*y+mask.*q₂ ) ;
        z = 2/(i+3)  * vₙ + (1 - 2/(i+3) ) * xₙ;
    end
    return(z)
end

function Nesta_Cont(y,z₀,mask,A,At,niter,μ::Float64=0.2, η::Float64=1.,c::Number=1.,Cont ::Number = 1)
    p = Progress(Cont, dt=0.5,desc = "l1 - minimisation...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
    for i = 1:Cont
        z₀ = Nesta(y,z₀,mask,A,At,niter,μ^i, η,c)
        next!(p)
    end
    return z₀
end

function generate_l2_dist(W,wf,L,N,f,lines_and_blocks)
    # W .... 2D Weight matrix of Wavelet Basis
    # wf ... Wavlet filter
    # L .... Maximum Wavelet levels
    # N .... Dimension
    # 
    # Returns the optimal subsampling distribution in the l2 sense
    # 2024 Simon Ruetz
    
    # Precompute 1D Wavelet Basis and Fourier transform of wavelet basis 
    Wav_Bas = zeros(L,N,N);
    current = zeros(N);
    for j = 1:L
        for i = 1:N
            current[i] = 1;
            Wav_Bas[j,:,i] =idwt(current,wf,j);
            current[i] = 0;
        end
    end
   
    F_Wav = f(Wav_Bas,2)/sqrt(N)
    F_Wav_sep = f(Wav_Bas[L,:,:],1)/sqrt(N)

    # Preallocate memory for speedup. Multithreaded
    Q = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
    Q_current = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
    Q_adapted = [zeros(N,N) for i = 1:Threads.nthreads()];
    current = [zeros(size(W)) for i = 1:Threads.nthreads()];
    n1 = ones(Int64,1,Threads.nthreads());
    n2 = ones(Int64,1,Threads.nthreads());
    lvl = ones(Int64,1,Threads.nthreads());
    K = N^2
    p = Progress(sum(sum(W.>=0)), dt=0.5,desc="Generating sampling distribution...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);

    # Iterate over all entries of the weight matrix W
    Threads.@threads for j in findall(W.>=0)
        i = Threads.threadid();
        # calculate levels of different wavelets for j_1,j_2
        n1[i] = Int(L-ceil(log2(j[1]))+1);
        n2[i] = Int(L-ceil(log2(j[2]))+1);
        lvl[i] = min(n1[i],n2[i],L);
        
        Q_current[i] = F_Wav[ lvl[i] ,:,j[1]]*F_Wav[lvl[i] ,:,j[2]]'; #Take the correct wavelets out of the precomputet matrix
        Q[i] .+= abs.(Q_current[i].^2) .*W[j].^2;
        Q_adapted[i] .= max.(Q_adapted[i] , abs.(Q_current[i].^2)) ;
        next!(p)
    end  
    Q = abs.(sum(Q)); 
    Q_max = zeros(N,N);
    for i = 1:Threads.nthreads()
        Q_max = max.(Q_max,Q_adapted[i])
    end
    
    K = N^2
    if lines_and_blocks ==1
        sampling_probabilities_lines = zeros(N)
        
        for ii = 1:N
            U = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
            Q_current = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
            Threads.@threads for j in findall(W.>0)
                i = Threads.threadid();
                # calculate levels of different wavelets for j_1,j_2
                n1[i] = Int(L-ceil(log2(j[1]))+1);
                n2[i] = Int(L-ceil(log2(j[2]))+1);
                lvl[i] = min(n1[i],n2[i],L);
                
                @views Q_current[i] = F_Wav_sep[ :,j[1]]*transpose(F_Wav_sep[ii,j[2]]) * (F_Wav_sep[:,j[1]]*transpose(F_Wav_sep[ii,j[2]]))' ; #Take the correct wavelets out of the precomputet matrix
                U[i] .+= Q_current[i].*W[j].^2
                #@inline Ut[i][:,(j[2]-1)*N+j[1]] = reshape(Q_current[i]',K,1)
                next!(p)
            end
            U = sum(U)
            sampling_probabilities_lines[ii] =opnorm(U)
        end

        sampling_probabilities_lines = sampling_probabilities_lines/sum(sampling_probabilities_lines)
        S2 = Int64(round(sqrt(N)))
        S3 = Int64(round(N/S2))
        
        sampling_probabilities_blocks = zeros(S3,S3)
        for ii1 = 1:S3
            for ii2 = 1:S3
                U = [complex(zeros(S2^2,S2^2)) for i = 1:Threads.nthreads()];
                Q_current = [complex(zeros(S2^2,S2^2)) for i = 1:Threads.nthreads()];
                Threads.@threads for j in findall(W.>0)
                    i = Threads.threadid();
                    # calculate levels of different wavelets for j_1,j_2
                    n1[i] = Int(L-ceil(log2(j[1]))+1);
                    n2[i] = Int(L-ceil(log2(j[2]))+1);
                    lvl[i] = min(n1[i],n2[i],L);
                    
                    @views Q_current[i] = reshape(F_Wav_sep[((ii1-1)*S2 + 1) : ii1*S2,j[1]]*transpose(F_Wav_sep[((ii2-1)*S2 + 1) : ii2*S2 ,j[2]]),S2^2,1) * (reshape(F_Wav_sep[((ii1-1)*S2 +1 ): ii1*S2,j[1]]* transpose(F_Wav_sep[((ii2-1)*S2 +1): ii2*S2 ,j[2]]),S2^2,1))' ; #Take the correct wavelets out of the precomputet matrix
                    U[i] .+= Q_current[i].*W[j].^2
                    next!(p)
                end
                U = sum(U)
                sampling_probabilities_blocks[ii1,ii2] =opnorm(U)
            end
            println(ii1)
        end

        sampling_probabilities_blocks = sampling_probabilities_blocks/sum(sampling_probabilities_blocks)
    else
        sampling_probabilities_lines = zeros(N)
        sampling_probabilities_blocks = zeros(N)
    end

    # Combine the Threads back
    Q_max = abs.(Q_max);
    Q_adapted = max.(Q,Q_max)
   
    Q_max = Q_max./sum(sum(Q_max));
    Q_adapted = Q_adapted./sum(Q_adapted)

    adapted_distribution = zeros(N,N);
    adapted_distribution = Q_adapted;
    adapted_distribution = adapted_distribution./sum(sum(adapted_distribution));

    max_coherence_distribution = zeros(N,N);
    max_coherence_distribution = Q_max;
    max_coherence_distribution = max_coherence_distribution./sum(sum(max_coherence_distribution));    


    return adapted_distribution, max_coherence_distribution, sampling_probabilities_lines, sampling_probabilities_blocks
end

function process_image(path_vec::String,wf,L,N::Int64,th,flp,dwts)
    # Function that processes images and generates the wavelet distribution
    # It loads 
    # path_vec ... Path to images
    # wf ... Wavlet Filter
    # L ... Maximum Level of Wavelet Basis
    # N ... size of images
    # th ... threshold of wavelet coefficients

    # 2024 Simon Ruetz

    coeffs = [zeros(N,N) for i in 1:Threads.nthreads()];
    list = readdir(path_vec)[2:end]
    img = [zeros(N,N) for i in 1:Threads.nthreads()]
    M = [0 for i in 1:Threads.nthreads() ]
    C = [zeros(N,N) for i in 1:Threads.nthreads()]
    p = Progress(length(list), dt=0.5,desc="Loading Images and generating distribution...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
    if occursin("Net",path_vec)
        # @infiltrate
        Threads.@threads for j in eachindex(list)
                i = Threads.threadid()
                M[i] = size(load(joinpath(path_vec,list[j])))[1]
                for l = 1:M[i]
                    img[i] = convert(Array{Float64},imresize( Gray.(convert(Matrix{Float64},load(joinpath(path_vec,list[j]))[l,:,:])),(N,N)))
                    img[i] =img[i]./norm(img[i]);
                    if flp == 1
                        C[i] = dwts(flip(img[i],wf,L,N^2,N),wf,L);
                    else
                        C[i] = dwts(img[i],wf,L);
                    end
                    coeffs[i] +=  (abs.(C[i]) .>= th*norm(C[i]));#sort(vec(abs.(C[i])))[end-165];#
                end
                next!(p)
        end
    else
        Threads.@threads for j in eachindex(list)
            i = Threads.threadid()
            img[i] = imresize( Gray.(load(joinpath(path_vec,list[j]))),(N,N))
            img[i] = convert(Matrix{Float64},img[i])
            img[i] = img[i]./norm(img[i]);
            if flp == 1
                C[i] = dwts(flip(img[i],wf,L,N^2,N),wf,L);
            else
                C[i] = dwts(img[i],wf,L);
            end
            coeffs[i] += abs.(C[i]) .>= th*norm(C[i]);#sort(vec(abs.(C[i])))[end-165];#
            next!(p)
    end
    end

    println("Generating Wavelet distribution complete")
    
    coeff = sum(coeffs);
    Nmax = length(readdir(path_vec)[2:end]);
    S_av = round(sum(sum(coeff)/Nmax))
    W = sqrt.(coeff./Nmax);
    W = coeff./Nmax;
    return coeff, Nmax, S_av, W
end


function calculate_norm(W,wf,L,N,f,lines_and_blocks,ind)
    # W .... 2D Weight matrix of Wavelet Basis
    # wf ... Wavlet filter
    # L .... Maximum Wavelet levels
    # N .... Dimension
    # 
    # Returns the optimal subsampling distribution in the l2 sense
    # 2022 Simon Ruetz
    

    # Precompute 1D Wavelet Basis and Fourier transform of wavelet basis 
    Wav_Bas = zeros(L,N,N);
    current = zeros(N);
    for j = 1:L
        for i = 1:N
            current[i] = 1;
            Wav_Bas[j,:,i] =idwt(current,wf,j);
            current[i] = 0;
        end
    end
    
    F_Wav = f(Wav_Bas,2)/sqrt(N)#[:,1:round(Int,N/2+1),:];
    #F_Wav = fftshift(F_Wav,2)
    # Preallocate memory for speedup. Multithreaded
    Q = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
    Q_current = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
    Q_est = [zeros(N,N) for i = 1:Threads.nthreads()];
    current = [zeros(size(W)) for i = 1:Threads.nthreads()];
    n1 = ones(Int64,1,Threads.nthreads());
    n2 = ones(Int64,1,Threads.nthreads());
    lvl = ones(Int64,1,Threads.nthreads());
    K = N^2
    #U = complex(zeros(K,K))
    p = Progress(sum(sum(W.>=0)), dt=0.5,desc="Calculating norm...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);

    K = N^2
    S2 = Int64(round(1))
    S3 = Int64(round(N^2))

    dt = zeros(length(ind),S3)
    iii = 1
    
    for ii1 = 1:N
        for ii2 = 1:N
            a = (ii1-1)*N + ii2
            if issubset(a,ind)
                U = [complex(zeros(length(ind),S3)) for i = 1:Threads.nthreads()];
                Q_current = [complex(zeros(1,1)) for i = 1:Threads.nthreads()];
                Threads.@threads for j in findall(W.>0)
                    i = Threads.threadid();
                    # calculate levels of different wavelets for j_1,j_2
                    n1[i] = Int(L-ceil(log2(j[1]))+1);
                    n2[i] = Int(L-ceil(log2(j[2]))+1);
                    lvl[i] = min(n1[i],n2[i],L);
                    
                    #@views Q_current[i] =  ; #Take the correct wavelets out of the precomputet matrix
                    @inline U[i][iii,(j[2]-1)*N+j[1]] = F_Wav[ lvl[i],ii1,j[1]]*adjoint(F_Wav[lvl[i],ii2,j[2]]).*W[j]
                    #@inline Ut[i][:,(j[2]-1)*N+j[1]] = reshape(Q_current[i]',K,1)
                    next!(p)
                end
                U = sum(U)
                dt += U
                #dt[ii1,ii2] = maximum(abs.(Q[((ii1-1)*S2 + 1) : ii1*S2,((ii2-1)*S2 + 1) : ii2*S2]))
                iii += 1
                
                #println(N)
            end

        end
        
    end
    println(iii)
    return opnorm(dt)
end
