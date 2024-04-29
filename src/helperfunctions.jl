function run_tests(;
    K::Int64=2^18,
    lk::Float64=0.0,
    flp::Int64=0,
    had::Int64=0,
    lines_and_blocks::Int64=0,
    train_path::String=joinpath(pwd(), "images", "Brain4"),
    test_path::String=joinpath(pwd(), "images", "brain.png"),
    wf=wavelet(WT.db4, WT.Filter),
    R=20,
    sep=1
    )

    N = Int(sqrt(K))
    th = (lk * 0.02) * sqrt(2 * (log(2 * K) - log(1 / 2)) / K)
    iter = 40
    #wf = wavelet(WT.db4, WT.Filter);
    L = maxtransformlevels(N)#;#min(5,maxtransformlevels(N))

    function wav(x)
        dwt(x, wf, L)
    end

    function iwav(x)
        idwt(x, wf, L)
    end

    function idwts(x, wf, L)
        if sep == 1
            x = mapslices(iwav, x; dims=2)
            return mapslices(iwav, x; dims=1)
        else
            return idwt(x, wf, L)
        end
    end

    function dwts(x, wf, L)
        if sep == 1
            x = mapslices(wav, x; dims=1)
            return mapslices(wav, x; dims=2)
        else

            return dwt(x, wf, L)
        end
    end

    if had == 1
        f = fwht
        ft = ifwht
    else
        f = fft
        ft = ifft
    end

    A(x) = f(idwts(x, wf, L)) / N
    At(x) = dwts(ft(x) * N, wf, L)

    # Load, transform and threshold images form path to get proxy of wavelet distribution W
    coeff, Nmax, S_av, W = process_image(train_path, wf, L, N, th, flp, dwts)

    Dist1, Dist3 , sampling_probabilities_lines, sampling_probabilities_blocks = generate_l2_dist(W, wf, L, N, f, lines_and_blocks)

    Dist2 = zeros(N, N)
    if had == 0
        for i = 1:N
            for j = 1:N
                Dist2[i, j] = 1 / max(1, (i - N / 2)^2 + (j - N / 2)^2)^(2.6)
            end
        end
        Dist2 = (fftshift(Dist2))
    else
        for i = 1:N
            for j = 1:N
                Dist2[i, j] = 1 / max(1, (i)^2 + (j)^2)^(2.6)
            end
        end
    end
    Dist2 = Dist2 / sum(Dist2)


    # now sample from the distributions
    mask1 = zeros(N, N)
    mask2 = zeros(N, N)

    if lines_and_blocks == 1
        # lines:
        w1 = FrequencyWeights(vec(sampling_probabilities_lines[1:N]))
        ind1 = sample(1:N, w1, Int(round(N / R)); replace=false)
        mask1[ind1[ind1.<=N], :] .= 1
        mask1[:, ind1[ind1.>=N+1].-N] .= 1

        # blocks:
        w2 = FrequencyWeights(vec(sampling_probabilities_blocks))
        ind2 = sample(1:length(sampling_probabilities_blocks), w2, Int(round(N / R)); replace=false)
        for l in eachindex(sampling_probabilities_blocks)
            In = Int64(round(sqrt(N)))
            ll1 = mod(l - 1, Int64(round(N / In)))
            ll2 = (l - 1) ÷ Int64(round(N / In))
            if l in ind2
                mask2[ll1*In+1:Int64(round(ll1 * In + In)), ll2*In+1:Int64(round(ll2 * In + In))] .= 1
            end
            Dist2[ll1*In+1:Int64(round(ll1 * In + In)), ll2*In+1:Int64(round(ll2 * In + In))] .= sampling_probabilities_blocks[l]
        end
    else
        w1 = FrequencyWeights(vec(reshape(Dist1, 1, K)))
        ind1 = sample(1:K, w1, Int(round(K / R)); replace=false)
        mask1[ind1] .= 1
        w2 = FrequencyWeights(vec(reshape(Dist2, 1, K)))
        ind2 = sample(1:K, w2, Int(round(K / R)); replace=false)
        mask2[ind2] .= 1
    end

    # load the image on which we test the performance of the sampling algorithms
    Im = load(test_path)
    if length(size(Im)) > 2
        l = Int64(round(size(Im)[1] * 1 / 10))
        Im = convert(Array{Float64}, imresize(Gray.(convert(Matrix{Float64}, load(test_path)[l, :, :])), (N, N)))
    else
        Im = imresize(Gray.(Im)::Array{Gray{N0f8},2}, (N, N))
        Im = convert(Array{Float64}, Im)
    end
    Im = Im ./ norm(Im)
    if flp == 1
        Im = flip(Im, wf, L, K, N)
    end
    Im = Im ./ norm(Im)

    # l1 minimisation starts here:
    δ = 1.0e-7
    μ = 0.2
    y = mask1 .* f(Im) / N
    im_rec = Nesta_Cont(y, y, mask1, A, At, iter, μ, δ, 1.0, 10)
    im_rec = ft(im_rec) * N
    im_rec = im_rec / norm(im_rec)

    y = mask2 .* f(Im) / N
    im_rec2 = Nesta_Cont(y, y, mask2, A, At, iter, μ, δ, 1.0, 10)
    im_rec2 = ft(im_rec2) * N
    im_rec2 = im_rec2 / norm(im_rec2)
    if flp == 1
        im_rec = flip(im_rec, wf, L, K, N)
        im_rec2 = flip(im_rec2, wf, L, K, N)
        Im = flip(Im, wf, L, K, N)
    else

    end

    var1 = psnr(real.(im_rec), Im, maximum(abs.(Im)))
    var2 = psnr(real.(im_rec2), Im, maximum(abs.(Im)))

    Dist1[Dist1.<=exp(-60)] .= exp(-60)

    if lines_and_blocks == 1
        fig = Figure(size = (2000, 1800))
        CairoMakie.Axis(fig[1, 1])
        lines!(1:N, fftshift(log.(sampling_probabilities_lines)))
        #CairoMakie.Axis(fig[1,2])
        ax, hm = heatmap(fig[1, 2], fftshift(log.(Dist2)))
        ax.xticks = (1:64:N, string.(Int.(-N/2:64:N/2-1)))
        ax.yticks = (1:64:N, string.(Int.(-N/2:64:N/2-1)))
        Colorbar(fig[1, 3], hm)
        ax, hm = heatmap(fig[2, 1], fftshift(mask1)', colormap=:grays)
        ax.xticks = (1:64:N, string.(Int.(-N/2:64:N/2-1)))
        ax.yticks = (1:64:N, string.(Int.(-N/2:64:N/2-1)))
        ax, hm = heatmap(fig[2, 2], fftshift(mask2)', colormap=:grays)
        ax.xticks = (1:64:N, string.(Int.(-N/2:64:N/2-1)))
        ax.yticks = (1:64:N, string.(Int.(-N/2:64:N/2-1)))
        ax, hm = heatmap(fig[1, 4], W', colormap=:grays)
        Colorbar(fig[1, 5], hm)
        #ax, hm = heatmap(fig[2,3],abs.(Im)',colormap = :grays)
        ax, hm = heatmap(fig[3, 1], abs.(im_rec)', colormap=:grays)
        ax, hm = heatmap(fig[3, 2], abs.(im_rec2)', colormap=:grays)
        ax, hm = heatmap(fig[3, 4], abs.(Im)', colormap=:grays)
        display(fig)
    elseif flp == 1
        clims = (minimum([Dist1 Dist2]), maximum([Dist1 Dist2]))
        fig = Figure(size = (2000, 1800))
        ax, hm = contourf(fig[1, 1], fftshift(log.(Dist1')),  colorrange=log.(clims))
        ax.xticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax.yticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        Colorbar(fig[1, 2], hm)
        ax, hm = contourf(fig[1, 3], fftshift(log.(Dist2')), colorrange=log.(clims))
        ax.xticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax.yticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        Colorbar(fig[1, 4], hm)
        ax, hm = heatmap(fig[2, 1], fftshift(mask1'), colormap=:grays)
        ax.xticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax.yticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax, hm = heatmap(fig[2, 3], fftshift(mask2'), colormap=:grays)
        ax.xticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax.yticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax, hm = heatmap(fig[1, 5], W' .^ 2, colormap=:grays)
        Colorbar(fig[1, 6], hm)
        # ax, hm = heatmap(fig[2, 5], abs.(Im)', colormap=:grays)
        # plot the reconstructions in a new row
        ax, hm = heatmap(fig[3, 1], abs.(im_rec)', colormap=:grays)
        ax, hm = heatmap(fig[3, 3], abs.(im_rec2)', colormap=:grays)
        ax, hm = heatmap(fig[3, 5], abs.(Im)', colormap=:grays)
        display(fig)
    else
        clims = (minimum([Dist1 Dist2]), maximum([Dist1 Dist2]))
        fig = Figure(size = (1800, 1000))
        ax, hm = contourf(fig[1, 1], fftshift(log.(Dist1')),  colorrange=log.(clims))
        ax.xticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax.yticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        Colorbar(fig[1, 2], hm)
        ax, hm = contourf(fig[1, 3], fftshift(log.(Dist2')), colorrange=log.(clims))
        ax.xticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax.yticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        Colorbar(fig[1, 4], hm)
        ax, hm = heatmap(fig[2, 1], fftshift(mask1'), colormap=:grays)
        ax.xticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax.yticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax, hm = heatmap(fig[2, 3], fftshift(mask2'), colormap=:grays)
        ax.xticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax.yticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax, hm = heatmap(fig[1, 5], W' .^ 2, colormap=:grays)
        Colorbar(fig[1, 6], hm)
        ax, hm = heatmap(fig[2, 5], abs.(Im)', colormap=:grays)
        # plot the reconstructions in a new row
        # ax, hm = heatmap(fig[3, 1], abs.(im_rec)', colormap=:grays)
        # ax, hm = heatmap(fig[3, 3], abs.(im_rec2)', colormap=:grays)
        # ax, hm = heatmap(fig[3, 5], abs.(Im)', colormap=:grays)
        display(fig)
    end
    println("adapted variable density sub psnr:")
    println(var1)
    println("Comparison:")
    println(var2)

    return var1, var2
end


function uniform_test(;
    K::Int64=2^16, had::Int64=1, lines::Int64=0,
    wf=wavelet(WT.haar, WT.Filter), R=20)
    N = Int(sqrt(K))
    iter = 100
    L = maxtransformlevels(N)
    if had == 1
        f = fwht
        ft = ifwht
    else
        f = fft
        ft = ifft
    end

    S_av = sqrt(K) / 2
    W = ones(N, N) * sqrt(S_av / K)

    mask1 = zeros(N, N)
    mask2 = zeros(N, N)
    mask3 = zeros(N, N)
    Dist1, Dist3, ~ , ~  = generate_l2_dist(W, wf, L, N, f, lines)

    Dist2 = ones(N, N)
    Dist2 = Dist2 / sum(Dist2)

    A(x) = f(idwt(x, wf, L)) / N
    At(x) = dwt(ft(x) * N, wf, L)
    var1 = 0
    var2 = 0
    var3 = 0

    mask1 = zeros(N, N)
    mask2 = zeros(N, N)
    mask3 = zeros(N, N)
    w1 = FrequencyWeights(vec(reshape(Dist1, 1, K)))
    ind1 = sample(1:K, w1, Int(round(K / R)); replace=false)
    mask1[ind1] .= 1
    w2 = FrequencyWeights(vec(reshape(Dist2, 1, K)))
    ind2 = sample(1:K, w2, Int(round(K / R)); replace=false)
    mask2[ind2] .= 1
    w3 = FrequencyWeights(vec(reshape(Dist3, 1, K)))
    ind3 = sample(1:K, w3, Int(round(K / R)); replace=false)
    mask3[ind3] .= 1

    # Generate test image:
    Im = zeros(N, N)
    Cff = dwt(Im, wf, L)
    Cff .= 0
    New = sample(1:K, Int64(round(S_av)))
    Cff[New] .= sign.(randn(length(New)))
    Im = idwt(Cff, wf, L)
    Im = Im / norm(Im)

    δ = 1.0e-7
    μ = 0.2

    y = mask1 .* f(Im) / N
    im_rec = Nesta_Cont(y, y, mask1, A, At, iter, μ, δ, 1.0, 10)
    im_rec = real.(ft(im_rec) * N)
    im_rec = im_rec / norm(im_rec)

    y = mask2 .* f(Im) / N
    im_rec2 = Nesta_Cont(y, y, mask2, A, At, iter, μ, δ, 1.0, 10)
    im_rec2 = real.(ft(im_rec2) * N)
    im_rec2 = im_rec2 / norm(im_rec2)

    y = mask3 .* f(Im) / N
    im_rec3 = Nesta_Cont(y, y, mask3, A, At, iter, μ, δ, 1.0, 10)
    im_rec3 = real.(ft(im_rec3) * N)
    im_rec3 = im_rec3 / norm(im_rec3)


    var1 += psnr(real.(im_rec), Im, maximum(abs.(Im)))
    var2 += psnr(real.(im_rec2), Im, maximum(abs.(Im)))
    var3 += psnr(real.(im_rec3), Im, maximum(abs.(Im)))

    Dist1[Dist1.<=exp(-60)] .= exp(-60)

    clims = (minimum([Dist1 Dist2 Dist3]), maximum([Dist1 Dist2 Dist3]) + 1e-6)

    if had == 0
        fig = Figure(size=(1800, 1000))
        ax, hm = contourf(fig[1, 1], fftshift(log.(Dist1')), colorrange=log.(clims))
        Colorbar(fig[1, 2], hm)
        ax, hm = contourf(fig[1, 3], fftshift(log.(Dist2')), colorrange=log.(clims))
        Colorbar(fig[1, 4], hm)
        ax, hm = contourf(fig[1, 5], fftshift(log.(Dist3')), colorrange=log.(clims))
        Colorbar(fig[1, 6], hm)
        heatmap(fig[2, 1], fftshift(mask1)', colormap=:grays)
        heatmap(fig[2, 3], fftshift(mask2)', colormap=:grays)
        heatmap(fig[2, 5], fftshift(mask3)', colormap=:grays)
        display(fig)
    else
        fig = Figure(size=(1800, 1000))
        ax, hm = contourf(fig[1, 1], log.(Dist1'), colorrange=log.(clims))
        Colorbar(fig[1, 2], hm)
        ax, hm = heatmap(fig[1, 3], log.(Dist2'), colorrange=log.(clims))
        Colorbar(fig[1, 4], hm)
        ax, hm = contourf(fig[1, 5], log.(Dist3'), colorrange=log.(clims))
        Colorbar(fig[1, 6], hm)
        heatmap(fig[2, 1], mask1', colormap=:grays)
        heatmap(fig[2, 3], mask2', colormap=:grays)
        heatmap(fig[2, 5], mask3', colormap=:grays)
        display(fig)
    end
    println(var1)
    println(var2)
    println(var3)
    println("S is:")
    println(S_av)
    return var1, var2, var3
end


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
                    coeffs[i] +=  (abs.(C[i]) .>= th*norm(C[i]));
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
            coeffs[i] += abs.(C[i]) .>= th*norm(C[i]);
            next!(p)
    end
    end

    println("Generating Wavelet distribution complete")
    
    coeff = sum(coeffs);
    Nmax = length(readdir(path_vec)[2:end]);
    S_av = round(sum(sum(coeff)/Nmax))
    # W = sqrt.(coeff./Nmax);
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
