
function wavelet_test()
    run_tests(; K=2^16, lk=8.0, flp=0, had=0, lines_and_blocks=0,
        train_path=joinpath(pwd(), "images", "Brain4"),
        test_path=joinpath(pwd(), "images", "brain4.tif"),
        wf=wavelet(WT.db4, WT.Filter), R=10, sep=0)
end

function wavelet_test_2()
    run_tests(; K=2^16, lk=8.0, flp=0, had=0, lines_and_blocks=0,
        train_path=joinpath(pwd(), "images", "Brain4"),
        test_path=joinpath(pwd(), "images", "brain4.tif"),
        wf=wavelet(WT.db4, WT.Filter), R=10, sep=0)
end

function levels_test()
    run_tests(; K=2^18, lk=9.0, flp=0, had=0, lines_and_blocks=0,
        train_path=joinpath(pwd(), "images", "Brain5"),
        test_path=joinpath(pwd(), "images", "brain.png"),
        wf=wavelet(WT.db4, WT.Filter), R=10, sep=0)
end


function flip_test()
    run_tests(; K=2^16, lk=15.0, flp=1, had=0, lines_and_blocks=0,
        train_path=joinpath(pwd(), "images", "Brain4"),
        test_path=joinpath(pwd(), "images", "brain.png"),
        wf=wavelet(WT.db4, WT.Filter), R=20, sep=0)
end

function knee_test()
    run_tests(; K=2^16, lk=7.0, flp=0, had=0, lines_and_blocks=0,
        train_path=joinpath(pwd(), "images", "MRNet", "train", "coronal"),
        test_path=joinpath(pwd(), "images", "MRNet", "valid", "coronal", "1190.npy"),
        wf=wavelet(WT.db4, WT.Filter), R=10, sep=0)
    # run_tests(;K = 2^14, lk = 70.,flp = 0, had = 0, lines = 1,
    # train_path = joinpath(pwd(), "images", "MRNet","train","coronal") ,
    # test_path = joinpath(pwd(),"images", "MRNet","valid","coronal","1190.npy"),
    # wf = wavelet(WT.db4, WT.Filter), R = 10)
end

function line_test()
    run_tests(; K=2^16, lk=59.0, flp=0, had=0, lines_and_blocks=1,
        train_path=joinpath(pwd(), "images", "Brain4"),
        test_path=joinpath(pwd(), "images", "brain4.tif"),
        wf=wavelet(WT.db4, WT.Filter), R=5, sep=1)
end


function run_tests(;
    #DB4 : Flip and Knee
    #K::Int64 = 2^16, lk::Float64 = 19.,flp::Int64 = 0, had::Int64 = 0, lines::Int64 = 0,
    #train_path::String = joinpath(pwd(), "images", "MRNet","train","coronal") ,test_path::String = joinpath(pwd(),"images", "MRNet","valid","coronal","1190.npy"))
    K::Int64=2^18, lk::Float64=0.0, flp::Int64=0, had::Int64=0, lines_and_blocks::Int64=0,
    train_path::String=joinpath(pwd(), "images", "Brain4"), test_path::String=joinpath(pwd(), "images", "brain.png"),
    wf=wavelet(WT.db4, WT.Filter), R=20, sep=1)

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

    #y = mask1.*f(idwt(pinv.(W).*dwt(Im,wf,L),wf,L))/N;
    y = mask1 .* f(Im) / N
    im_rec = Nesta_Cont(y, y, mask1, A, At, iter, μ, δ, 1.0, 10)
    #im_rec = idwt(W.*dwt(ft(im_rec),wf,L),wf,L)
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
        @infiltrate
        fig = Figure(resolution = (2000, 1800))
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
        save("lines.pdf", fig)
    else 
        clims = (minimum([Dist1 Dist2]), maximum([Dist1 Dist2]))
        fig = Figure(resolution = (1800, 1000))
        ax, hm = contourf(fig[1, 1], log.(Dist1'),  colorrange=log.(clims))
        ax.xticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax.yticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        Colorbar(fig[1, 2], hm)
        ax, hm = contourf(fig[1, 3], log.(Dist2'), colorrange=log.(clims))
        ax.xticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax.yticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        Colorbar(fig[1, 4], hm)
        ax, hm = heatmap(fig[2, 1], mask1', colormap=:grays)
        ax.xticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax.yticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax, hm = heatmap(fig[2, 3], mask2', colormap=:grays)
        ax.xticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax.yticks = (1:50:N, string.(Int.(-N/2:50:N/2-1)))
        ax, hm = heatmap(fig[1, 5], W' .^ 2, colormap=:grays)
        Colorbar(fig[1, 6], hm)
        ax, hm = heatmap(fig[2, 5], abs.(Im)', colormap=:grays)
        display(fig)
    end
    println("adapted variable density sub psnr:")
    println(var1)
    println("Comparison:")
    println(var2)

    return var1, var2
end


function uniform_test(;
    #DB4 : Flip and Knee
    #K::Int64 = 2^16, lk::Float64 = 19.,flp::Int64 = 0, had::Int64 = 0, lines::Int64 = 0,
    #train_path::String = joinpath(pwd(), "images", "MRNet","train","coronal") ,test_path::String = joinpath(pwd(),"images", "MRNet","valid","coronal","1190.npy"))
    K::Int64=2^16, lk::Float64=15.0, flp::Int64=0, had::Int64=1, lines::Int64=0, blocks::Int64=0,
    train_path::String=joinpath(pwd(), "images", "Brain4"), test_path::String=joinpath(pwd(), "images", "brain.png"),
    wf=wavelet(WT.haar, WT.Filter), R=20, sep=0)
    N = Int(sqrt(K))
    runs = 1
    th = (lk * 0.02) * sqrt(2 * (log(2 * K) - log(1 / 2)) / K)
    iter = 100
    L = maxtransformlevels(N)#;#min(5,maxtransformlevels(N))
    if had == 1
        f = fwht
        ft = ifwht
    else
        f = fft
        ft = ifft
    end

    # Load, transform and threshold images form path to get proxy of wavelet distribution W
    #coeff, Nmax, S_av, W =  process_image(train_path,wf,L,N,th,flp);
    S_av = sqrt(K) / 2
    W = ones(N, N) * sqrt(S_av / K)

    mask1 = zeros(N, N)
    mask2 = zeros(N, N)
    mask3 = zeros(N, N)
    Dist1, Dist2, Dist3, U = generate_l2_dist(W, wf, L, N, f, lines)
    #Dist1 = max.(Dist1,Dist2)
    #@infiltrate

    Dist2 = zeros(N, N)
    if had == 0
        for i = 1:N
            for j = 1:N
                Dist2[i, j] = 1 / max(1, (i - N / 2)^2 + (j - N / 2)^2)^(2.5)
            end
        end
        Dist2 = (fftshift(Dist2))
    else
        for i = 1:N
            for j = 1:N
                Dist2[i, j] = 1 / max(1, (i)^2 + (j)^2)^(2.5)
            end
        end
    end
    Dist2 = Dist2 / sum(Dist2)


    Dist2 = ones(N, N)
    Dist2 = Dist2 / sum(Dist2)

    A(x) = f(idwt(x, wf, L)) / N
    At(x) = dwt(ft(x) * N, wf, L)
    var1 = 0
    var2 = 0
    var3 = 0

    #for i = 1:runs
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

    for j = 1:2
        Im = zeros(N, N)
        Cff = dwt(Im, wf, L)
        Cff .= 0
        New = sample(1:K, Int64(round(S_av)))
        Cff[New] .= sign.(randn(length(New)))
        Im = idwt(Cff, wf, L)
        #Im = Cff;
        Im = Im / norm(Im)
        #mask1 = mask1'
        #W[W .== 0 ] .= 0.1^10

        #@sync begin
        δ = 1.0e-7
        μ = 0.2
        #@async begin
        #y = mask1.*f(idwt(pinv.(W).*dwt(Im,wf,L),wf,L))/N;
        y = mask1 .* f(Im) / N
        im_rec = Nesta_Cont(y, y, mask1, A, At, iter, μ, δ, 1.0, 10)
        #im_rec = idwt(W.*dwt(ft(im_rec),wf,L),wf,L)
        im_rec = real.(ft(im_rec) * N)
        #im_rec = (im_rec .- minimum(im_rec))./(maximum(im_rec) - minimum(im_rec));
        im_rec = im_rec / norm(im_rec)
        #end

        #@async begin
        y = mask2 .* f(Im) / N
        im_rec2 = Nesta_Cont(y, y, mask2, A, At, iter, μ, δ, 1.0, 10)
        im_rec2 = real.(ft(im_rec2) * N)
        #im_rec = (im_rec .- minimum(im_rec))./(maximum(im_rec) - minimum(im_rec));
        im_rec2 = im_rec2 / norm(im_rec2)

        y = mask3 .* f(Im) / N
        im_rec3 = Nesta_Cont(y, y, mask3, A, At, iter, μ, δ, 1.0, 10)
        im_rec3 = real.(ft(im_rec3) * N)
        #im_rec = (im_rec .- minimum(im_rec))./(maximum(im_rec) - minimum(im_rec));
        im_rec3 = im_rec3 / norm(im_rec3)
        #end
        #end

        var1 += psnr(real.(im_rec), Im, maximum(abs.(Im)))
        var2 += psnr(real.(im_rec2), Im, maximum(abs.(Im)))
        var3 += psnr(real.(im_rec3), Im, maximum(abs.(Im)))
    end

    #end
    Dist1[Dist1.<=exp(-60)] .= exp(-60)

    clims = (minimum([Dist1 Dist2 Dist3]), maximum([Dist1 Dist2 Dist3]) + 1e-6)
    @infiltrate
    if had == 0
        fig = Figure(resolution=(1800, 1000))
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
        fig = Figure(resolution=(1800, 1000))
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
    save("uniform.pdf", fig)
    println(var1 / runs)
    println(var2 / runs)
    println(var3 / runs)
    println("S is:")
    println(S_av)
    return var1, var2, var3
end


function runs()
    A = zeros(6, 3)
    for i = 1:10
        A[1, 1:2] .+= wavelet_test()
        A[2, 1:2] .+= knee_test()
        A[3, 1:2] .+= flip_test()
        A[4, 1:3] .+= uniform_test()
        #A[5,1:2] .+= levels_test()
        A[6, 1:2] .+= line_test()
    end
    return A
end