
#Q = [zeros(round(Int,N/2+1),N) for i = 1:Threads.nthreads()];
 #Q_current = [zeros(round(Int,N/2+1),N) for i = 1:Threads.nthreads()];
#Q_current2 = [complex(zeros(round(Int,N/2+1),N)) for i = 1:Threads.nthreads()];
#Q_est = [zeros(round(Int,N/2+1),N) for i = 1:Threads.nthreads()];


            #@inbounds @inline B[i] = get_Bas_elem(ind1[i],ind2[i],n1[i],n2[i],lvl[i],j,L2)
            #if norm(idwt(current[i],wf,L)-B[i])>0.01
            #    print("fail")
            #end#Wav_Bas(:,ind1)*Wav_Bas(:,ind2)'
            #B[i] = Wav_Bas[ lvl[i],:,ind1[i]]*Wav_Bas[lvl[i],:,ind2[i]]';
            #Q_current2[i] = plan*nablaT(current[i]);
            #if (maximum(abs.((plan*B[i])[1:round(Int,N/2+1),round(Int,N/2+2):end]) - abs.(Q_current2[i][:,end-1:-1:2,]))>0.000001)
            #    println("fail")
            #end
            #current[i][j] = 0;
            #@inline Q_current[i] .= (abs.(plan*B[i])).^2;
            #@inline Q[i] .+= Q_current[i].*W[j].^2;


            L2 = maxtransformlevels(N)
            current = zeros(N,N);
            i = 30;
        
            current[i] = 1;
            B2 = idwt(current, wf,L);
            current[i] = 0;
        
        
            #ind1 = rem(i-1,N)+1;
            #ind2 = Int(ceil(i/N));
            #n1 = Int(L2-ceil(log2(ind1))+1);
            #n2 = Int(L2-ceil(log2(ind2))+1);
            #lvl = min(n1,n2,L);
        
            #B = Wav_Bas[ lvl,:,ind1]*Wav_Bas[lvl,:,ind2]';
            #norm(B-B2);
            #F1 = fft(B);
        
            #F3 = rfft(B);
        
            #F2 = F_Wav[ lvl,:,ind1]*tra
            



            function test_images(path_vec::String,Dist,K::Int64,R,iter::Int64,N::Int64,L::Int64,wf)
                psnr_ = [float64(0) for i in 1:Threads.nthreads()];
                list = readdir(path_vec)[2:end]
                img = [zeros(N,N) for i in 1:Threads.nthreads()]
                rec = [complex(zeros(N,N)) for i in 1:Threads.nthreads()]
                p = Progress(length(list), dt=0.5,desc="Calculating psrn for each image in test set...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
                
                w = FrequencyWeights(vec(reshape(Dist,1,K)));
                ind = sample(1:K,w,Int(round(K/R));replace=false);
                mask = zeros(N,N);
                mask[ind] .= 1;
                f(x) = mask.*fft(idwt(reshape(x,N,N),wf,L))/N;
                ft(x) = dwt(ifft(x)*N,wf,L);
                    
                Threads.@threads for j in list
                        i = Threads.threadid()
                        img[i] = imresize( Gray.(load(joinpath(path_vec,j))),(N,N))
                        img[i] = convert(Matrix{Float64},img[i])
                        img[i] = img[i]./norm(img[i]);
                        rec[i] = Douglas_Rachford(mask.*fft(img[i])/N,mask,iter,N,L,wf,lk);
                        psnr_[i] += psnr(abs.(idwt(rec[i],wf,L)),img[i],maximum(abs.(img[i])));
                        next!(p)
                end
                println("l1 complete")
                return sum(psnr_)/length(list)
            end



            function gen_dist_lines(N,L,K,W,f,wf);
                # Precompute 1D Wavelet Basis and Fourier transform of wavelet basis 
                Wav_Bas = zeros(L,N,N);
                current = zeros(N);
                for j = 1:L
                    for i = 1:N
                        current[i] = 1;
                        Wav_Bas[j,:,i] = idwt(current,wf,j);
                        current[i] = 0;
                    end
                end
                F_Wav = f(Wav_Bas,2)#[:,1:round(Int,N/2+1),:];
            
            
            
                Q_current2 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
                Q1 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
                Q2 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
                #
                Q_current = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
                #Q_current2 = [zeros(N,N) for i = 1:Threads.nthreads()];
                #Q_est = [zeros(N,N) for i = 1:Threads.nthreads()];
            
                #current = [zeros(N,N) for i = 1:Threads.nthreads()];
                B = [zeros(N,N) for i = 1:Threads.nthreads()];
                n1 = ones(Int64,1,Threads.nthreads());
                n2 = ones(Int64,1,Threads.nthreads());
                lvl = ones(Int64,1,Threads.nthreads());
                ind1 = ones(Int64,1,Threads.nthreads());
                ind2 = ones(Int64,1,Threads.nthreads());
                p = Progress(K, dt=0.5,desc="Generating sampling distribution...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
            
                L2 = maxtransformlevels(N);
                pp = zeros(N,1);
                pp2 = zeros(N,1);
                for l = 1:N#round(Int,N/2+1)
                    Q_current2 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
                    
                    Q_current = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
                    
                    Threads.@threads for j = 1:K
                        i = Threads.threadid();
                        ind1[i] = rem(j-1,N)+1;
                        ind2[i] = Int(ceil(j/N));
                        n1[i] = Int(L2-ceil(log2(ind1[i]))+1);
                        n2[i] = Int(L2-ceil(log2(ind2[i]))+1);
                        lvl[i] = min(n1[i],n2[i],L);
                        Q1[i] = F_Wav[lvl[i],:,ind1[i]]*transpose(F_Wav[lvl[i],:,ind1[i]]);
                        Q_current2[i] += W[j].^2* F_Wav[ lvl[i],l,ind2[i]].^2 .*Q1[i];
                    #    Q_current2[i] += W[j].^2 .* kron(F_Wav[ lvl[i],:,ind2[i]].^2 ,Q1[i]);
                        Q2[i] = F_Wav[lvl[i],:,ind2[i]]*transpose(F_Wav[lvl[i],:,ind2[i]]);
                    #    Q_current[i] += W[j].^2 .* kron(F_Wav[ lvl[i],:,ind1[i]].^2,Q2[i]);
                        Q_current[i] += W[j].^2* F_Wav[ lvl[i],l,ind1[i]].^2 .*Q2[i];
                        #if (maximum(abs.((plan*B[i])[1:round(Int,N/2+1),round(Int,N/2+2):end]) - abs.(Q_current2[i][:,end-1:-1:2,]))>0.000001)
                        #    println("fail")
                        #end
                        next!(p)
                    end 
                pp[l] = opnorm(sum(Q_current)-Diagonal(sum(Q_current))) ;
                pp2[l] = opnorm(sum(Q_current2)-Diagonal(sum(Q_current2))) ;
                #
            
                end
            
                ppp = zeros(N,1);
                #ppp[round(Int,N/2):end] = pp;
                #ppp[1:round(Int,N/2-1)] = pp[end-1:-1:2];
                
                #ppp2 = zeros(N,1);
                #ppp2[round(Int,N/2):end] = pp2;
                #ppp2[1:round(Int,N/2-1)] = pp2[end-1:-1:2];
                ppp= pp;
                ppp2 = pp2;
            
            
                return pp, pp2
            end
            


            function Douglas_Rachford(y,mask,niter,N,L,wf,lk)
                f(x) = mask.*fwht(idwt(reshape(x,N,N),wf,L))/N;
                ft(x) = dwt(ifwht(x)*N,wf,L);
                Norm_y=1;
                gamma= .00 + 0.5^2; 
                Norm_y=norm(y[:]);
                y=y/Norm_y;
                x = zeros(size(ft(y)));
                ze = zeros(size(x));
                on = ones(size(x));
                Prox_l1(x,tau) = max.(ze,on-tau./max.(1e-15,abs.(x))).*x;
                Proj_set(x) = x + ft(y-f(x));
                #Parameters of Douglas Rachford 
                lambda= 0 + 11 * 0.15;
                z=zeros(size(ft(y)));
                p = Progress(niter, dt=0.5,desc = "l1 - minimisation...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
                for i=1:niter
                    x = Proj_set(z);
                    z += lambda.*(Prox_l1(2*x-z,gamma)-x);
                    next!(p)    
                end
            return x*Norm_y;
            end
            
            
            
            function Fista()
             end            


             function gen_lines(W,wf,L,N,f)

                # Precompute 1D Wavelet Basis and Fourier transform of wavelet basis 
                Wav_Bas = zeros(L,N,N);
                current = zeros(N);
                for j = 1:L
                    for i = 1:N
                        current[i] = 1;
                        Wav_Bas[j,:,i] = idwt(current,wf,j);
                        current[i] = 0;
                    end
                end
                F_Wav = f(Wav_Bas,2)#[:,1:round(Int,N/2+1),:];
                
                Q = [zeros(N,N) for i = 1:Threads.nthreads()];
                d2_hor = [zeros(N) for i = 1:Threads.nthreads()];
                d2_vert = [transpose(zeros(N)) for i = 1:Threads.nthreads()];
                Q_current2 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
                Q_est = [zeros(N,N) for i = 1:Threads.nthreads()];
            
                n1 = ones(Int64,1,Threads.nthreads());
                n2 = ones(Int64,1,Threads.nthreads());
                lvl = ones(Int64,1,Threads.nthreads());
                p = Progress(sum(sum(W.>0)), dt=0.5,desc="Generating sampling distribution...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
            
                L2 = maxtransformlevels(N);
            
                Threads.@threads for j in findall(W.>0)
                    i = Threads.threadid();
                    n1[i] = Int(L2-ceil(log2(j[1]))+1);
                    n2[i] = Int(L2-ceil(log2(j[2]))+1);
                    lvl[i] = min(n1[i],n2[i],L);
                    Q_current2[i] = F_Wav[ lvl[i],:,j[1]]*transpose(F_Wav[lvl[i],:,j[2]]);
                    @inline Q[i] .+= abs.(Q_current2[i]).^2 .*W[j].^2;
                    @inline d2_hor[i] .+= abs.(sum(Q_current2[i],dims=2)).^2 .*W[j].^2;
                    @inline d2_vert[i] .+= abs.(sum(Q_current2[i],dims=1)).^2 .*W[j].^2;
                    @inline Q_est[i] .= max.(Q_est[i] , abs.(Q_current2[i]).^2) ;
                    next!(p)
                end  
                d1_hor = abs.(sum(sum(Q),dims = 2));
                d1_vert = abs.(sum(sum(Q),dims = 1));
                d1 = [d1_hor; d1_vert'];
                d2 = [sum(d2_hor); sum(d2_vert)'];
                d1 = d1./sum(d1);
                d2 = d2./sum(d2);
                return d1,d2
            end
            
            

            function mult_lvl_gauss(N, a , r0  ,r , m)
                # r is the number of levels,
                # r0 are the fully sampled levels,
                # is a decay Parameter
                a = 1;
                r0 = 2 ;
            
                l = sqrt(2*(N/2)^2)/r;
                p = zeros(r,1);
                for i = 1:r
                    if i <= r0
                        p[i] = 1;
                    else
                        p[i] = exp(-((i-r0)/(r-r0))^a);
                    end
                end
            
                for j = 1:100
                    p = p./sum(p)*m
                    if i <= r0
                        p[i] = 1;
                    end
                end
                return
            end
                        

            fig = Figure(resolution = (1400, 1000))
            ga = fig[1, 1] = GridLayout()
            gb = fig[1, 2] = GridLayout()
            #gc = fig[1, 3] = GridLayout()
            gd = fig[2, 1] = GridLayout()
            ge = fig[2, 2] = GridLayout()
            #gf = fig[2, 3] = GridLayout()
            #axtop = CairoMakie.Axis(ga[1, 1])
            #axmain = CairoMakie.Axis(ga[2, 1], xlabel = "before", ylabel = "after")
            #axright = CairoMakie.Axis(ga[2, 2])
            #linkyaxes!(axmain, axright)
            #linkxaxes!(axmain, axtop)
            
        
            #print(size([fftshift(vec(d1[1:N]))' fftshift(vec(d1[N+1:end]))'] ))
            # if lines >= 1
            #     CairoMakie.Axis(fig[1, 1])
            #     lines!(1:length(vec(d1)),vec(log.([fftshift(d1[1:N])' fftshift(d1[N+1:end])'])) )
            #     # CairoMakie.Axis(fig[1, 2])
            #     lines!(1:length(vec(d1)),vec(log.([fftshift(d2[1:N])' fftshift(d2[N+1:end])'])))
            #     CairoMakie.Axis(fig[1, 3])
            #     lines!(1:K,log.(sort(vec(W))))
            # else
                # contour(fig[1,1],fftshift(log.(Dist1)),
                #     levels=-20:1:1,
                #     axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis", 
                #     title=string("Distribution optimal density")) 
                # )
                ax= contour!(CairoMakie.Axis(ga[1,1][1,1],aspect = 1),fftshift(log.(Dist1')), levels = 10)
        
                Colorbar(ga[1,1][1,2],ax)
                #lines!(axtop,1:N,vec(log.(fftshift(sum(Dist1, dims = 1)))))
                #lines!(axtop,1:N,vec(log.(fftshift(sum(Dist1, dims = 2)))'))
                #lines!(axright,vec(fftshift(sum(Dist1, dims = 2))),1:N)#, direction = :y)
                #hidedecorations!(axtop, grid = false)
                #hidedecorations!(axright, grid = false)
                #colgap!(ga, 10)
                #rowgap!(ga, 10)
                #ax.aspect = 1
                #Colorbar(ga[1, 2], axmain)
                # contour(fig[1,2],fftshift(log.(Dist2)),
                #     levels=-20:1:1,
                #     axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis",
                #     title=string("Exponential decay"))
                # )
                ax = 
                contour!(CairoMakie.Axis(gb[1,1][1,1],aspect = 1),fftshift(log.(Dist2')), levels = 12)
                Colorbar(gb[1, 1][1, 2], ax)
            
                #lines!(CairoMakie.Axis(gc[1,1]),1:K,log.(sort(vec(Dist1*S_av))))
                #lines!(CairoMakie.Axis(gc[1,1]),1:K,log.(sort(vec(W))))
            # end
            heatmap!(CairoMakie.Axis(gd[1,1],aspect = 1),fftshift(mask1)',colormap = :grays,
                #axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis")
                #)
            )
            heatmap!(CairoMakie.Axis(ge[1,1],aspect = 1),fftshift(mask2)',colormap = :grays,
                #axis = (aspect = 1,# xlabel = "x axis", ylabel = "y axis",
                #)
            )
            ax, hm = heatmap(fig[1,3],W',colormap = :grays,
                axis = (aspect = 1,# xlabel = "x axis", ylabel = "y axis",
                )
            )
            Colorbar(fig[1,4],hm)
            #heatmap(fig[3,1],abs.(im_rec)',colormap = :grays,
            #     axis = (aspect = 1, title = string( round( var1,digits=2 )))
            # )
            #heatmap(fig[3,2],abs.(im_rec2)',colormap = :grays,
            #     axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis", 
            #     title =  string( round(var2,digits=2)))
            # )
             ax, hm = heatmap(fig[2,3],abs.(Im)',colormap = :grays,
                 axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis",
                )
             )
            
            display(fig)
            #save("sym6_K20_R10_Brain2_lk20_iter100C.pdf", fig) 
        

#train_path = joinpath(pwd(), "Brain" , "Training" , "all");
#train_path = joinpath(pwd(), "images", "Brain3")
#test_path = joinpath(pwd(), "Brain" , "Testing" , "all");
#test_path = joinpath(pwd(), "Brain2" , "test" ,"all");
#train_path = joinpath(pwd(), "images", "DIV2K" , "DIV2K_train_HR");
#test_path = joinpath(pwd(), "DIV2K" , "DIV2K_valid_HR");
#train_path = joinpath(pwd(), "images", "dandelions", "dandelion" );
#test_path = joinpath(pwd(), "Grass" , "ImageTestset");
#train_path = joinpath(pwd(), "images", "forest_etc", "train" ,"forest");
#test_path = joinpath(pwd(), "forest_etc", "test","forest" );




function generate_l2_dist_fourier(W,wf,L,N,f)
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
            Wav_Bas[j,:,i] = current#idwt(current,wf,j);
            current[i] = 0;
        end
    end
    
    F_Wav = f(Wav_Bas,2)/sqrt(N)#[:,1:round(Int,N/2+1),:];
  
    # Preallocate memory for speedup. Multithreaded
    Q = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
    Q_current = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
    Q_est = [zeros(N,N) for i = 1:Threads.nthreads()];
    current = [zeros(size(W)) for i = 1:Threads.nthreads()];
    n1 = ones(Int64,1,Threads.nthreads());
    n2 = ones(Int64,1,Threads.nthreads());
    lvl = ones(Int64,1,Threads.nthreads());
    
    p = Progress(sum(sum(W.>0)), dt=0.5,desc="Generating sampling distribution...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);

    # Iterate over all entries of the weight matrix W
    Threads.@threads for j in findall(W.>=0)
        i = Threads.threadid();
        # calculate levels of different wavelets for j_1,j_2
        n1[i] = Int(L-ceil(log2(j[1]))+1);
        n2[i] = Int(L-ceil(log2(j[2]))+1);
        lvl[i] = 1#min(n1[i],n2[i],L);
        
        Q_current[i] = F_Wav[ lvl[i],:,j[1]]*transpose(F_Wav[lvl[i],:,j[2]]); #Take the correct wavelets out of the precomputet matrix
        @inline Q[i] .+= abs.(Q_current[i].^2) .*W[j].^2;
        @inline Q_est[i] .= max.(Q_est[i] , abs.(Q_current[i].^2)) ;
        next!(p)
    end  
    Q_max = zeros(N,N);
    for i = 1:Threads.nthreads()
        Q_max = max.(Q_max,Q_est[i])
    end

    K = N^2
    if K <= 2^12
        U = [zeros(K,K) for i = 1:Threads.nthreads()];
        Threads.@threads for j in findall(W.>=0)
            i = Threads.threadid();
            # calculate levels of different wavelets for j_1,j_2
            n1[i] = Int(L-ceil(log2(j[1]))+1);
            n2[i] = Int(L-ceil(log2(j[2]))+1);
            lvl[i] = min(n1[i],n2[i],L);
            
            Q_current[i] = F_Wav[ lvl[i],:,j[1]]*transpose(F_Wav[lvl[i],:,j[2]]); #Take the correct wavelets out of the precomputet matrix
            @inline U[i][:,(j[2]-1)*N+j[1]] = reshape(abs.(Q_current[i]).^2,K,1)
            next!(p)
        end
        U = sum(U)
    else
        U = 0
    end
    
    # Combine the Threads back
    Q = abs.(sum(Q)); 
   
    Q_max = abs.(Q_max);
    Q_est = max.(Q,Q_max)
   
    Q = Q./sum(sum(Q));
    Q_max = Q_max./sum(sum(Q_max));
    Q_est = Q_est./sum(Q_est)
    @infiltrate
    Dist1 = zeros(N,N);
    Dist1 = Q_est;
    Dist1 = Dist1./sum(sum(Dist1));
    
    Dist2 = zeros(N,N);
    Dist2 = Q;
    Dist2 = Dist2./sum(sum(Dist2));    

    Dist3 = zeros(N,N);
    Dist3 = Q_max;
    Dist3 = Dist3./sum(sum(Dist3));    


    return Dist1, Dist2, Dist3 ,U
end

function process_image_fourier(path_vec::String,wf,L,N::Int64,th,flp)
    #Loads images from a given path, transform them into the specified wavelet basis and get a proxy for the distribution of wavelet coefficients
    # path_vec ... Path to images
    # wf ... Wavlet Filter
    # L ... Maximum Level of Wavelet Basis
    # N ... size of images
    # th ... threshold of wavelet coefficients

    # 2022 Simon Ruetz

    coeffs = [zeros(N,N) for i in 1:Threads.nthreads()];
    list = readdir(path_vec)[2:end]
    img = [zeros(N,N) for i in 1:Threads.nthreads()]
    M = [0 for i in 1:Threads.nthreads() ]
    C = [zeros(N,N) for i in 1:Threads.nthreads()]
    p = Progress(length(list), dt=0.5,desc="Loading Images and generating distribution...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
    if occursin("Net",path_vec)
        Threads.@threads for j in list
                i = Threads.threadid()
                M[i] = size(load(joinpath(path_vec,j)))[1]
                for l = 1:M[i]
                    img[i] = convert(Array{Float64},imresize( Gray.(convert(Matrix{Float64},load(joinpath(path_vec,j))[l,:,:])),(N,N)))
                    img[i] = img[i]./norm(img[i]);
                    if flp == 1
                        C[i] = dwt(flip(img[i],wf,L,N^2,N),wf,L);
                    else
                        C[i] = dwt(img[i],wf,L);
                    end
                    #print(size(C[i]))
                    coeffs[i] += (abs.(C[i]) .>= th*norm(C[i]))/M[i];#sort(vec(abs.(C[i])))[end-165];#
                end
                next!(p)
        end
    else
        Threads.@threads for j in list
            i = Threads.threadid()
            img[i] = imresize( Gray.(load(joinpath(path_vec,j))),(N,N))
            img[i] = convert(Matrix{Float64},img[i])
            img[i] = img[i]./norm(img[i]);
            if flp == 1
                C[i] = dwt(flip(img[i],wf,L,N^2,N),wf,L);
            else
                C[i] = dwt(img[i],wf,L);
            end
            #print(size(C[i]))
            coeffs[i] += abs.(C[i]) .>= th*norm(C[i]);#sort(vec(abs.(C[i])))[end-165];#
            next!(p)
    end
    end

    println("Generating Wavelet distribution complete")
    
    coeff = sum(coeffs);
    Nmax = length(readdir(path_vec)[2:end]);
    S_av = round(sum(sum(coeff)/Nmax))
    W = sqrt.(coeff./Nmax);
    return coeff, Nmax, S_av, W
end


function fourier_test( ; 
    #DB4 : Flip and Knee
    #K::Int64 = 2^16, lk::Float64 = 19.,flp::Int64 = 0, had::Int64 = 0, lines::Int64 = 0,
    #train_path::String = joinpath(pwd(), "images", "MRNet","train","coronal") ,test_path::String = joinpath(pwd(),"images", "MRNet","valid","coronal","1190.npy"))
    K::Int64 = 2^14, lk::Float64 = 15.,flp::Int64 = 0, had::Int64 = 0, lines::Int64 = 0,
    train_path::String = joinpath(pwd(), "images", "Brain4") ,test_path::String = joinpath(pwd(),"images","brain.png"),
    wf = wavelet(WT.sym10, WT.Filter))
    runs = 1
    N = Int(sqrt(K));
    th = (lk*0.02)*sqrt(2*(log(2*K)-log(1/2))/K);
    R=100;
    iter = 100;
    L =1# maxtransformlevels(N);#;#min(5,maxtransformlevels(N))
    if had == 1
        f = fwht
        ft = ifwht
    else
        f = FFTW.dct
        ft = FFTW.idct
    end

    # Load, transform and threshold images form path to get proxy of wavelet distribution W
    #coeff, Nmax, S_av, W =  process_image_fourier(train_path,wf,L,N,th,flp);
    S_av =50;
    W = ones(N,N)*sqrt(S_av/K);
    
    mask1 = zeros(N,N);
    mask2 = zeros(N,N);
    mask3 = zeros(N,N);
    Dist1, Dist2, Dist3 ,U = generate_l2_dist_fourier(W,wf,L,N,f)
    #Dist1 = max.(Dist1,Dist2)
    #@infiltrate

    # Dist2 = zeros(N,N);
    # if had == 0
    #     for i = 1:N
    #         for j = 1:N
    #             Dist2[i,j] = 1/max(1,(i-N/2)^2 + (j-N/2)^2)^(2.5);
    #         end
    #     end
    #     Dist2 = (fftshift(Dist2));
    # else
    #     for i = 1:N
    #         for j = 1:N
    #             Dist2[i,j] = 1/max(1,(i)^2 + (j)^2)^(2.5);
    #         end
    #     end
    # end
    # Dist2 = Dist2/sum(Dist2);


    #Dist2 = ones(N,N);
    #Dist2 = Dist2/sum(Dist2);

    A(x) = f(x)/N;#f(idwt(x,wf,L))/N;
    At(x) =ft(x)*N# dwt(ft(x)*N,wf,L);
    var1 = 0
    var2 = 0
    var3 = 0

   # for i = 1:runs
        mask1 = zeros(N,N);
        mask2 = zeros(N,N);
        mask3 = zeros(N,N);
        w1 = FrequencyWeights(vec(reshape(Dist1,1,K)));
        ind1 = sample(1:K,w1,Int(round(K/R));replace=false);
        mask1[ind1] .= 1;
        w2 = FrequencyWeights(vec(reshape(Dist2,1,K)));
        ind2 = sample(1:K,w2,Int(round(K/R));replace=false);
        mask2[ind2] .= 1;
        w3 = FrequencyWeights(vec(reshape(Dist3,1,K)));
        ind3 = sample(1:K,w3,Int(round(K/R));replace=false);
        mask3[ind3] .= 1;

        
        Im = zeros(N,N);
        Cff = dwt(Im,wf,L);
        Cff .= 0;
        w4 = FrequencyWeights(vec(reshape(W,1,K)));
        New = sample(1:K,w4,Int64(round(S_av)));
        Cff[New] .= 1;
        #Im = idwt(Cff,wf,L)
        Im = Cff;
        Im = Im/norm(Im)
        #mask1 = mask1'
        #W[W .== 0 ] .= 0.1^10
        # Im = load(test_path);
        # if length(size(Im))>2
        #     l = Int64(round(size(Im)[1]*1/10))
        #     Im = convert(Array{Float64},imresize( Gray.(convert(Matrix{Float64},load(test_path)[l,:,:])),(N,N)))
        # else
        #     Im = imresize(Gray.(Im)::Array{Gray{N0f8},2},(N,N));
        #     Im = convert(Array{Float64},Im);
        # end
        # Im = Im./norm(Im);
        #@sync begin
            δ = 1.0e-7
            μ = 0.2
            #@async begin
                #y = mask1.*f(idwt(pinv.(W).*dwt(Im,wf,L),wf,L))/N;
                y = mask1.*f(Im)/N;
                im_rec = Nesta_Cont(y,y,mask1,A,At,iter,μ,δ,1.,10);
                #im_rec = idwt(W.*dwt(ft(im_rec),wf,L),wf,L)
                im_rec = real.(ft(im_rec)*N)
                #im_rec = (im_rec .- minimum(im_rec))./(maximum(im_rec) - minimum(im_rec));
                im_rec = im_rec/norm(im_rec);
            #end

            #@async begin
                y = mask2.*f(Im)/N;
                im_rec2 = Nesta_Cont(y,y,mask2,A,At,iter,μ,δ,1.,10);
                im_rec2 = real.(ft(im_rec2)*N)
                #im_rec = (im_rec .- minimum(im_rec))./(maximum(im_rec) - minimum(im_rec));
                im_rec2 = im_rec2/norm(im_rec2);

                y = mask3.*f(Im)/N;
                im_rec3 = Nesta_Cont(y,y,mask3,A,At,iter,μ,δ,1.,10);
                im_rec3 = real.(ft(im_rec3)*N)
                #im_rec = (im_rec .- minimum(im_rec))./(maximum(im_rec) - minimum(im_rec));
                im_rec3 = im_rec3/norm(im_rec3);
            #end
        #end
        
        var1 += psnr(real.(im_rec),Im,maximum(abs.(Im)));   
        var2 += psnr(real.(im_rec2),Im,maximum(abs.(Im))); 
        var3 += psnr(real.(im_rec3),Im,maximum(abs.(Im))); 
        
    #end
    Dist1[Dist1 .<= exp(-60)] .= exp(-60)

    clims = (minimum([Dist1 Dist2 Dist3]),maximum([Dist1 Dist2 Dist3])+1e-6)
    
    if had == 0
        fig = Figure(resolution = (1800, 1000))
        ax ,hm = heatmap(fig[1,1],fftshift(log.(Dist1')), colorrange = log.(clims))
        Colorbar(fig[1,2],hm)
        ax, hm = heatmap(fig[1,3],fftshift(log.(Dist2')), colorrange = log.(clims))
        Colorbar(fig[1,4], hm)
        ax, hm = heatmap(fig[1,5],fftshift(log.(Dist3')), colorrange = log.(clims))
        Colorbar(fig[1,6], hm)
        heatmap(fig[2,1],fftshift(mask1)',colormap = :grays)
        heatmap(fig[2,3],fftshift(mask2)',colormap = :grays)
        heatmap(fig[2,5],fftshift(mask3)',colormap = :grays)
        ax, hm = heatmap(fig[3,1],abs.(im_rec)',colormap = :grays)
        ax, hm = heatmap(fig[3,3],abs.(im_rec2)',colormap = :grays)
        ax, hm = heatmap(fig[3,5],abs.(im_rec3)',colormap = :grays)
        display(fig)
    else
        fig = Figure(resolution = (1800, 1000))
        ax ,hm = heatmap(fig[1,1],log.(Dist1'), colorrange = log.(clims))
        Colorbar(fig[1,2],hm)
        ax, hm = heatmap(fig[1,3],log.(Dist2'), colorrange = log.(clims))
        Colorbar(fig[1,4], hm)
        ax, hm = heatmap(fig[1,5],log.(Dist3'), colorrange = log.(clims))
        Colorbar(fig[1,6], hm)
        heatmap(fig[2,1],mask1',colormap = :grays)
        heatmap(fig[2,3],mask2',colormap = :grays)
        heatmap(fig[2,5],mask3',colormap = :grays)
        display(fig)
    end
    @infiltrate
    println(var1/runs)
    println(var2/runs)
    println(var3/runs)
    println("S is:")
    println(S_av)
    return var1, var2, var3
end



var_ = zeros(3,2,500);
#for lk = 1:1:30
lk = 20
# lk = 20 for haar 18, ... , R4 Brain
# lk = 45 for haar 20, lk steps 0.02 starting from 0.02, R10 Brain
# lk = 20 for db4 18, lk stpes 0.02 starting from 0.05, R4 Brain
# lk = 60 for beyl 14, lk steps 0.02 starting from 0.05, R10 forest
# lk = 60 for beyl 18, lk steps 0.02 starting from 0.05, R10 Brain
# lk 40 for db4 18, lk steps 0.02 starting from 0.05, R 10 Brain
# lk 8 for haar 16, lk steps 0.02 starting from 0.05, R 10 Brain
# lk 20 for db4 16, lk steps 0.02 starting from 0.05, R 10 Brain
    #print(lk)
    #Dict_Q = matread("Brain/tumor_18_db4_R58.mat") 
    #Q = Dict_Q["Q"];
    #Q_new = Dict_Q["Q"];
    #Q_max = Q_new;
    #Q_est = Q_new;
    CairoMakie.activate!(type = "png")
    K = 2^18;
    N = Int(sqrt(K));
    th = (0.05+lk*0.02)*sqrt(2*(log(2*K)-log(1/2))/K);
    R=4;
    runs = 0;
    iter = 100;
    wf = wavelet(WT.db4, WT.Filter);
    L = maxtransformlevels(N);#;#min(5,maxtransformlevels(N))

    Q = []
    #@load "DIV2K/DIV2K_22_db4_th03_L5.jld2"
    #@load "Brain/Tumor_20_db4_th03_L5.jld2"
    #Dist2[1:Int(Nø2+1),:] = Q_est;
    #Dist2[Int(N/2+2):end,:] = Q_est[end-1:-1:2,:];
    #Q_new = zeros(N,N);

    # for a quarter 
    #Q_new[1:Int(N/2+1),1:Int(N/2+1)]  = Q;
    #Q_new[Int(N/2+2):end,1:Int(N/2+1)] = Q[end-1:-1:2,:]; #
    #Q_new[:,Int(N/2+2):end] = Q_new[:,Int(N/2):-1:2];

    # for half
   # Q_new[1:Int(N/2+1),:]  = Q;
    #Q_new[Int(N/2+2):end,:] = Q[end-1:-1:2,:];

    #Q = Q_new;
    #Q_est = Q_new;

    FFTW.set_num_threads(1);
    #train_path = joinpath(pwd(), "Brain" , "Training" , "all");
    #train_path = joinpath(pwd(), "images", "Brain3")
    #test_path = joinpath(pwd(), "Brain" , "Testing" , "all");
    train_path = joinpath(pwd(), "images", "Brain2" , "all");
    #test_path = joinpath(pwd(), "Brain2" , "test" ,"all");
    #train_path = joinpath(pwd(), "images", "DIV2K" , "DIV2K_train_HR");
    #test_path = joinpath(pwd(), "DIV2K" , "DIV2K_valid_HR");
    #train_path = joinpath(pwd(), "images", "dandelions", "dandelion" );
    #test_path = joinpath(pwd(), "Grass" , "ImageTestset");
    #train_path = joinpath(pwd(), "images", "forest_etc", "train" ,"forest");
    #test_path = joinpath(pwd(), "forest_etc", "test","forest" );


    Wav_Bas = zeros(L,N,N);
    current = zeros(N);
    for j = 1:L
        for i = 1:N
            current[i] = 1;
            Wav_Bas[j,:,i] = idwt(current,wf,j);
            current[i] = 0;
        end
    end

    F_Wav = fft(Wav_Bas,2)#[:,1:round(Int,N/2+1),:];

    L2 = maxtransformlevels(N)
    current = zeros(N,N);
    i = 30;

    current[i] = 1;
    B2 = idwt(current, wf,L);
    current[i] = 0;


    #ind1 = rem(i-1,N)+1;
    #ind2 = Int(ceil(i/N));
    #n1 = Int(L2-ceil(log2(ind1))+1);
    #n2 = Int(L2-ceil(log2(ind2))+1);
    #lvl = min(n1,n2,L);

    #B = Wav_Bas[ lvl,:,ind1]*Wav_Bas[lvl,:,ind2]';
    #norm(B-B2);
    #F1 = fft(B);

    #F3 = rfft(B);

    #F2 = F_Wav[ lvl,:,ind1]*transpose(F_Wav[lvl,:,ind2])

function fliptest(M)
    reshape(reshape(M,1,K)[:,end:-1:1],N,N)
end
    #if Q == []
        #Loads an image from a given path and performs some basic transformations to it.
        function process_image(path_vec::String,h::Int64,w::Int64)
            coeffs = [zeros(N,N) for i in 1:Threads.nthreads()];
            list = readdir(path_vec)[2:end]
            img = [zeros(N,N) for i in 1:Threads.nthreads()]
            C = [zeros(N,N) for i in 1:Threads.nthreads()]
            p = Progress(length(list), dt=0.5,desc="Loading Images and generating distribution...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
            Threads.@threads for j in list
                    i = Threads.threadid()
                    img[i] = imresize( Gray.(load(joinpath(path_vec,j))),(h,w))
                    img[i] = convert(Matrix{Float64},img[i])
                    img[i] = img[i]./norm(img[i]);
                    C[i] = dwt(img[i],wf,L);
                    #print(size(C[i]))
                    coeffs[i] += abs.(C[i]) .>= th*norm(C[i]);#sort(vec(abs.(C[i])))[end-165];#
                    next!(p)
            end
            println("Generating Wavelet distribution complete")
            return coeffs
        end


        #Processes all images
        coeffs =  process_image(train_path,N,N);
        coeff = sum(coeffs);
        Nmax = length(readdir(train_path)[2:end]);
        S_av = round(sum(sum(coeff)/Nmax))



        W = sqrt.(fliptest(coeff)./Nmax);
        R_est = round(K/(S_av*log(K)));
            
        println(R_est)
        println(R)
        if R_est > R
            println("u should decrease the threshold")
        else
            println("m_est is smaller than m")
        end



        #Q = [zeros(round(Int,N/2+1),N) for i = 1:Threads.nthreads()];
        #Q_current = [zeros(round(Int,N/2+1),N) for i = 1:Threads.nthreads()];
        #Q_current2 = [complex(zeros(round(Int,N/2+1),N)) for i = 1:Threads.nthreads()];
        #Q_est = [zeros(round(Int,N/2+1),N) for i = 1:Threads.nthreads()];

        Q = [zeros(N,N) for i = 1:Threads.nthreads()];
        d2 = [zeros(N) for i = 1:Threads.nthreads()];
        Q_current = [zeros(N,N) for i = 1:Threads.nthreads()];
        Q_current2 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
        Q_est = [zeros(N,N) for i = 1:Threads.nthreads()];

        current = [zeros(size(W)) for i = 1:Threads.nthreads()];
        B = [zeros(N,N) for i = 1:Threads.nthreads()];
        n1 = ones(Int64,1,Threads.nthreads());
        n2 = ones(Int64,1,Threads.nthreads());
        lvl = ones(Int64,1,Threads.nthreads());
        ind1 = ones(Int64,1,Threads.nthreads());
        ind2 = ones(Int64,1,Threads.nthreads());
        p = Progress(sum(sum(W.>0)), dt=0.5,desc="Generating sampling distribution...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);

        #U = zeros(K,K);
        plan = plan_rfft(zeros(N,N); flags=FFTW.EXHAUSTIVE, timelimit=Inf);
        L2 = maxtransformlevels(N);

        Threads.@threads for j in findall(W.>0)
            i = Threads.threadid();
            #current[i][j] = 1;
            #ind1[i] = rem(j-1,N)+1;
            #ind2[i] = Int(ceil(j/N));
            n1[i] = Int(L2-ceil(log2(j[1]))+1);
            n2[i] = Int(L2-ceil(log2(j[2]))+1);
            lvl[i] = min(n1[i],n2[i],L);
            

            Q_current2[i] = F_Wav[ lvl[i],:,j[1]]*transpose(F_Wav[lvl[i],:,j[2]]);
            #@inbounds @inline B[i] = get_Bas_elem(ind1[i],ind2[i],n1[i],n2[i],lvl[i],j,L2)
            #if norm(idwt(current[i],wf,L)-B[i])>0.01
            #    print("fail")
            #end#Wav_Bas(:,ind1)*Wav_Bas(:,ind2)'
            #B[i] = Wav_Bas[ lvl[i],:,ind1[i]]*Wav_Bas[lvl[i],:,ind2[i]]';
            #Q_current2[i] = plan*nablaT(current[i]);
            #if (maximum(abs.((plan*B[i])[1:round(Int,N/2+1),round(Int,N/2+2):end]) - abs.(Q_current2[i][:,end-1:-1:2,]))>0.000001)
            #    println("fail")
            #end
            #current[i][j] = 0;
            #@inline Q_current[i] .= (abs.(plan*B[i])).^2;
            #@inline Q[i] .+= Q_current[i].*W[j].^2;
            @inline Q[i] .+= abs.(Q_current2[i]).^2 .*W[j].^2;
            
            @inline Q_est[i] .= max.(Q_est[i] , abs.(Q_current2[i]).^2) ;
            next!(p)
            #print(j)
            #U[:,j] = vec(abs.(Q_current2[i]).^2 );
        end  

        Q = abs.(sum(Q));
        Q_max = abs.(sum(Q_est));
        Q_est = Q.*Q_max;

        Q = Q./sum(sum(Q));
        Q_max = Q_max./sum(sum(Q_max));
        Q_est = Q_est./sum(sum(Q_est));




        S_av = sum(sum(W.^2));
        R_est = round(K/(S_av*log(K)));
                
        print(R_est)
        print(R)
        if R_est > R
            print("u should decrease the threshold")
        else
            print("m_est is smaller than m")
        end

    #end

    function gen_dist_lines(N,L,K,F_Wav,W);
        
        Q_current2 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
        

        Q1 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
        Q2 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
        #
        Q_current = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
        #Q_current2 = [zeros(N,N) for i = 1:Threads.nthreads()];
        #Q_est = [zeros(N,N) for i = 1:Threads.nthreads()];

        #current = [zeros(N,N) for i = 1:Threads.nthreads()];
        B = [zeros(N,N) for i = 1:Threads.nthreads()];
        n1 = ones(Int64,1,Threads.nthreads());
        n2 = ones(Int64,1,Threads.nthreads());
        lvl = ones(Int64,1,Threads.nthreads());
        ind1 = ones(Int64,1,Threads.nthreads());
        ind2 = ones(Int64,1,Threads.nthreads());
        p = Progress(K, dt=0.5,desc="Generating sampling distribution...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);

        plan = plan_rfft(zeros(N,N); flags=FFTW.EXHAUSTIVE, timelimit=Inf);
        L2 = maxtransformlevels(N);
        pp = zeros(N,1);
        pp2 = zeros(N,1);
        @showprogress for l = 1:N#round(Int,N/2+1)
            Q_current2 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
            
            Q_current = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
            
            Threads.@threads for j = 1:K
                i = Threads.threadid();
                ind1[i] = rem(j-1,N)+1;
                ind2[i] = Int(ceil(j/N));
                n1[i] = Int(L2-ceil(log2(ind1[i]))+1);
                n2[i] = Int(L2-ceil(log2(ind2[i]))+1);
                lvl[i] = min(n1[i],n2[i],L);
                Q1[i] = F_Wav[lvl[i],:,ind1[i]]*transpose(F_Wav[lvl[i],:,ind1[i]]);
                Q_current2[i] += W[j].^2* F_Wav[ lvl[i],l,ind2[i]].^2 .*Q1[i];
            #    Q_current2[i] += W[j].^2 .* kron(F_Wav[ lvl[i],:,ind2[i]].^2 ,Q1[i]);
                Q2[i] = F_Wav[lvl[i],:,ind2[i]]*transpose(F_Wav[lvl[i],:,ind2[i]]);
            #    Q_current[i] += W[j].^2 .* kron(F_Wav[ lvl[i],:,ind1[i]].^2,Q2[i]);
                Q_current[i] += W[j].^2* F_Wav[ lvl[i],l,ind1[i]].^2 .*Q2[i];
                #if (maximum(abs.((plan*B[i])[1:round(Int,N/2+1),round(Int,N/2+2):end]) - abs.(Q_current2[i][:,end-1:-1:2,]))>0.000001)
                #    println("fail")
                #end
            #    next!(p)
            end 
        pp[l] = opnorm(sum(Q_current)-Diagonal(sum(Q_current))) ;
        pp2[l] = opnorm(sum(Q_current2)-Diagonal(sum(Q_current2))) ;
        #

        end

        ppp = zeros(N,1);
        #ppp[round(Int,N/2):end] = pp;
        #ppp[1:round(Int,N/2-1)] = pp[end-1:-1:2];
        
        #ppp2 = zeros(N,1);
        #ppp2[round(Int,N/2):end] = pp2;
        #ppp2[1:round(Int,N/2-1)] = pp2[end-1:-1:2];
        ppp= pp;
        ppp2 = pp2;


        fig = Figure()
        lines(fig[1,1],1:N,vec(log.(ppp)))
        lines(fig[1,2],1:N,vec(log.(ppp2)))
        fig    
    end





    #R = R_est
    cmaximum = maximum(maximum(maximum(log.(Q))));
    cminimum = minimum(minimum(minimum(log.(Q))));
    #plot(contour(1:N,1:N, Q))

    Dist1 = zeros(N,N);
    #Dist1[1:Int(N/2+1),:] = Q;
    #Dist1[Int(N/2+2):end,:] = Q[end-1:-1:2,:];
    Dist1 = sqrt.(Q);
    Dist1 = Dist1./sum(sum(Dist1));


    figure = (; resolution=(1000, 800), font="CMU Serif")
    axis = (; xlabel=L"x", ylabel=L"y", aspect=DataAspect())
    fig, ax, pltobj = heatmap(abs.(fftshift(log.(Dist1))); colorrange=(abs.(cminimum), abs.(cmaximum)),
        colormap=Reverse(:viridis), axis=axis, figure=figure)
    Colorbar(fig[1, 2], pltobj, label = "Reverse sequential colormap")
    fig

    Dist2 = zeros(N,N);
    #Dist2[1:Int(N/2+1),:] = Q_est;
    #Dist2[Int(N/2+2):end,:] = Q_est[end-1:-1:2,:];
    Dist2 = sqrt.(Q_est);
    Dist2 = Dist2./sum(sum(Dist2));


    Dist3 = zeros(N,N);
    #Dist3[1:Int(N/2+1),:] = Q_max;
    #Dist3[Int(N/2+2):end,:] = Q_max[end-1:-1:2,:];
    #Dist1 = Q;
    #Dist2 = Q_est;
    #Dist3 = Q_max;

    Dist3 = zeros(N,N);
    for i = 1:N
        for j = 1:N
            Dist3[i,j] = 1/max(1,(i-N/2)^2 + (j-N/2)^2)^(2.75);
        end
    end
    Dist3 = (fftshift(Dist3));
    Dist3 = Dist3/sum(Dist3);

    #W_new = sqrt.(abs.(reshape(pinv(U)*vec(Dist3),N,N)));
    #W_new = W_new./sum(sum(W_new))*sum(sum(W));
    #plot(sort(vec(W)))
    #plot(sort(vec(W_new)))

    #contour(1:N, 1:N, fftshift(log.(Dist1)), xlabel="x", ylabel="y", fill=true, aspect_ratio=:equal)
    #contour(1:N, 1:N, fftshift(log.(Dist2)), xlabel="x", ylabel="y", fill=true, aspect_ratio=:equal)
    #contour(1:N, 1:N, fftshift(log.(Dist3)), xlabel="x", ylabel="y", fill=true, aspect_ratio=:equal)



    #R = 12
    #iter = 100;
    sav_var = zeros(3,1);


    function Solve_l1_problemDR(y,mask,niter,N,L,wf,lk)
        f(x) = mask.*fft(idwt(reshape(x,N,N),wf,L))/N;
        ft(x) = dwt(ifft(x)*N,wf,L);
        Norm_y=1;
        gamma= .00 + 0.5^2; 
        Norm_y=norm(y[:]);
        y=y/Norm_y;
        x = zeros(size(ft(y)));
        ze = zeros(size(x));
        on = ones(size(x));
        Prox_l1(x,tau) = max.(ze,on-tau./max.(1e-15,abs.(x))).*x;
        Proj_set(x) = x + ft(y-f(x));
        #Parameters of Douglas Rachford 
        lambda= 0 + 11 * 0.15;
        z=zeros(size(ft(y)));
        p = Progress(niter, dt=0.5,desc = "l1 - minimisation...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
        for i=1:niter
            x = Proj_set(z);
            z += lambda.*(Prox_l1(2*x-z,gamma)-x);
            next!(p)    
        end
    return x*Norm_y;
    end




Im = load(joinpath(pwd(),"images","SINGINGINTHEBRAIN.png"));
Im = imresize(Gray.(Im)::Array{Gray{N0f8},2},(N,N));
Im = convert(Array{Float64},Im);
Im = Im/norm(Im);
Im_org = Im;
Im = idwt(fliptest(dwt(Im,wf,L)),wf,L);
#Im = reshape(reshape(Im,1,K)[end:-1:1],N,N);


w1 = FrequencyWeights(vec(reshape(Dist1,1,K)));
ind1 = sample(1:K,w1,Int(round(K/R));replace=false);
#ind1 = zeros(Int64,Int(round(K/R)),1);
#StatsBase.naive_wsample_norep!(1:K,w1,ind1);
mask1 = zeros(N,N);
mask1[ind1] .= 1;

# w2 = FrequencyWeights(vec(reshape(Dist2,1,K)));
# ind2 = sample(1:K,w2,Int(round(K/R));replace=false);
# mask2 = zeros(N,N);
# mask2[ind2] .= 1;

w3 = FrequencyWeights(vec(reshape(Dist3,1,K)));
ind3 = sample(1:K,w3,Int(round(K/R));replace=false);
mask3= zeros(N,N);
mask3[ind3] .= 1;

#@time rec1 = Solve_l1_problemDR(mask1.*fft(Im)/N,mask1,1,N,L,wf,lk);
# rec2 = Solve_l1_problemDR(mask2.*fft(Im)/N,mask2,1,N,L,wf,lk);
#rec3 = Solve_l1_problemDR(mask3.*fft(Im)/N,mask3,1,N,L,wf,lk);





    function T(x,μ::Float64)
        return(x./(max.(abs.(x),μ)))
    end
    
    
    function Nesta(y,z₀,mask,A,At,niter,μ::Float64=0.2, η::Float64=1.,c::Number=1.)
        #z₀ = At(y);
        # Ensure that A is normalised
        q₂ = copy(z₀);
        z = copy(z₀);
        p = Progress(niter, dt=0.5,desc = "l1 - minimisation...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
        for i = 1:niter 
            Tₙ = μ * A(T(At(z),μ));
            #println(norm(Tₙ))
            q = z - Tₙ;
            λ = max(0, η^(-1) * norm(y - √c * mask.*q) -1 );
            #println(λ)
            xₙ = ( λ / √c * mask.*y + q ) - λ/((λ + 1)*c)* (λ / √c * mask.*y+mask.*q ) ;
            q₂ .-= (i+1)/2 * Tₙ;
            λ = max(0, η^(-1) * norm(y - √c * mask.*q₂) -1 );
            #println(λ)
            vₙ = ( λ / √c * mask.*y + q₂ ) - λ/((λ + 1)*c)* (λ / √c * mask.*y+mask.*q₂ ) ;
            z = 2/(i+3)  * vₙ + (1 - 2/(i+3) ) * xₙ;
            #println(norm(z))
            next!(p)
        end
        return(z)
    end


A(x) = fft(idwt(x,wf,L))/N;
At(x) = dwt(ifft(x)*N,wf,L);


y = mask1.*fft(Im)/N;
#iter = 10;
im_rec = Nesta(y,y,mask1,A,At,20,0.2,0.01,1.);
im_rec = Nesta(y,im_rec, mask1,A,At,20,0.02,0.01,1.);
im_rec = Nesta(y,im_rec,mask1,A,At,20,0.002,0.01,1.);
im_rec = Nesta(y,im_rec,mask1,A,At,20,0.0002,0.01,1.);
im_rec = Nesta(y,im_rec,mask1,A,At,20,0.00002,0.01,1.);
im_rec = ifft(im_rec)
#im_rec = (im_rec .- minimum(im_rec))./(maximum(im_rec) - minimum(im_rec));
im_rec = im_rec/norm(im_rec);




y = mask3.*fft(Im)/N;
#iter = 10;
im_rec3 = Nesta(y,y,mask3,A,At,20,0.2,0.01,1.);
im_rec3 = Nesta(y,im_rec3, mask3,A,At,20,0.02,0.01,1.);
im_rec3 = Nesta(y,im_rec3,mask3,A,At,20,0.002,0.01,1.);
im_rec3 = Nesta(y,im_rec3,mask3,A,At,20,0.0002,0.01,1.);
im_rec3 = Nesta(y,im_rec3,mask3,A,At,20,0.00002,0.01,1.);
im_rec3 = ifft(im_rec3)
#im_rec = (im_rec .- minimum(im_rec))./(maximum(im_rec) - minimum(im_rec));
im_rec3 = im_rec3/norm(im_rec3);

im_rec = idwt(fliptest(dwt(im_rec,wf,L)),wf,L);
im_rec3 = idwt(fliptest(dwt(im_rec3,wf,L)),wf,L);
#var_[1,2,lk] = psnr(abs.(idwt(rec1,wf,L)),Im_org,maximum(abs.(Im_org))); 
var_[2,2,lk] = psnr(abs.(im_rec),Im_org,maximum(abs.(Im_org)));   
var_[3,2,lk] = psnr(abs.(im_rec3),Im_org,maximum(abs.(Im_org))); 






fig = Figure(resolution=(1200, 1200))
    contour(fig[1,1],fftshift(log.(Dist1)),
     levels=-30:1:1,
        axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis", 
        colormap = :grays, title=string(round(av_psnr1,digits=4)))
    )
    #Colorbar(fig[1,2],limits = [minimum(log.(Dist1)),maximum(log.(Dist1))])
    # contour(fig[1,2],fftshift(log.(Dist2)),
    #levels=-20:1:1,
    #     axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis", 
    #     colormap = :grays, title=string(round(av_psnr2,digits=4)))
    # )
    contour(fig[1,2],fftshift(log.(Dist3)),
    levels=-30:1:1,
        axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis",
        colormap = :grays, title=string(round(av_psnr3,digits=4)))
    )
    heatmap(fig[2,1],fftshift(mask1),colormap = :grays,
        figure = (backgroundcolor = :pink,),
        axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis")
        colormap = :grays)
    )
    # heatmap(fig[2,2],fftshift(mask1),colormap = :grays,
    #     figure = (backgroundcolor = :pink,),
    #     axis = (aspect = 1,# xlabel = "x axis", ylabel = "y axis",
    #     colormap = :grays)
    # )
    heatmap(fig[2,2],fftshift(mask3),colormap = :grays,
        figure = (backgroundcolor = :pink,),
        axis = (aspect = 1,# xlabel = "x axis", ylabel = "y axis",
        colormap = :grays)
    )
    heatmap(fig[2,3],W,colormap = :grays,
        figure = (backgroundcolor = :pink,),
        axis = (aspect = 1,# xlabel = "x axis", ylabel = "y axis",
        colormap = :grays)
    )
    # image(fig[3,1],abs.(idwt(rec1,wf,L)),
    # #    figure = (backgroundcolor = :pink,),
    #     axis = (aspect = 1, 
    #     title = string( var_[1,2,lk] ))
    # )
    image(fig[3,1],abs.(im_rec),
        axis = (aspect = 1, 
        title = string( var_[2,2,lk] ))
    )
    image(fig[3,2],abs.(im_rec3),
        axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis",
        title =  string( var_[3,2,lk] ))
    )
    image(fig[3,3],abs.(idwt(fliptest(dwt(Im,wf,L)),wf,L)),
        figure = (backgroundcolor = :pink,),
        axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis",
        title = "original")
    )
    fig
    save("flip.pdf", fig) 




    var_ = zeros(3,2,500);
    #for lk = 1:1:30
    lk = 50
    # lk = 45 for haar 20, lk steps 0.02 starting from 0.02, R10 Brain
    # lk = 20 for db4 18, lk stpes 0.02 starting from 0.05, R4 Brain
    # lk = 60 for beyl 14, lk steps 0.02 starting from 0.05, R10 forest
    # lk = 60 for beyl 18, lk steps 0.02 starting from 0.05, R10 Brain
    # lk 40 for db4 18, lk steps 0.02 starting from 0.05, R 10 Brain
    # lk 8 for haar 16, lk steps 0.02 starting from 0.05, R 10 Brain
    # lk 20 for db4 16, lk steps 0.02 starting from 0.05, R 10 Brain
        #print(lk)
        #Dict_Q = matread("Brain/tumor_18_db4_R58.mat") 
        #Q = Dict_Q["Q"];
        #Q_new = Dict_Q["Q"];
        #Q_max = Q_new;
        #Q_est = Q_new;
        CairoMakie.activate!(type = "png")
        K = 2^18;
        N = Int(sqrt(K));
        th = (0.05+lk*0.02)*sqrt(2*(log(2*K)-log(1/2))/K);
        R=10;
        runs = 0;
    
        iter = 20;
        wf = wavelet(WT.db6, WT.Filter);
        L = maxtransformlevels(N);#;#min(5,maxtransformlevels(N))
    
        Q = []
        #@load "DIV2K/DIV2K_22_db4_th03_L5.jld2"
        #@load "Brain/Tumor_20_db4_th03_L5.jld2"
        #Dist2[1:Int(N/2+1),:] = Q_est;
        #Dist2[Int(N/2+2):end,:] = Q_est[end-1:-1:2,:];
        #Q_new = zeros(N,N);
    
        # for a quarter 
        #Q_new[1:Int(N/2+1),1:Int(N/2+1)]  = Q;
        #Q_new[Int(N/2+2):end,1:Int(N/2+1)] = Q[end-1:-1:2,:]; #
        #Q_new[:,Int(N/2+2):end] = Q_new[:,Int(N/2):-1:2];
    
        # for half
        # Q_new[1:Int(N/2+1),:]  = Q;
        #Q_new[Int(N/2+2):end,:] = Q[end-1:-1:2,:];
    
        #Q = Q_new;
        #Q_est = Q_new;
    
        FFTW.set_num_threads(1);
        #train_path = joinpath(pwd(), "Brain" , "Training" , "all");
        #test_path = joinpath(pwd(), "Brain" , "Testing" , "all");
        #train_path = joinpath(pwd(), "Brain_2" , "all");
        #test_path = joinpath(pwd(), "Brain_2" , "test" ,"all");
        #train_path = joinpath(pwd(), "DIV2K" , "DIV2K_train_HR");
        #test_path = joinpath(pwd(), "DIV2K" , "DIV2K_valid_HR");
        #train_path = joinpath(pwd(), "dandelions", "dandelion" );
        #test_path = joinpath(pwd(), "Grass" , "ImageTestset");
        #train_path = joinpath(pwd(), "forest_etc", "train" ,"forest");
        train_path = joinpath(pwd(),"images","bamboo")
        #train_path = joinpath(pwd(), "images","Brain2","all");
        
        Wav_Bas = zeros(L,N,N);
        current = zeros(N);
        for j = 1:L
            for i = 1:N
                current[i] = 1;
                Wav_Bas[j,:,i] = idwt(current,wf,j);
                current[i] = 0;
            end
        end
    
        F_Wav = fft(Wav_Bas,2)#[:,1:round(Int,N/2+1),:];
    
        L2 = maxtransformlevels(N)
        current = zeros(N,N);
        i = 30;
    
        current[i] = 1;
        B2 = idwt(current, wf,L);
        current[i] = 0;
    
    
        #ind1 = rem(i-1,N)+1;
        #ind2 = Int(ceil(i/N));
        #n1 = Int(L2-ceil(log2(ind1))+1);
        #n2 = Int(L2-ceil(log2(ind2))+1);
        #lvl = min(n1,n2,L);
    
        #B = Wav_Bas[ lvl,:,ind1]*Wav_Bas[lvl,:,ind2]';
        #norm(B-B2);
        #F1 = fft(B);
    
        #F3 = rfft(B);
    
        #F2 = F_Wav[ lvl,:,ind1]*transpose(F_Wav[lvl,:,ind2])
    
    
        #if Q == []
            #Loads an image from a given path and performs some basic transformations to it.
            function process_image(path_vec::String,h::Int64,w::Int64)
                coeffs = [zeros(N,N) for i in 1:Threads.nthreads()];
                list = readdir(path_vec)[2:end]
                img = [zeros(N,N) for i in 1:Threads.nthreads()]
                C = [zeros(N,N) for i in 1:Threads.nthreads()]
                p = Progress(length(list), dt=0.5,desc="Loading Images and generating distribution...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
                Threads.@threads for j in list
                        i = Threads.threadid()
                        img[i] = imresize( Gray.(load(joinpath(path_vec,j))),(h,w))
                        img[i] = convert(Matrix{Float64},img[i])
                        img[i] = img[i]./norm(img[i]);
                        C[i] = dwt(img[i],wf,L);
                        #print(size(C[i]))
                        coeffs[i] += abs.(C[i]) .>= th*norm(C[i]);
                        next!(p)
                end
                println("Generating Wavelet distribution complete")
                return coeffs
            end
    
    
    
        mask1 = zeros(N,N);
        mask2 = zeros(N,N);
        
        w1 = FrequencyWeights(vec(d1[1:2*N]));
        ind1 = sample(1:2*N,w1,Int(round(N/R));replace=false);
        mask1[ind1[ind1.<=N],:].= 1;
        mask1[:,ind1[ind1.>=N+1].-N] .= 1;
        w2 = FrequencyWeights(vec(d2[1:2*N]));
        ind2 = sample(1:2*N,w2,Int(round(N/R));replace=false);
        mask2[ind2[ind2.<=N],:].= 1;
        mask2[:,ind2[ind2.>=N+1].-N] .= 1;
    
        #mask1 = mask1';
        #mask2 = mask2';
    
        #R = 12
        #iter = 100;
        sav_var = zeros(3,1);
    
    
        av_psnr1 = 0#test_images(test_path,Dist1,K,R,iter,N,L,wf)
        av_psnr2 = 0#test_images(test_path,Dist2,K,R,iter,N,L,wf)
        av_psnr3 = 0#test_images(test_path,Dist3,K,R,iter,N,L,wf)
    
        #a = 1, r0 = 2 and r = 100
        #Im = load(joinpath(pwd(),"Brain_2","all","Y258.JPG"));
        #Im = load(joinpath(pwd(),"DIV2K","DIV2K_valid_HR","0822.png"));
        #Im = load(joinpath(pwd(),"forest_etc","pred","22.jpg"));
        #Im = load(joinpath(pwd(),"dandelions","dandelion","IMG_1140.jpg"));
        #Im = load(joinpath(pwd(),"images","SINGINGINTHEBRAIN.png"));
        Im = load(joinpath(pwd(),"images","bamboo.jpg"));
        Im = imresize(Gray.(Im)::Array{Gray{N0f8},2},(N,N));
        Im = convert(Array{Float64},Im);
        Im = Im./norm(Im,2);
    
    
    
        A(x) = fft(idwt(x,wf,L))/N;
        At(x) = dwt(ifft(x)*N,wf,L);
    
        δ = 1.0e-7
        μ = 0.2
        y = mask1.*fft(Im)/N;
        im_rec = Nesta_Cont(y,y,mask1,A,At,iter,μ,δ,1.,3);
        im_rec = ifft(im_rec)
        #im_rec = (im_rec .- minimum(im_rec))./(maximum(im_rec) - minimum(im_rec));
        im_rec = im_rec/norm(im_rec);
    
    
        y = mask2.*fft(Im)/N;
        im_rec2 = Nesta_Cont(y,y,mask2,A,At,iter,μ,δ,1.,3);
        im_rec2 = ifft(im_rec2)
        #im_rec = (im_rec .- minimum(im_rec))./(maximum(im_rec) - minimum(im_rec));
        im_rec2 = im_rec2/norm(im_rec2);
    
    
    
    
    
    var_[1,2,lk] = psnr(abs.(im_rec),Im,maximum(abs.(Im)));   
    var_[2,2,lk] = psnr(abs.(im_rec2),Im,maximum(abs.(Im)));   
    
    
    
    fig = Figure(resolution=(1200, 1200))
        lines(fig[1,1],1:length(vec(d1)),vec(d1))
        lines(fig[1,2],1:length(vec(d1)),vec(d2))
        heatmap(fig[2,1],fftshift(mask1),colormap = :grays,
            axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis")
            )
        )
        heatmap(fig[2,2],fftshift(mask2),colormap = :grays,
            axis = (aspect = 1,# xlabel = "x axis", ylabel = "y axis",
            )
        )
        heatmap(fig[2,3],W,colormap = :grays,
            axis = (aspect = 1,# xlabel = "x axis", ylabel = "y axis",
            )
        )
        image(fig[3,1],abs.(im_rec),
            axis = (aspect = 1, 
            title = string( var_[1,2,lk] ))
        )
        image(fig[3,2],abs.(im_rec2),
            axis = (aspect = 1, 
            title = string( var_[2,2,lk] ))
        )
        image(fig[3,3],Im,
            axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis",
            title = "original")
        )
        fig
        #save("dist.pdf", fig) 


        using Wavelets, Hadamard, FFTW
using Makie
using CairoMakie
include("helperfunctions.jl")
using Images
using ImageIO
using ProgressMeter
using StatsBase
using JLD2
using LinearAlgebra
using MAT
#using Shearlab

var_ = zeros(3,2,250);
#for lk = 1:50
lk = 60
# lk 8 for haar 16, lk steps 0.02 starting from 0.05, R 10 Brain
# lk 20 for db4 16, lk steps 0.02 starting from 0.05, R 10 Brain
    #print(lk)
    #Dict_Q = matread("Brain/tumor_18_db4_R58.mat") 
    #Q = Dict_Q["Q"];
    #Q_new = Dict_Q["Q"];
    #Q_max = Q_new;
    #Q_est = Q_new;
    CairoMakie.activate!(type = "png")
    K = 2^10;
    N = Int(sqrt(K));
    th = (0.05+lk*0.02)*sqrt(2*(log(2*K)-log(1/2))/K);
    R=15;
    iter = 20;
    nScales = 4;
    # shearLevels = ceil.((1:nScales)/2)
    # scalingFilter = Shearlab.filt_gen("scaling_shearlet");
    # directionalFilter = Shearlab.filt_gen("directional_shearlet");
    # waveletFilter = Shearlab.mirror(scalingFilter);
    # scalingFilter2 = scalingFilter;
    full = 1;
    scales = 1;


    Im = load(joinpath(pwd(),"images","SINGINGINTHEBRAIN.png"));
    Im = imresize(Gray.(Im)::Array{Gray{N0f8},2},(N,N));
    Im = convert(Array{Float64},Im);
    @time shearletSystem_nopar = Shearlab.getshearletsystem2D(size(Im,1), size(Im,2), nScales);#scales,shearLevels,full,directionalFilter,scalingFilter,0);
    Q = []
    @time coeffs_nopar = Shearlab.sheardec2D( Im, shearletSystem_nopar);
    siz = prod(size(coeffs_nopar));
    @time Xrec_nopar = Shearlab.shearrec2D(coeffs_nopar, shearletSystem_nopar);
    #Q = Q_new;
    #Q_est = Q_new;

    FFTW.set_num_threads(1);
    #train_path = joinpath(pwd(), "Brain" , "Training" , "all");
    #test_path = joinpath(pwd(), "Brain" , "Testing" , "all");
    train_path = joinpath(pwd(), "images","Brain2" , "all");
    test_path = joinpath(pwd(), "images","Brain2" , "all");
    #train_path = joinpath(pwd(), "DIV2K" , "DIV2K_train_HR");
    #test_path = joinpath(pwd(), "DIV2K" , "DIV2K_valid_HR");


    #if Q == []
        #Loads an image from a given path and performs some basic transformations to it.
        function process_image(path_vec::String,h::Int64,w::Int64,siz::Int64)
            coeffs = [zeros(size(coeffs_nopar)) for i in 1:Threads.nthreads()];
            println(size(coeffs[1]))
            list = readdir(path_vec)[2:end]
            img = [zeros(N,N) for i in 1:Threads.nthreads()]
            C = [complex(zeros(size(coeffs_nopar))) for i in 1:Threads.nthreads()]
            println("")
            p = Progress(length(list), dt=0.5,desc="Loading Images and generating distribution...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
            Threads.@threads for j in list
                    i = Threads.threadid()
                    img[i] = imresize( Gray.(load(joinpath(path_vec,j))),(h,w))
                    img[i] = convert(Matrix{Float64},img[i])
                    img[i] = img[i]./norm(img[i]);
                    C[i] = Shearlab.sheardec2D( img[i],shearletSystem_nopar);
                    #print()
                    coeffs[i] += abs.(C[i]) ;#.>= sort(vec(abs.(C[i])))[end-Int(round(1000))];#th*norm(C[i]);#
                    next!(p)
            end
            println("Generating Wavelet distribution complete")
            return coeffs
        end
        #Processes all images
        coeffs =  process_image(train_path,N,N,siz);
        coeff = abs.(sum(coeffs));
        Nmax = length(readdir(train_path)[2:end]);
        S_av = round(sum(sum(coeff)/Nmax))

        W = sqrt.(coeff./Nmax);
        R_est = round(K/(S_av*log(K)));
            
        print(R_est)
        print(R)
        if R_est > R
            print("u should decrease the threshold")
        else
            print("m_est is smaller than m")
        end



        #Q = [zeros(round(Int,N/2+1),N) for i = 1:Threads.nthreads()];
        #Q_current = [zeros(round(Int,N/2+1),N) for i = 1:Threads.nthreads()];
        #Q_current2 = [complex(zeros(round(Int,N/2+1),N)) for i = 1:Threads.nthreads()];
        #Q_est = [zeros(round(Int,N/2+1),N) for i = 1:Threads.nthreads()];

        Q = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
        Q_current = [zeros(N,N) for i = 1:Threads.nthreads()];
        Q_current2 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
        Q_est = [zeros(N,N) for i = 1:Threads.nthreads()];

        current = [zeros(size(W)) for i = 1:Threads.nthreads()];
        B = [zeros(N,N) for i = 1:Threads.nthreads()];
        n1 = ones(Int64,1,Threads.nthreads());
        n2 = ones(Int64,1,Threads.nthreads());
        lvl = ones(Int64,1,Threads.nthreads());
        ind1 = ones(Int64,1,Threads.nthreads());
        ind2 = ones(Int64,1,Threads.nthreads());
        p = Progress(sum(sum(W.>0)), dt=0.5,desc="Generating sampling distribution...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);

        #U = zeros(K,K);
        plan = plan_fft(zeros(N,N); flags=FFTW.EXHAUSTIVE, timelimit=Inf);
        L2 = maxtransformlevels(N);

        Threads.@threads for j in findall(W.>0)
            i = Threads.threadid();
            current[i][j] = 1;
            B[i] = Shearlab.shearrec2D(current[i], shearletSystem_nopar);
            Q_current2[i] = plan*B[i];#/norm(B[i]);
            current[i][j] = 0;
            #@inline Q_current[i] .= (abs.(plan*B[i])).^2;
            #@inline Q[i] .+= Q_current[i].*W[j].^2;
            @inline Q[i] .+= abs.(Q_current2[i]).^2 .*abs.(coeffs_nopar)[j].^2;
            @inline Q_est[i] .= max.(Q_est[i] , abs.(Q_current2[i]).^2) ;
            next!(p)
            #print(j)
            #U[:,j] = vec(abs.(Q_current2[i]).^2 );
        end  

        Q = abs.(sum(Q));
        Q_max = abs.(sum(Q_est));
        Q_est = Q.*Q_max;

        Q = Q./sum(sum(Q));
        Q_max = Q_max./sum(sum(Q_max));
        Q_est = Q_est./sum(sum(Q_est));




        S_av = sum(sum(W.^2));
        R_est = round(K/(S_av*log(K)));
                
        println(R_est)
        println(R)
        if R_est > R
            println("u should decrease the threshold")
        else
            println("m_est is smaller than m")
        end

    #end


    Dist1 = zeros(N,N);
    #Dist1[1:Int(N/2+1),:] = Q;
    #Dist1[Int(N/2+2):end,:] = Q[end-1:-1:2,:];
    Dist1 = sqrt.(Q);
    Dist1 = Dist1./sum(sum(Dist1));

    Dist2 = zeros(N,N);
    #Dist2[1:Int(N/2+1),:] = Q_est;
    #Dist2[Int(N/2+2):end,:] = Q_est[end-1:-1:2,:];
    Dist2 = Q_est;
    Dist2 = Dist2./sum(sum(Dist2));


    Dist3 = zeros(N,N);
    #Dist3[1:Int(N/2+1),:] = Q_max;
    #Dist3[Int(N/2+2):end,:] = Q_max[end-1:-1:2,:];
    #Dist1 = Q;
    #Dist2 = Q_est;
    #Dist3 = Q_max;

    Dist3 = zeros(N,N);
    for i = 1:N
        for j = 1:N
            Dist3[i,j] = 1/max(1,(i-N/2)^2 + (j-N/2)^2)^(2.5);
        end
    end
    Dist3 = (fftshift(Dist3));

    sav_var = zeros(3,1);
    #a = 1, r0 = 2 and r = 100
    #Im = load(joinpath(pwd(),"Brain_2","all","Y258.JPG"));
    #Im = load(joinpath(pwd(),"DIV2K","DIV2K_train_HR","0016.png"));
    Im = load(joinpath(pwd(),"images","SINGINGINTHEBRAIN.png"));
    Im = imresize(Gray.(Im)::Array{Gray{N0f8},2},(N,N));
    Im = convert(Array{Float64},Im);
    Im = Im/norm(Im);

    function T(x,μ::Float64)
        return(x./(max.(abs.(x),μ)))
    end
    
    
    function Nesta(y,z₀,mask,A,At,niter,μ::Float64=0.2, η::Float64=1.,c::Number=1.)
        #z₀ = At(y);
        # Ensure that A is normalised
        q₂ = copy(z₀);
        z = copy(z₀);
        p = Progress(niter, dt=0.5,desc = "l1 - minimisation...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
        for i = 1:niter 
            Tₙ = μ * A(T(At(z),μ));
            q = z - Tₙ;
            λ = max(0, η^(-1) * norm(y - √c * mask.*q) -1 );
            #println(λ)
            xₙ = ( λ / √c * mask.*y + q ) - λ/((λ + 1)*c)* (λ / √c * mask.*y+mask.*q ) ;
            q₂ .-= (i+1)/2 * Tₙ;
            λ = max(0, η^(-1) * norm(y - √c * mask.*q₂) -1 );
            #println(λ)
            vₙ = ( λ / √c * mask.*y + q₂ ) - λ/((λ + 1)*c)* (λ / √c * mask.*y+mask.*q₂ ) ;
            z = 2/(i+3)  * vₙ + (1 - 2/(i+3) ) * xₙ;
            #println(norm(z))
            next!(p)
        end
        return(z)
    end


    w1 = FrequencyWeights(vec(reshape(Dist1,1,K)));
    ind1 = sample(1:K,w1,Int(round(K/R));replace=false);
    #ind1 = zeros(Int64,Int(round(K/R)),1);
    #StatsBase.naive_wsample_norep!(1:K,w1,ind1);
    mask1 = zeros(N,N);
    mask1[ind1] .= 1;
    
    w2 = FrequencyWeights(vec(reshape(Dist2,1,K)));
    ind2 = sample(1:K,w2,Int(round(K/R));replace=false);
    mask2 = zeros(N,N);
    mask2[ind2] .= 1;
    
    w3 = FrequencyWeights(vec(reshape(Dist3,1,K)));
    ind3 = sample(1:K,w3,Int(round(K/R));replace=false);
    mask3= zeros(N,N);
    mask3[ind3] .= 1;

A(x) = fft(Shearlab.shearrec2D(reshape(x,size(W)),shearletSystem_nopar))/N;
At(x) = Shearlab.sheardec2D(ifft(x)*N,shearletSystem_nopar); 
iter = 10
δ = 0.000001
y = mask1.*fft(Im)/N;
im_rec = Nesta(y,y,mask1,A,At,iter,0.02,δ,1.);
im_rec = Nesta(y,im_rec, mask1,A,At,iter,0.000002,δ,1.);
#im_rec = Nesta(y,im_rec,mask1,A,At,iter,0.002,δ,1.);
#im_rec = Nesta(y,im_rec,mask1,A,At,20,0.0002,δ,1.);
#im_rec = Nesta(y,im_rec,mask1,A,At,20,0.00002,δ,1.);
im_rec = abs.(ifft(im_rec))
#im_rec = (im_rec .- minimum(im_rec))./(maximum(im_rec) - minimum(im_rec));
im_rec = im_rec/norm(im_rec);



y = mask3.*fft(Im)/N;
#iter = 10;
im_rec3 = Nesta(y,y,mask3,A,At,iter,0.2,δ,1.);
im_rec3 = Nesta(y,im_rec3, mask3,A,At,iter,0.00002,δ,1.);
#im_rec3 = Nesta(y,im_rec3,mask3,A,At,iter,0.002,δ,1.);
#im_rec3 = Nesta(y,im_rec3,mask3,A,At,iter,0.0002,δ,1.);
#im_rec3 = Nesta(y,im_rec3,mask3,A,At,20,0.00002,δ,1.);
im_rec3 = abs.(ifft(im_rec3))
#im_rec = (im_rec .- minimum(im_rec))./(maximum(im_rec) - minimum(im_rec));
im_rec3 = im_rec3/norm(im_rec3);


var_[2,2,lk] = psnr(abs.(im_rec),Im,maximum(abs.(Im)))
var_[3,2,lk] = psnr(abs.(im_rec3),Im,maximum(abs.(Im))) 
println("")
println(var_[2,2,lk])
println(var_[3,2,lk])


plotss = 1;
if plotss == 1
    fig = Figure(resolution=(1200, 1200))
        contour(fig[1,1],fftshift(log.(Dist1)),
        levels=-20:1:1,
        axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis", 
        colormap = :grays, title=string(1))
        )
        #Colorbar(fig[1,2],limits = [minimum(log.(Dist1)),maximum(log.(Dist1))])
        contour(fig[1,2],fftshift(log.(Dist2)),
        levels=-20:1:1,
            axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis", 
            colormap = :grays, title=string(1))
        )
        contour(fig[1,3],fftshift(log.(Dist3)),
        levels=-20:1:1,
            axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis",
            colormap = :grays, title=string(1))
        )
        heatmap(fig[2,1],fftshift(mask1),colormap = :grays,
            figure = (backgroundcolor = :pink,),
            axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis")
            colormap = :grays)
        )
        heatmap(fig[2,2],fftshift(mask2),colormap = :grays,
            figure = (backgroundcolor = :pink,),
            axis = (aspect = 1,# xlabel = "x axis", ylabel = "y axis",
            colormap = :grays)
        )
        heatmap(fig[2,3],fftshift(mask3),colormap = :grays,
            figure = (backgroundcolor = :pink,),
            axis = (aspect = 1,# xlabel = "x axis", ylabel = "y axis",
            colormap = :grays)
        )
        image(fig[3,2],abs.(im_rec),
            axis = (aspect = 1, 
            title = string( var_[2,2,lk] ))
        )
        image(fig[3,3],abs.(im_rec3),
            axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis",
            title =  string( var_[3,2,lk] ))
        )
        image(fig[3,4],Im,
            figure = (backgroundcolor = :pink,),
            axis = (aspect = 1, #xlabel = "x axis", ylabel = "y axis",
            title = "original")
        )
        fig
        #save("dist.pdf", fig) 


end




Q = []
@load "DIV2K_20_db4_th03_L5.jld2"

K = 2^14;
N = Int(sqrt(K));
th = 0.3*sqrt(2*(log(2*K)-log(1/2))/K);
R=10;
runs = 1;
wf = wavelet(WT.db4, WT.Filter);
L =  min(5,maxtransformlevels(N))

FFTW.set_num_threads(1);
#img_path = joinpath(pwd(), "archive" , "Training" , "all");
img_path = joinpath(pwd(), "DIV2K" , "DIV2K_train_HR");

Wav_Bas = zeros(L,N,N);
current = zeros(N);
for j = 1:L
    for i = 1:N
        current[i] = 1;
        Wav_Bas[j,:,i] = idwt(current,wf,j);
        current[i] = 0;
    end
end
N = 6;
Four_Bas = complex(zeros(N,N));
current = zeros(N);

for i = 1:N
    current[i] = 1;
    Four_Bas[:,i] = ifft(current);
    current[i] = 0;
end


F_Wav = fft(Wav_Bas,2)[:,1:round(Int,N/2+1),:];

L2 = maxtransformlevels(N)
current = zeros(N,N);
i = 30;

current[i] = 1;
B2 = idwt(current, wf,L);
current[i] = 0;


ind1 = rem(i-1,N)+1;
ind2 = Int(ceil(i/N));
n1 = Int(L2-ceil(log2(ind1))+1);
n2 = Int(L2-ceil(log2(ind2))+1);
lvl = min(n1,n2,L);

B = Wav_Bas[ lvl,:,ind1]*Wav_Bas[lvl,:,ind2]';
norm(B-B2);
F1 = fft(B);

F3 = rfft(B);

F2 = F_Wav[ lvl,:,ind1]*transpose(F_Wav[lvl,:,ind2])

function nabla(x)
    G=zeros(size(x,1),size(x,2),2);
    G[1:size(x,1),1:size(x,2)-1,1]-=x[1:size(x,1),1:size(x,2)-1]
    G[1:size(x,1),1:size(x,2)-1,1]+=x[1:size(x,1),2:size(x,2)]
    #G[1:size(x,1)-1,1:size(x,2),2]-=x[1:size(x,1)-1,1:size(x,2)]
    #G[1:size(x,1)-1,1:size(x,2),2]+=x[2:size(x,1),1:size(x,2)]
    return G
end

function nablaT(G)
    x=zeros(size(G,1),size(G,2))
    x[1:size(x,1),1:size(x,2)-1]-=G[1:size(x,1),1:size(x,2)-1,1]
    x[1:size(x,1),2:size(x,2)]-=G[1:size(x,1),1:size(x,2)-1,1]
    #x[1:size(x,1)-1,1:size(x,2)]-=G[1:size(x,1)-1,1:size(x,2),2]
    #x[2:size(x,1),1:size(x,2)]+=G[1:size(x,1)-1,1:size(x,2),2]
    return x
end

#if Q == []
    #Loads an image from a given path and performs some basic transformations to it.
    function process_image(path_vec::String,h::Int64,w::Int64)
        coeffs = [zeros(N,N,2) for i in 1:Threads.nthreads()];
        list = readdir(path_vec)[2:end]
        img = [zeros(N,N) for i in 1:Threads.nthreads()]
        C = [zeros(N,N,2) for i in 1:Threads.nthreads()]
        p = Progress(length(list), dt=0.5,desc="Loading Images and generating distribution...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
        Threads.@threads for j in list
                i = 1
                img[i] = imresize( Gray.(load(joinpath(path_vec,j))),(h,w))
                img[i] = convert(Matrix{Float64},img[i])
                img[i] = img[i]./norm(img[i]);
                C[i] = nabla(img[i]);#dwt(img[i],wf,L);
                #print(size(C[i]))
                coeffs[i] += abs.(C[i]) .>= th*norm(C[i]);
                next!(p)
        end
        print("Generating Wavelet distribution complete")
        return coeffs
    end


    #Processes all images
    coeffs =  process_image(img_path,N,N);
    coeff = sum(coeffs);
    Nmax = length(readdir(img_path)[2:end]);
    S_av = round(sum(sum(coeff)/Nmax))



    W = sqrt.(coeff./Nmax);
    R_est = round(K/(S_av*log(K)));
        
    print(R_est)
    print(R)
    if R_est > R
        print("u should decrease the threshold")
    else
        print("m_est is smaller than m")
    end



    Q = [zeros(round(Int,N/2+1),N) for i = 1:Threads.nthreads()];
    Q_current = [zeros(round(Int,N/2+1),N) for i = 1:Threads.nthreads()];
    Q_current2 = [complex(zeros(round(Int,N/2+1),N)) for i = 1:Threads.nthreads()];
    Q_est = [zeros(round(Int,N/2+1),N) for i = 1:Threads.nthreads()];

    #Q = [zeros(N,N) for i = 1:Threads.nthreads()];
    #Q_current = [zeros(N,N) for i = 1:Threads.nthreads()];
    #Q_current2 = [zeros(N,N) for i = 1:Threads.nthreads()];
    #Q_est = [zeros(N,N) for i = 1:Threads.nthreads()];

    current = [zeros(size(W)) for i = 1:Threads.nthreads()];
    B = [zeros(N,N) for i = 1:Threads.nthreads()];
    n1 = ones(Int64,1,Threads.nthreads());
    n2 = ones(Int64,1,Threads.nthreads());
    lvl = ones(Int64,1,Threads.nthreads());
    ind1 = ones(Int64,1,Threads.nthreads());
    ind2 = ones(Int64,1,Threads.nthreads());
    p = Progress(K, dt=0.5,desc="Generating sampling distribution...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);


    plan = plan_rfft(zeros(N,N); flags=FFTW.EXHAUSTIVE, timelimit=Inf);
    L2 = maxtransformlevels(N);
    lk = 0
    Threads.@threads for j in eachindex(W)
        i = Threads.threadid();
        current[i][j] = 1;
        ind1[i] = rem(j-1,N)+1;
        ind2[i] = Int(ceil(j/N));
        n1[i] = Int(L2-ceil(log2(ind1[i]))+1);
        n2[i] = Int(L2-ceil(log2(ind2[i]))+1);
        lvl[i] = min(n1[i],n2[i],L);
        

        #Q_current2[i] = F_Wav[ lvl[i],:,ind1[i]]*transpose(F_Wav[lvl[i],:,ind2[i]]);
        #@inbounds @inline B[i] = get_Bas_elem(ind1[i],ind2[i],n1[i],n2[i],lvl[i],j,L2)
        #if norm(idwt(current[i],wf,L)-B[i])>0.01
        #    print("fail")
        #end#Wav_Bas(:,ind1)*Wav_Bas(:,ind2)'
        #B[i] = Wav_Bas[ lvl[i],:,ind1[i]]*Wav_Bas[lvl[i],:,ind2[i]]';
        Q_current2[i] = plan*nablaT(current[i]);
        #if (maximum(abs.((plan*B[i])[1:round(Int,N/2+1),round(Int,N/2+2):end]) - abs.(Q_current2[i][:,end-1:-1:2,]))>0.000001)
        #    println("fail")
        #end
        current[i][j] = 0;
        #@inline Q_current[i] .= (abs.(plan*B[i])).^2;
        #@inline Q[i] .+= Q_current[i].*W[j].^2;
        @inline Q[i] .+= abs.(Q_current2[i]).^2 .*W[j].^2;
        @inline Q_est[i] .= max.(Q_est[i] , abs.(Q_current2[i]).^2) ;
        next!(p)
    end  


function get_Bas_elem(ind1,ind2,n1,n2,lvl,j,L2)
    ind1 = rem(j-1,N)+1;
    ind2 = Int(ceil(j/N));
    n1 = Int(L2-ceil(log2(ind1))+1);
    n2 = Int(L2-ceil(log2(ind2))+1);
    lvl = min(n1,n2,L);
    return Wav_Bas[ lvl,:,ind1]*Wav_Bas[lvl,:,ind2]';
end

function gen_dist_lines(N,L,K,Wav_Bas,W);
    
    Q_current2 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
    

    Q1 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
    Q2 = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
    #
    Q_current = [complex(zeros(N,N)) for i = 1:Threads.nthreads()];
    #Q_current2 = [zeros(N,N) for i = 1:Threads.nthreads()];
    #Q_est = [zeros(N,N) for i = 1:Threads.nthreads()];

    #current = [zeros(N,N) for i = 1:Threads.nthreads()];
    B = [zeros(N,N) for i = 1:Threads.nthreads()];
    n1 = ones(Int64,1,Threads.nthreads());
    n2 = ones(Int64,1,Threads.nthreads());
    lvl = ones(Int64,1,Threads.nthreads());
    ind1 = ones(Int64,1,Threads.nthreads());
    ind2 = ones(Int64,1,Threads.nthreads());
    p = Progress(K, dt=0.5,desc="Generating sampling distribution...",  barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);

    plan = plan_rfft(zeros(N,N); flags=FFTW.EXHAUSTIVE, timelimit=Inf);
    L2 = maxtransformlevels(N);
    pp = zeros(round(Int,N/2+1),1);
    pp2 = zeros(round(Int,N/2+1),1);
    @showprogress for l = 1:round(Int,N/2+1)
        Q_current2 = [complex(zeros(K,N)) for i = 1:Threads.nthreads()];
        
        Q_current = [complex(zeros(K,N)) for i = 1:Threads.nthreads()];
        
        Threads.@threads for j = 1:K
            i = Threads.threadid();
            ind1[i] = rem(j-1,N)+1;
            ind2[i] = Int(ceil(j/N));
            n1[i] = Int(L2-ceil(log2(ind1[i]))+1);
            n2[i] = Int(L2-ceil(log2(ind2[i]))+1);
            lvl[i] = min(n1[i],n2[i],L);
            Q1[i] = F_Wav[lvl[i],:,ind1[i]]*transpose(F_Wav[lvl[i],:,ind1[i]]);
            Q_current2[i] += W[j].^2* F_Wav[ lvl[i],l,ind2[i]].^2 .*Q1[i];
        #    Q_current2[i] += W[j].^2 .* kron(F_Wav[ lvl[i],:,ind2[i]].^2 ,Q1[i]);
            Q2[i] = F_Wav[lvl[i],:,ind2[i]]*transpose(F_Wav[lvl[i],:,ind2[i]]);
        #    Q_current[i] += W[j].^2 .* kron(F_Wav[ lvl[i],:,ind1[i]].^2,Q2[i]);
            Q_current[i] += W[j].^2* F_Wav[ lvl[i],l,ind1[i]].^2 .*Q2[i];
            #if (maximum(abs.((plan*B[i])[1:round(Int,N/2+1),round(Int,N/2+2):end]) - abs.(Q_current2[i][:,end-1:-1:2,]))>0.000001)
            #    println("fail")
            #end
        #    next!(p)
        end 
    pp[l] = opnorm(sum(Q_current)-Diagonal(sum(Q_current))) ;
    pp2[l] = opnorm(sum(Q_current2)-Diagonal(sum(Q_current2))) ;
    #

    end

    ppp = zeros(N,1);
    ppp[round(Int,N/2):end] = pp;
    ppp[1:round(Int,N/2-1)] = pp[end-1:-1:2];
    
    ppp2 = zeros(N,1);
    ppp2[round(Int,N/2):end] = pp2;
    ppp2[1:round(Int,N/2-1)] = pp2[end-1:-1:2];
    
    fig = Figure()
    lines(fig[1,1],1:N,vec(log.(ppp)))
    lines(fig[1,2],1:N,vec(log.(ppp2)))
    fig    
end




Q = abs.(sum(Q));
Q_max = abs.(sum(Q_est));
Q_est = Q.* Q_max;

Q = Q./sum(sum(Q));
Q_max = Q_max./sum(sum(Q_max));
Q_est = Q_est./sum(sum(Q_est));

#end

S_av = sum(sum(W.^2));
R_est = round(K/(S_av*log(K)));
        
print(R_est)
print(R)
if R_est > R
    print("u should decrease the threshold")
else
    print("m_est is smaller than m")
end

R = R_est
cmaximum = maximum(maximum(maximum(log.(Q))));
cminimum = minimum(minimum(minimum(log.(Q))));
#plot(contour(1:N,1:N, Q))

Dist1 = zeros(N,N);
Dist1[1:Int(N/2+1),:] = Q;
Dist1[Int(N/2+2):end,:] = Q[end-1:-1:2,:];
Dist1 = Dist1./sum(sum(Dist1));


Dist2 = zeros(N,N);
Dist2[1:Int(N/2+1),:] = Q_est;
Dist2[Int(N/2+2):end,:] = Q_est[end-1:-1:2,:];

Dist3 = zeros(N,N);
Dist3[1:Int(N/2+1),:] = Q_max;
Dist3[Int(N/2+2):end,:] = Q_max[end-1:-1:2,:];
#Dist1 = Q;
#Dist2 = Q_est;
#Dist3 = Q_max;

Dist3 = zeros(N,N);
for i = 1:N
    for j = 1:N
        Dist3[i,j] = 1/max(1,(i-N/2)^2 + (j-N/2)^2)^2;
    end
end
Dist3 = fftshift(Dist3);

#p = plot(plot(contour(z =fftshift(log.(Dist1)))),plot(contour(z =fftshift(log.(Dist2)))),plot(contour(z =fftshift(log.(Dist3)))))
#using Colors, Plots
#plot(fftshift(log.(Dist1)))
#plotlyjs()
#contour(1:N, 1:N, fftshift(log.(Dist1)), xlabel="x", ylabel="y", fill=true, aspect_ratio=:equal)
#contour(1:N, 1:N, fftshift(log.(Dist2)), xlabel="x", ylabel="y", fill=true, aspect_ratio=:equal)
#contour(1:N, 1:N, fftshift(log.(Dist3)), xlabel="x", ylabel="y", fill=true, aspect_ratio=:equal)
#I = load(joinpath(pwd(),"DIV2K","DIV2K_train_HR","0015.png"));
Im = load(joinpath(pwd(),"images","brain.png"));
Im = imresize(Gray.(Im)::Array{Gray{N0f8},2},(N,N));
Im = convert(Array{Float64},Im);
CairoMakie.activate!(type = "png")
#im = image(I)
#save("normal.pdf", im) 
K_space = fft(Im)/N;



R  = 20;

#FFTW.set_num_threads(8);
#plan = plan_fft(zeros(N,N); flags=FFTW.EXHAUSTIVE, timelimit=Inf);


w1 = FrequencyWeights(vec(reshape(Dist1,1,K)));
ind1 = sample(1:K,w1,Int(round(K/R));replace=false)

mask1 = zeros(N,N);
mask1[ind1] .= 1;

DATA1 = mask1.*K_space;

#f1(x) = mask1.*fft(idwt(reshape(x,N,N),wf,L))/N;
#f1t(x) = dwt(ifft(x)*N,wf,L);
f1(x) = mask1.*fft(nablaT(reshape(x,N,N,2)))/N;
f1t(x) = nabla(ifft(x)*N);
w2 = FrequencyWeights(vec(reshape(Dist2,1,K)));
ind2 = sample(1:K,w2,Int(round(K/R));replace=false)
mask2 = zeros(N,N);
mask2[ind2] .= 1;
DATA2 = mask2.*K_space;
#f2(x) = mask2.*fft(idwt(reshape(x,N,N),wf,L))/N;
#f2t(x) = dwt(ifft(x),wf,L)*N;
f2(x) = mask2.*fft(nablaT(reshape(x,N,N,2)))/N;
f2t(x) = nabla(ifft(x)*N);
w3 = FrequencyWeights(vec(reshape(Dist3,1,K)));
ind3 = sample(1:K,w3,Int(round(K/R));replace=false)
mask3 = zeros(N,N);
mask3[ind3] .= 1;
DATA3 = mask3.*K_space;
#f3(x) = mask3.*fft(idwt(reshape(x,N,N),wf,L))/N;
#f3t(x) = dwt(ifft(x),wf,L)*N;
f3(x) = mask3.*fft(nablaT(reshape(x,N,N,2)))/N;
f3t(x) = nabla(ifft(x)*N);



iter = 400;
sav_var = zeros(3,1);

function Solve_l1_problemDR(y,A,At,niter)
    Norm_y=1;
    gamma=.05; 
    Norm_y=norm(y[:]);
    y=y/Norm_y;
    x = zeros(size(At(y)));
    ze = zeros(size(x));
    on = ones(size(x));
    Prox_l1(x,tau) = max.(ze,on-tau./max.(1e-15,abs.(x))).*x;
    Proj_set(x) = x + At(y-A(x));
    #Parameters of Douglas Rachford 
    lambda=1.5;
    z=zeros(size(At(y)));
    #L1=zeros(1,niter);
    p = Progress(niter, dt=0.5,desc = "l1 - minimisation...", barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:black);
    for i=1:niter
        x = Proj_set(z);
        z += lambda.*(Prox_l1(2*x-z,gamma)-x);
        next!(p)    
    end
    #print(x)
    #figure, plot(L1)
return x*Norm_y;
end


rec1=Solve_l1_problemDR(DATA1,f1,f1t,30);
rec2=Solve_l1_problemDR(DATA2,f2,f2t,iter);
rec3=Solve_l1_problemDR(DATA3,f3,f3t,iter);
#using ImageQualityIndexes

#println(psnr(abs.(idwt(rec1,wf,L)),I,max(maximum(abs.(idwt(rec1,wf,L))),maximum(abs.(I)))))
#println(psnr(abs.(idwt(rec2,wf,L)), I,max(maximum(abs.(idwt(rec2,wf,L))),maximum(abs.(I)))))
#println(psnr(abs.(idwt(rec3,wf,L)), I,max(maximum(abs.(idwt(rec3,wf,L))),maximum(abs.(I)))))

println(psnr(abs.(nablaT(rec1)),Im,max(maximum(abs.(nablaT(rec1))),maximum(abs.(Im)))))
println(psnr(abs.(nablaT(rec2)), Im,max(maximum(abs.(nablaT(rec2))),maximum(abs.(Im)))))
println(psnr(abs.(nablaT(rec3)), Im,max(maximum(abs.(nablaT(rec3))),maximum(abs.(Im)))))



#a = 1, r0 = 2 and r = 100

fig = Figure()
#image(fig[1,1],abs.(idwt(rec1,wf,L)),
##    figure = (backgroundcolor = :pink,),
 #   axis = (aspect = 1, xlabel = "x axis", ylabel = "y axis")
#)
image(fig[1,1],abs.(nablaT(rec1)),
    figure = (backgroundcolor = :pink,),
    axis = (aspect = 1, xlabel = "x axis", ylabel = "y axis")
)
image(fig[1,2],abs.(idwt(rec2,wf,L)),
    figure = (backgroundcolor = :pink,),
    axis = (aspect = 1, xlabel = "x axis", ylabel = "y axis")
)
image(fig[2,1],abs.(idwt(rec3,wf,L)),
    figure = (backgroundcolor = :pink,),
    axis = (aspect = 1, xlabel = "x axis", ylabel = "y axis")
)
image(fig[2,2],Im,
    figure = (backgroundcolor = :pink,),
    axis = (aspect = 1, xlabel = "x axis", ylabel = "y axis")
)
fig


fig = Figure()

#ax, contourplot1 = contour(fig[1,1],fftshift(log.(abs.(Dist1-Dist1_matlab))))
#Colorbar(fig[1,2], contourplot1, width=25)
#fig

contour(fig[1,1],fftshift(log.(Dist1)),
    figure = (backgroundcolor = :pink,),
    axis = (aspect = 1, xlabel = "x axis", ylabel = "y axis")
)
#Colorbar(fig[1,2],limits = [minimum(log.(Dist1)),maximum(log.(Dist1))])
#contour(fig[1,3],fftshift(log.(Dist1_matlab)),
#    figure = (backgroundcolor = :pink,),
#    axis = (aspect = 1, xlabel = "x axis", ylabel = "y axis")
#)
#Colorbar(fig[1,4],limits = [minimum(log.(Dist1_matlab)),maximum(log.(Dist1))])
#fig
heatmap(fig[1,2],fftshift(log.(Dist2)),
    figure = (backgroundcolor = :pink,),
    axis = (aspect = 1, xlabel = "x axis", ylabel = "y axis")
)
heatmap(fig[1,3],fftshift(log.(Dist3)),
    figure = (backgroundcolor = :pink,),
    axis = (aspect = 1, xlabel = "x axis", ylabel = "y axis",
    colormap = :grays)
)
heatmap(fig[2,1],fftshift(mask1),
    figure = (backgroundcolor = :pink,),
    axis = (aspect = 1, xlabel = "x axis", ylabel = "y axis")
)
heatmap(fig[2,2],fftshift(mask2),
    figure = (backgroundcolor = :pink,),
    axis = (aspect = 1, xlabel = "x axis", ylabel = "y axis",
    colormap = :grays)
)
heatmap(fig[2,3],fftshift(mask3),
    figure = (backgroundcolor = :pink,),
    axis = (aspect = 1, xlabel = "x axis", ylabel = "y axis",
    colormap = :grays)
)
fig
#save("dist.pdf", fig) 

