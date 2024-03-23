
using StatsBase
using Makie
using CairoMakie
using IterTools

trials = 200
D = zeros(trials)
K = 2^15;
S = Int(round(sqrt(K)));

for l = 1:trials
    w = FrequencyWeights(vec(abs.(randn(K)/(2*exp(1)*S))));
    #w = (1:K).^(-1)

    for i = 1:10
        w[w.>1/(exp(1)*S)] .= 1/(exp(1)*S);
        w = w./sum(w);
    end

    w = FrequencyWeights(vec(w./sum(w)));

    println(prod(ones(length(w))-S*w))
end
# function rej(w,I)
#     w2 = copy(w);
#     w2[I] .= 0;
#     prod(w[I])*prod(ones(length(w))-w2)
# end


# function succ(w,I)
#     l = 1;
#     for i = 1:length(I)
#         l = l*w[i]*1/(1-sum(w[I[1:i]]))
#     end
#     return l
# end



# for i = 1:trials
#     I = sample(1:K,FrequencyWeights(ones(1:K)),Int(round(S)));
#     D[i] = K*rej(S*w,I) > succ(w,I)
# end


# # prob = 0;
# # prob_succ = 0;
# # for i in subsets(1:K)
# #     prob += rej(S*w,i);
# #     if length(i) == S

# #         prob_succ += succ(w,i);
# #     end
# # end

# # for i in subsets([1, 2, 3])
# #     if length(i) == 2
# #         @show i
# #     end
# #  end




# println(sum(D) == trials)