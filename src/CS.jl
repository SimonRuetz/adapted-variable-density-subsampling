# export run_tests

#workspace()
using Revise, Wavelets, Hadamard, FFTW, CairoMakie, FileIO, Images, ProgressMeter, StatsBase, LinearAlgebra, Infiltrator, NPZ

include("helperfunctions.jl")
include("main.jl")

# end # module CS
# wavelet_test_2()