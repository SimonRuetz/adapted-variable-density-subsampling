using Revise, Wavelets, Hadamard, FFTW, CairoMakie, FileIO, Images, ProgressMeter, StatsBase, LinearAlgebra, Infiltrator, NPZ

include("helperfunctions.jl")


# replace the following lines with the path to your data
train_path_brains = joinpath(pwd(), "images", "Brain4")
test_path_brain = joinpath(pwd(), "images", "brain4.tif")

train_path_knees = joinpath(pwd(), "images", "MRNet", "train", "coronal")
test_path_knees = joinpath(pwd(), "images", "MRNet", "valid", "coronal", "1190.npy")


function wavelet_test()
    run_tests(; K=2^16, lk=8.0, flp=0, had=0, lines_and_blocks=0,
        train_path=train_path_brains,
        test_path=test_path_brain,
        wf=wavelet(WT.db4, WT.Filter), R=10, sep=0)
end


function flip_test()
    run_tests(; K=2^16, lk=15.0, flp=1, had=0, lines_and_blocks=0,
        train_path=joinpath(pwd(), "images", "Brain4"),
        test_path=joinpath(pwd(), "images", "brain.png"),
        wf=wavelet(WT.db4, WT.Filter), R=4, sep=0)
end

function knee_test()
    run_tests(; K=2^16, lk=7.0, flp=0, had=0, lines_and_blocks=0,
        train_path=train_path_knees,
        test_path=test_path_knees,
        wf=wavelet(WT.db4, WT.Filter), R=10, sep=0)
end

function line_test()
    run_tests(; K=2^16, lk=59.0, flp=0, had=0, lines_and_blocks=1,
        train_path=train_path_brains,
        test_path=test_path_brain,
        wf=wavelet(WT.db4, WT.Filter), R=5, sep=1)
end


"""
This function runs all tests and returns the average results over `runs` runs.

# Arguments
- `runs::Int`: The number of runs to average over.

# Returns
- `A::Array{Float64, 2}`: A 5x3 matrix containing the average results of the tests.
    The rows correspond to the tests and the columns to the metrics.
    The tests are:
        1. Wavelet test
        2. Knee test
        3. Flip test
        4. Uniform test
        5. Line test
"""
function main(runs=10)
    A = zeros(5, 3)
    for i = 1:runs
        A[1, 1:2] .+= wavelet_test()
        A[2, 1:2] .+= knee_test()
        A[3, 1:2] .+= flip_test()
        A[4, 1:3] .+= uniform_test()
        A[5, 1:2] .+= line_test()
    end
    return A ./runs
end