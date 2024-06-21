"""
Unit tests
"""

using Test

using Random
using LinearAlgebra

using BlockTensorDecomposition

const VERBOSE = true

@testset verbose=true "BlockTensorDecomposition" begin
    @testset verbose=VERBOSE "Utils" begin
        @testset verbose=VERBOSE "interlace" begin
        @test interlace(1:10,10:15) == [1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 8, 9, 10]
        @test interlace(1:5,10:20) == [1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 15, 16, 17, 18, 19, 20]
        @test interlace(1:3, ("this", "that", "other")) == Any[1, "this", 2, "that", 3, "other"]
        end

        @testset verbose=VERBOSE "norm2" begin
            @test norm2([3, 4]) == 25
            @test norm2(1:10) == sum((1:10) .^ 2)
        end

        @testset verbose=VERBOSE "geomean" begin
            @test geomean(5) == 5
            @test geomean(2, 1/2) == 1
            @test geomean((1,2,3)) ≈ 1.8171205928321397
        end
    end

@testset verbose=VERBOSE "Constraints" begin
    @testset "L1" begin
        v = collect(1:5)
        l1normalize!(v)
        @test v == [0, 0, 0, 0, 1]

        v = Vector{Float64}([1, -1, 1, 1])
        l1normalize!(v)
        @test v == [1, -1, 1, 1] / 4

        v = [1, -1, 1, 1]
        @test_broken l1normalize!(v) # v can only hold Int, so there is an error here

        A = Array{Float64}(reshape(1:12, 3,4))
        l1normalize_rows!(A)
        @test all(A .≈ [0 0 0 1; 0 0 0 1; 0 0 0 1])

        A = Array{Float64}(reshape(1:12, 3,4))
        l1normalize_cols!(A)
        @test all(A .≈ [0 0 0 0; 0 0 0 0; 1 1 1 1])

        A = Array{Float64}(ones(2, 3))
        l1normalize_rows!(A)
        @test all(A .≈ ones(2, 3) / 3)

        A = Array{Float64}(ones(2, 3))
        l1normalize_cols!(A)
        @test all(A .≈ ones(2, 3) / 2)

        A = Array{Float64}(reshape(1:12, 3,2,2))
        l1scaled_1slices!(A)
        @test A ≈ [0.045454545454545456 0.18181818181818182; 0.07692307692307693 0.19230769230769232; 0.1 0.2;;; 0.3181818181818182 0.45454545454545453; 0.3076923076923077 0.4230769230769231; 0.3 0.4]
    end

    @testset "L2" begin
        v = Vector{Float64}([1, 1])
        l2normalize!(v)
        @test v == [1, 1] / sqrt(2)

        v= Vector{Float64}([1, -1, 1, 1])
        l2normalize!(v)
        @test v == [1, -1, 1, 1] / 2

        A = Array{Float64}(ones(2, 3))
        l2normalize_rows!(A)
        @test all(A .≈ ones(2, 3) / √3)

        A = Array{Float64}(ones(2, 3))
        l2normalize_cols!(A)
        @test all(A .≈ ones(2, 3) / √2)
    end

    @testset "Linfinity" begin
        v = collect(1:5)
        linftynormalize!(v)
        @test v == [1, 1, 1, 1, 1]

        v = collect(1:5) / 5
        linftynormalize!(v)
        @test v == (1:5) / 5

        v = [0, 0.1, 0.8, 0.2]
        linftynormalize!(v)
        @test v == [0, 0.1, 1, 0.2]

        v = [0, 0.1, -0.8, 0.2]
        linftynormalize!(v)
        @test v == [0, 0.1, -1, 0.2]

        v = [0, 0.1, -0.8, 0.2]
        linftynormalize!(v)
        @test v == [0, 0.1, -1, 0.2]

        A = Array{Float64}(reshape(1:12, 3,4))
        l1normalize_rows!(A)
        @test A == [0 0 0 1; 0 0 0 1; 0 0 0 1]

        A = Array{Float64}(reshape(1:12, 3,4))
        l1normalize_cols!(A)
        @test A == [0 0 0 0; 0 0 0 0; 1 1 1 1]
    end
end

@testset verbose=VERBOSE "SuperDiagonal" begin
    v = 1:10
    S = SuperDiagonal(v, 2)
    @test diag(S) === v
    @test size(S) == (10, 10)
    @test ndims(S) == 2
    @test eltype(S) == Int
    @test_throws BoundsError S[10, 11]
    @test_throws BoundsError S[10, 0]
    @test S[9,9] == 9
    @test S[4,6] == 0
    @test array(S) == Diagonal(v)
    @test eltype(array(S)) == Int

    S = SuperDiagonal(v, 3)
    @test diag(S) === v
    @test size(S) == (10, 10, 10)
    @test ndims(S) == 3

    @test size(array(S)) == (10, 10, 10)

end

@testset verbose=VERBOSE "AbstractDecomposition" begin

    A = randn(3,3);
    B = randn(4,3);
    C = randn(5,3);

    G = CPDecomposition((A, B, C))

    @testset verbose=VERBOSE "Copying" begin
        G_copy = copy(G)
        G_deepcopy = deepcopy(G)
        @test A == matrix_factor(G,1)
        @test A === matrix_factor(G,1)
        @test A == matrix_factor(G_copy, 1)
        @test !(A === matrix_factor(G_copy,1))
        @test A == matrix_factor(G_deepcopy, 1)
        @test !(A === matrix_factor(G_deepcopy,1))
    end

    G = CPDecomposition((A, B))

    G = Tucker((randn(3,3,3), A, B, C))

    @test rankof(G) == (3,3,3)



    G = Tucker1(([1 2]', [3 4]))
    @test size(G) == (1, 1)
    @test_throws MethodError Tucker1{Int, 3}(([1 2]', [3 4])) # missing the freeze argument
    @test_throws ArgumentError Tucker1{Int, 3}(([1 2]', [3 4]), (false, false)) # the 3 in Tucker1{Int, 3} should be 2 since this combines to a 1×1 matrix

    G = Tucker1((randn(3,3,3), A))

    G = Tucker1((B', A)) # Can it handle types that are an abstract matrix like Ajoint

    @test_throws ArgumentError Tucker((G, A)) # Can handle auto conversion to TuckerN in the future

    G = Tucker1((10,11,12), 5);
    Y = Tucker1((10,11,12), 5; init=abs_randn); # check if other initilizations work

    @test isfrozen(G, 0) == false # the core is not frozen
    @test isfrozen(G, 1) == false # the matrix factor A is not frozen

    # CPDecomposition test
    A = reshape(1:6, 3, 2)
    B = reshape(1:8, 4, 2)
    C = reshape(1:4, 2, 2)
    D = reshape(1:10, 5, 2)

    G = CPDecomposition((A, B, C, D))

    @test isfrozen(G, 0) == true # the core is frozen
    @test isfrozen(G, 1) == false # the matrix factor A is not frozen

    T = [361 434 507 580; 452 544 636 728; 543 654 765 876;;; 482 580 678 776; 604 728 852 976; 726 876 1026 1176;;;; 422 508 594 680; 529 638 747 856; 636 768 900 1032;;; 564 680 796 912; 708 856 1004 1152; 852 1032 1212 1392;;;; 483 582 681 780; 606 732 858 984; 729 882 1035 1188;;; 646 780 914 1048; 812 984 1156 1328; 978 1188 1398 1608;;;; 544 656 768 880; 683 826 969 1112; 822 996 1170 1344;;; 728 880 1032 1184; 916 1112 1308 1504; 1104 1344 1584 1824;;;; 605 730 855 980; 760 920 1080 1240; 915 1110 1305 1500;;; 810 980 1150 1320; 1020 1240 1460 1680; 1230 1500 1770 2040]

    @test array(G) == T
    @test ndims(G) == 4
    @test rankof(G) == 2
    @test all(diag(core(G)) .== 1)
    @test length(diag(core(G))) == 2 # same as rankof(G)

    @test_throws ArgumentError Tucker((A, B, C))

    frozen_factors = (false, true, false, false)
    G = CPDecomposition((A, B, C, D), frozen_factors)
    @test frozen(G) == frozen_factors

end

@testset verbose=VERBOSE "BlockUpdatedDecomposition" begin
    G = Tucker1((10,11,12), 5);
    Y = Tucker1((10,11,12), 5);
    Y = array(Y);

    # check hitting maximum number of iterations
    decomposition, stats_data = BlockTensorDecomposition.factorize(Y; rank=5, momentum=false, maxiter=2);
    # check convergence on first iteration
    decomposition, stats_data = BlockTensorDecomposition.factorize(Y; rank=5, momentum=false, tolerence=Inf);
    # check momentum
    decomposition, stats_data = BlockTensorDecomposition.factorize(Y; rank=5, momentum=true, tolerence=Inf);
    # check constraints
    ## a single constraint, to be applied on every block
    decomposition, stats_data = BlockTensorDecomposition.factorize(Y; rank=5, constraints=nnegative!, tolerence=Inf);
    ## a collection of constraints
    decomposition, stats_data = BlockTensorDecomposition.factorize(Y; rank=5, tolerence=Inf,
        constraints=[ConstraintUpdate(0, nnegative!), ConstraintUpdate(0, l1scaled_12slices!), ConstraintUpdate(1, nnegative!)],
    );
    ## check if you can constrain the initialization
    decomposition, stats_data = BlockTensorDecomposition.factorize(Y; rank=5, tolerence=Inf, constrain_init=true,
        constraints=[ConstraintUpdate(0, nnegative!), ConstraintUpdate(0, l1scaled_12slices!), ConstraintUpdate(1, nnegative!)],
    );

    #=
    bgd! = block_gradient_decent(G, Y);

    @test typeof(bgd!) <: AbstractUpdate # make sure it correctly made the update struct

    N = 100
    v = zeros(N)
    for i in 1:N
        bgd!(G);
        v[i] = norm(array(G)-Y)
    end

    @test v[end] / v[begin] < 0.2 # expect to see at least 80% improvement of error

    G = Tucker1((10,11,12), 5; init=abs_randn);
    Y = Tucker1((10,11,12), 5; init=abs_randn);
    Y = array(Y);

    bgd! = scaled_nn_block_gradient_decent(G, Y;
        scale=l1scaled!,
        whats_rescaled=(x -> factor(x, 2))
    )

    bgd!(G)

    @test l1norm(core(G)) ≈ 1

    N = 100
    v = zeros(N)
    for i in 1:N
        bgd!(G);
        v[i] = norm(array(G)-Y)
    end

    @test v[end] / v[begin] < 0.4 # only expect at least 60% improvement in 100 iterations
    @test l1norm(core(G)) ≈ 1 # G should still be normalized

    G = Tucker1((10,11,12), 5; init=randn);
    Y = Tucker1((10,11,12), 5; init=randn);
    Y = array(Y);
=#

    #=
    bgd! = proj_nn_block_gradient_decent(G, Y;
        proj=l1normalize!,
    )

    bgd!(G)

    @test l1norm(core(G)) ≈ 1

    N = 100
    v = zeros(N)
    for i in 1:N
        bgd!(G);
        v[i] = norm(array(G)-Y)
    end

    @test v[end] / v[begin] < 0.4 # only expect at least 60% improvement in 100 iterations
    @test isapprox(l1norm(core(G)), 1; atol=0.015)
    @test_broken l1norm(core(G)) ≈ 1 # G should still be normalized
    =#
end

end
