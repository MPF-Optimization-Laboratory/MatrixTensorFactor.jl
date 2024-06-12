"""
Unit tests
"""

using Test

using Random
using LinearAlgebra

using BlockTensorDecomposition

const VERBOSE = true

@testset verbose=true "BlockTensorDecomposition" begin

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

    CPD = CPDecomposition((A, B, C))

    CPD = CPDecomposition((A, B))

    G = randn(3,3,3)

    G = Tucker((G, A, B, C))

    @test rankof(G) == (3,3,3)

    G = Tucker1((G, A))

    G = Tucker1((B', A)) # Can it handle types that are an abstract matrix like Ajoint

    @test_throws ArgumentError Tucker((G, A)) # Can handle auto conversion to TuckerN in the future

    G = Tucker1((10,11,12), 5);
    Y = Tucker1((10,11,12), 5);

    # CPDecomposition test
    A = reshape(1:6, 3, 2)
    B = reshape(1:8, 4, 2)
    C = reshape(1:4, 2, 2)
    D = reshape(1:10, 5, 2)

    CPD = CPDecomposition((A, B, C, D))

    T = [361 434 507 580; 452 544 636 728; 543 654 765 876;;; 482 580 678 776; 604 728 852 976; 726 876 1026 1176;;;; 422 508 594 680; 529 638 747 856; 636 768 900 1032;;; 564 680 796 912; 708 856 1004 1152; 852 1032 1212 1392;;;; 483 582 681 780; 606 732 858 984; 729 882 1035 1188;;; 646 780 914 1048; 812 984 1156 1328; 978 1188 1398 1608;;;; 544 656 768 880; 683 826 969 1112; 822 996 1170 1344;;; 728 880 1032 1184; 916 1112 1308 1504; 1104 1344 1584 1824;;;; 605 730 855 980; 760 920 1080 1240; 915 1110 1305 1500;;; 810 980 1150 1320; 1020 1240 1460 1680; 1230 1500 1770 2040]

    @test array(CPD) == T
    @test ndims(CPD) == 4
    @test rankof(CPD) == 2
    @test all(diag(core(CPD)) .== 1)
    @test length(diag(core(CPD))) == 2 # same as rankof(CPD)

    @test_throws ArgumentError Tucker((A, B, C))

    frozen_factors = (false, true, false, false)
    CPD = CPDecomposition((A, B, C, D), frozen_factors)
    @test frozen(CPD) == frozen_factors

end

@testset verbose=VERBOSE "BlockUpdatedDecomposition" begin
    G = Tucker1((10,11,12), 5);
    Y = Tucker1((10,11,12), 5);
    Y = array(Y)
    bgd! = block_gradient_decent(G, Y);

    @test typeof(bgd!) <: AbstractUpdate # make sure it correctly made the update struct

    N = 100
    v = zeros(N)
    for i in 1:N
        bgd!(G);
        v[i] = norm(array(G)-Y)
    end

    @test v[end] / v[begin] < 1e-2 # expect to see at least 99% improvment of error
end

end
