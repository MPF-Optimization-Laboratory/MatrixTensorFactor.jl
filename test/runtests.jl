"""
Unit tests
"""

using Test

using Random
using BlockTensorDecomposition

VERBOSE = true

@testset verbose=VERBOSE "AbstractDecomposition" begin

A = randn(3,3);
B = randn(4,3);
C = randn(5,3);

CPD = CPDecomposition((A, B, C))

CPD = CPDecomposition((A, B))

G = randn(3,3,3)

G = Tucker((G, A, B, C))

G = Tucker((G, A))

G = Tucker((B', A))

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

@test_throws ArgumentError Tucker((A, B, C))

end

@testset verbose=VERBOSE "BlockUpdatedDecomposition" begin
    G = Tucker1((10,11,12), 5);
    Y = Tucker1((10,11,12), 5);
    BUD = least_square_updates(G, Y);
end
