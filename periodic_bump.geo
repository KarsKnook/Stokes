H = 1;
L = 2;
h = 0.5;
l = 0.5;

Point(1) = {-L,  H, 0, 0.1};
Point(2) = {-L, -H, 0, 0.1};
Point(3) = {-l, -H, 0, 0.1};
Point(4) = {-l, h, 0, 0.1};
Point(5) = {l, h, 0, 0.1};
Point(6) = {l, -H, 0, 0.1};
Point(7) = {L, -H, 0, 0.1};
Point(8) = {L, H, 0, 0.1};

Line(10) = {1, 2};
Line(11) = {2, 3};
Line(12) = {3, 4};
Line(13) = {4, 5};
Line(14) = {5, 6};
Line(15) = {6, 7};
Line(16) = {7, 8};
Line(17) = {8, 1};

Curve Loop(100) = {10, 11, 12, 13, 14, 15, 16, 17};
Plane Surface(1000) = {100};

Periodic Curve {10} = {16} Translate {2*L, 0, 0};

Recombine Surface(1000);

Physical Curve("lower", 1) = {10, 11, 12, 13, 14, 15, 16};
Physical Curve("upper", 2) = {17};
Physical Surface("interior", 3) = {1000};