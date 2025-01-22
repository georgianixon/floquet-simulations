settings.outformat = "pdf";
// settings.render=20;
defaultpen(fontsize(9pt));
//defaultpen(arrowsize(9));
//defaultpen(arrowsize(5bp));
unitsize(3mm);
settings.tex="pdflatex" ;


import graph;
string colour1 = "1565C0";
string colour2 = "C30934";
string colour3 = "006F63";
string colour4 = "F57F17";

string colour_u = colour3;
string colour_e = colour4;

//size


real func_freq1 = 0.5;
real func_a1 = 3;
real func_p1 = 0.3*pi;
real func_freq2 = func_freq1*2;
real func_a2 = func_a1*0.8;
real func_p2 = pi*0.8;
real T1 = 2*pi/func_freq1;

real func_x0 = 0;
real func_xf = T1*1.7;

real func_t(real t) { return func_a1*sin(func_freq1*t + func_p1)+ func_a2*sin(func_freq2*t + func_p2); }
path func_path = graph(func_t, func_x0, func_xf, n=200);
draw(func_path, p=linewidth(1pt));

real t0 = 0.08*T1;
real t1 = 0.53*T1;
draw((t0, func_t(t0)) -- (t0 + T1, func_t(t0 + T1)), p=rgb(colour3)+linewidth(1pt), arrow=ArcArrow(SimpleHead, size=4));
draw((t1, func_t(t1)) -- (t1 + T1, func_t(t1 + T1)), p=rgb(colour2)+linewidth(1pt), arrow=ArcArrow(SimpleHead, size=4));
draw ((t0, func_t(t0)) -- (t0, -func_a1*func_a2), p=linetype("2 2")+linewidth(0.7pt)+rgb("6C6C6C"));
draw ((t0 + T1, func_t(t0 + T1)) -- (t0+T1, -func_a1*func_a2), p=linetype("2 2")+linewidth(0.7pt)+rgb("6C6C6C"));
draw ((t1, func_t(t1)) -- (t1, -func_a1*func_a2), p=linetype("2 2")+linewidth(0.7pt)+rgb("6C6C6C"));
draw ((t1+T1, func_t(t1 + T1)) -- (t1 + T1, -func_a1*func_a2), p=linetype("2 2")+linewidth(0.7pt)+rgb("6C6C6C"));

draw((0, -func_a1*func_a2) -- (func_xf,  -func_a1*func_a2), p=linewidth(0.9pt), arrow=ArcArrow(SimpleHead, size=4));

real t0_lab_ygap = 0.8;
label("time", (func_xf+0.1*T1, -func_a1*func_a2));
label("$t_0$", (t0, -func_a1*func_a2 - t0_lab_ygap));
label("$t_0\!+\!T$", (t0+T1, -func_a1*func_a2 - t0_lab_ygap));
label("$t_1$", (t1, -func_a1*func_a2 - t0_lab_ygap));
label("$t_1\!+\!T$", (t1+T1, -func_a1*func_a2 - t0_lab_ygap));


label("$H_S^{t_0}$", (t0 + T1/2, func_t(t0) + t0_lab_ygap), p=rgb(colour3));
label("$H_S^{t_1}$", (0.97*T1, func_t(t1) + t0_lab_ygap), p=rgb(colour2));



