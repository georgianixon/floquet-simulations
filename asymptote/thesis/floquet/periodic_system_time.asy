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

string colour_u = colour2;
string colour_e = colour4;

//size
real u_amp = 1;
real u_w = 1.1;
real u_origin_x = 0;
real u_final_x = 2*pi/u_w*3;
real exp_amp = 8;
real exp_w = u_w/3;


real u_t(real t) { return (u_amp)*sin(u_w*t)**2*exp_amp*sin(exp_w*t); }
path u_t_path = graph(u_t, u_origin_x, u_final_x, n=200);
draw(u_t_path, p=rgb(colour_u)+linewidth(1pt));


real exp_t(real t) { return exp_amp*sin(exp_w*t); }
path exp_t_path = graph(exp_t, u_origin_x, u_final_x, n=200);
draw(exp_t_path, p=rgb(colour_e)+linewidth(1pt)+linetype("2 2"));


// axes arrow
pair axes_start = (0,-exp_amp);
draw(axes_start -- axes_start+ (2*pi/u_w*0.4, 0), p=linewidth(0.7pt), arrow=ArcArrow(SimpleHead, size=4));
label("$t$",axes_start+ (2*pi/u_w*0.5, 0) );

// a size double arrow
real lattice_const_arrow_y = u_amp*exp_amp*-0.09;
pair lattice_constant_arrow_0 = (2*pi/u_w*0.1, lattice_const_arrow_y);
pair lattice_constant_arrow_f = (2*pi/u_w*0.4, lattice_const_arrow_y);
draw(lattice_constant_arrow_0 -- lattice_constant_arrow_f, p=linewidth(0.7pt), arrow=ArcArrows(SimpleHead, size=4));
label("$T$", (2*pi/u_w*0.25, lattice_const_arrow_y*2));

// ux label
pair ux_label_pos = (2*pi/u_w*2.1,u_amp*exp_amp*0.1);
label("$e^{i \epsilon t / \hbar} u_{\epsilon}(t)$", (ux_label_pos),  p=rgb(colour_u));

// exp label
pair exp_label_pos = (2*pi/u_w*1.35,u_amp*exp_amp*0.8);
label("$e^{i \epsilon t / \hbar }$", (exp_label_pos),  p=rgb(colour_e));


