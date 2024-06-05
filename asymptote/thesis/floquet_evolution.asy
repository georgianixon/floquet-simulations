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

// for the horizontal lines
real t0 = -0.08*T1;
real tf = t0+2.8*T1;
real x_horizontal_line_gap = 0.3;
real y_hfline = func_a1*func_a2-0.1;
real y_uline = y_hfline+4;
real y_timeline = -func_a1*func_a2;

// for the function and t line
real x_func_0 = -0.2*T1;
real x_func_f = tf + 0.2*T1;

// for label gaps
real y_label_gap = 0.8;
real x_timelabel_gap = 0.1*T1;
real y_linespace = 0.1*T1;

real func_t(real t) { return func_a1*sin(func_freq1*t + func_p1)+ func_a2*sin(func_freq2*t + func_p2); }
path func_path = graph(func_t, x_func_0, x_func_f, n=200);
draw(func_path, p=linewidth(1pt));


// draw hf horizontal lines
draw((t0 + x_horizontal_line_gap, y_hfline) -- (t0 + 2*T1 - x_horizontal_line_gap,y_hfline), p=rgb(colour3)+linewidth(1pt), arrow=ArcArrow(SimpleHead, size=4));
draw((t0+2*T1 + x_horizontal_line_gap, y_hfline) -- (tf - x_horizontal_line_gap,y_hfline), p=rgb(colour1)+linewidth(1pt), arrow=ArcArrow(SimpleHead, size=4));
// hf labels
label("$U(t_0+2T, t_0)$ ",(t0 + T1, y_hfline + y_label_gap), p=rgb(colour3));
label("$ = \mathrm{e}^{-i2TH_S^{t_0}/\hbar}$",(t0 + T1, y_hfline - y_label_gap), p=rgb(colour3));
label("$U(t_{\mathrm{f}}, t_0+2T)$ ",((t0 + 2T1)/2 + tf/2, y_hfline + y_label_gap), p=rgb(colour1));
label("$ = P(t_{\mathrm{f}}, t_0) \times$",((t0 + 2T1)/2 + tf/2, y_hfline - y_label_gap), p=rgb(colour1));
label("$\mathrm{e}^{-i(t_{\mathrm{f}} - (t_0 + 2T))H_S^{t_0}/\hbar}$",((t0 + 2T1)/2 + tf/2, y_hfline - y_label_gap-y_linespace), p=rgb(colour1));
// draw u horizontal lines
draw((t0+ x_horizontal_line_gap, y_uline) -- (tf - x_horizontal_line_gap,y_uline), p=rgb(colour2)+linewidth(1pt), arrow=ArcArrow(SimpleHead, size=4));
// u label
label("$U(t_{\mathrm{f}}, t_0)$ ",(t0/2 + tf/2, y_uline + y_label_gap), p=rgb(colour2));
label("$ =  P(t_{\mathrm{f}}, t_0) \mathrm{e}^{-i(t_{\mathrm{f}} - t_0)H_S^{t_0}/\hbar}$",(t0/2 + tf/2, y_uline - y_label_gap), p=rgb(colour2));

// grey lines
draw ((t0, y_uline) -- (t0, y_timeline), p=linetype("2 2")+linewidth(0.7pt)+rgb("6C6C6C"));
draw ((t0 + T1, func_t(t0 + T1)) -- (t0+T1, y_timeline), p=linetype("2 2")+linewidth(0.7pt)+rgb("6C6C6C"));
draw ((t0 + 2*T1, y_hfline) -- (t0+2*T1, y_timeline), p=linetype("2 2")+linewidth(0.7pt)+rgb("6C6C6C"));
draw ((tf, y_uline) -- (tf, y_timeline), p=linetype("2 2")+linewidth(0.7pt)+rgb("6C6C6C"));

//draw timeline
draw((x_func_0, y_timeline) -- (x_func_f,  y_timeline), p=linewidth(0.9pt), arrow=ArcArrow(SimpleHead, size=4));


label("time", (x_func_f + x_timelabel_gap, y_timeline));
label("$t_0$", (t0, y_timeline - y_label_gap));
label("$t_0 + T$", (t0+T1, y_timeline - y_label_gap));
label("$t_0 + 2T$", (t0+2*T1, y_timeline - y_label_gap));
label("$t_{\mathrm{f}}$", (tf, y_timeline - y_label_gap));






