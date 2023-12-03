settings.outformat = "pdf";
defaultpen(fontsize(9pt));
unitsize(3mm);
settings.tex="pdflatex" ;


import graph;
//size(7cm);

//-0.37242316  0.85520254 -0.51623059  1.        

string colour1 = "C30934";
string colour2 = "1565C0";
string colour3 = "006F63";


// grey tweezer goes first to be behind
real optical_tweez_height = 3.2;
real optical_tweez_width_min = 0.5;
real optical_tweez_width_max = 1.7;
fill((12 - optical_tweez_width_min,0){up} .. (12 - optical_tweez_width_max,optical_tweez_height) -- (12 + optical_tweez_width_max,optical_tweez_height) .. (12+optical_tweez_width_min,0){down} .. (12+optical_tweez_width_max,-optical_tweez_height) -- (12 - optical_tweez_width_max, -optical_tweez_height) .. cycle, mediumgray);

// ################## FIRST ONE
label("(a)", (4,4.2));
//shakes
real large_shake_height = 1;
// draw((0,large_shake_height) -- (0,-large_shake_height), p=rgb("C30934")+linewidth(1pt), arrow=ArcArrows());
draw((6,large_shake_height) -- (6,-large_shake_height ), p=rgb(colour1)+linewidth(0.7pt), arrow=ArcArrows());
draw((12,large_shake_height) -- (12,-large_shake_height), p=rgb(colour2)+linewidth(0.7pt), arrow=ArcArrows());
draw((18,large_shake_height) -- (18,-large_shake_height), p=rgb(colour3)+linewidth(0.7pt), arrow=ArcArrows());

//dots
// dot((0,0));
dot((6,0));
dot((12,-1));
dot((18,0));

// e_i bit
label("$\epsilon_l(t)$", (14.7,-0.8));
draw((13,0) -- (13,-1), bar=Bars);

//tunnelling black curves
real tunnelling_line_height = 0.2;
real tunnelling_curve_height = 1;
real tunnelling_curve_space_to_dot = 0.4;
// draw((tunnelling_curve_space_to_dot,tunnelling_line_height) .. (3,tunnelling_line_height+tunnelling_curve_height) .. (6 - tunnelling_curve_space_to_dot,tunnelling_line_height));
draw((6 + tunnelling_curve_space_to_dot,tunnelling_line_height) .. (9,tunnelling_line_height+tunnelling_curve_height) .. (12 - tunnelling_curve_space_to_dot,tunnelling_line_height));
draw((12 + tunnelling_curve_space_to_dot,tunnelling_line_height) .. (15,tunnelling_line_height+tunnelling_curve_height) .. (18 - tunnelling_curve_space_to_dot,tunnelling_line_height));

real label_height = 1.9;
// label("$J_0$", (2.9,label_height) );
label("$J_0$", (8.9,label_height));
label("$J_0$", (14.9,label_height));

real A_vals_height = 1.8;
// label("$A_1$", (-1, A_vals_height), p=rgb("C30934"));
label("$A_{l-1}$", (6, A_vals_height));
label("$A_l$", (12, A_vals_height));
label("$A_{l+1}$", (18, A_vals_height));


// ################## second time-depenent pic
label("(b)", (20,4.2));
real origin_x = 23;
real origin_y = -1;
pair origin = (origin_x,origin_y);
real wave_amplitude = 2;

// axes
draw(origin -- origin + (0,3.7), arrow=Arrow(TeXHead));
draw(origin -- origin + (2.4*pi,0), arrow=Arrow(TeXHead));
label("$t$", origin + (2.4*pi+0.3,-0.4));
label("$\epsilon_l(t)$", origin +(-1.5, 3.7));
draw(origin + (0,wave_amplitude) -- origin +(-0.3, wave_amplitude));
label("$A_l$", origin + (-1.3, wave_amplitude));
draw(origin + (2*pi, 0) -- origin + (2*pi, -0.3));
draw("$T$", origin +(2*pi, -1.3));

//function
real f(real t) { return (wave_amplitude)*sin(t-origin_x) + origin_y; }
path g = graph(f, 0+origin_x, 2.4*pi+origin_x, n=200);
draw(g, p=rgb("1565C0")+linewidth(1pt));

real second_row_label_height = -4.5;
real image_height = -10.1;
real third_row_label_height = second_row_label_height - 11;

label(graphic("/home/gnixon/floquet-simulations/figures/black_hole_paper/a_vals_alternating_medium.pdf"),(9.8,image_height));
label("(c)", (4,second_row_label_height));

label(graphic("/home/gnixon/floquet-simulations/figures/black_hole_paper/stroboscopic_ham_medium.pdf"),(23.7,image_height+0.8));
label("(d)", (17,second_row_label_height));


label(graphic("/home/gnixon/floquet-simulations/figures/black_hole_paper/tunnellings_alternating_long.pdf"),(18,third_row_label_height-3.9));


label("(e)", (4,third_row_label_height));



