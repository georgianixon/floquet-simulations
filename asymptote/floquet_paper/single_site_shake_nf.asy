settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");

size(7cm);

string colour1 = "AD7A99"; // pink
string colour2 = "7CDEDC"; // light blue
string colour3 = "006F63"; // green
string colour4 = "F57F17"; //orange
string colour5 = "0F1980"; //purple


pair label_loc = (-1,2.5);
label("(a)", label_loc);
dot((0,0));
dot((5,0));
dot((10,0));
dot((15,0));
dot((20,0));


// shaking arrow
real arrow_height = 1.8;
draw((10,0) -- (10, arrow_height), p=rgb(colour5)+linewidth(0.9pt), arrow=ArcArrow(SimpleHead, size=4));
draw((10,0) -- (10,-arrow_height), p=rgb(colour5)+linewidth(0.9pt)+linetype("2 2"), arrow=ArcArrow(SimpleHead, size=4));


real y0_tunnelling_curve = 0.5;
real y_height_tunnelling_turve = 0.9;
draw((0.3, y0_tunnelling_curve) .. (2.5,y_height_tunnelling_turve+ y0_tunnelling_curve) .. (4.7, y0_tunnelling_curve));
draw((5.3, y0_tunnelling_curve) .. (7.5,y_height_tunnelling_turve+ y0_tunnelling_curve) .. (9.7, y0_tunnelling_curve));
draw((10.3, y0_tunnelling_curve) .. (12.5,y_height_tunnelling_turve+ y0_tunnelling_curve) .. (14.7, y0_tunnelling_curve));
draw((15.3, y0_tunnelling_curve) .. (17.5,y_height_tunnelling_turve+ y0_tunnelling_curve) .. (19.7,y0_tunnelling_curve));


// label
real y_j_label =y0_tunnelling_curve+y_height_tunnelling_turve+0.7;
label("$J$", (12.5, y_j_label),  black);
label("$J$", (7.5, y_j_label), black);
label("$J$", (2.5, y_j_label), black);
label("$J$", (17.5, y_j_label), black);


real y_b_label = -arrow_height - 0.9;
label("$b$", (10,y_b_label));

// ~~~~~~ Second Lattice

real y_fig_shift = -6.4;
pair fig_shift = (0,y_fig_shift);
label("(b)", label_loc+fig_shift);

dot((0,0)+fig_shift);
dot((5,0)+fig_shift);
dot((10,0)+fig_shift);
dot((15,0)+fig_shift);
dot((20,0)+fig_shift);

label("$J'$", (12.5,y_j_label)+fig_shift,  p=rgb(colour4));
label("$J'$", (7.5,y_j_label)+fig_shift, p=rgb(colour4));
label("$J$", (2.5,y_j_label)+fig_shift, black);
label("$J$", (17.5,y_j_label)+fig_shift, black);

real y0_tunnelling_curve_b = y0_tunnelling_curve + y_fig_shift;
draw((0.3, y0_tunnelling_curve_b) .. (2.5,y_height_tunnelling_turve+ y0_tunnelling_curve_b) .. (4.7, y0_tunnelling_curve_b));
draw((5.3, y0_tunnelling_curve_b) .. (7.5,y_height_tunnelling_turve+ y0_tunnelling_curve_b) .. (9.7, y0_tunnelling_curve_b), p=rgb(colour4)+linewidth(1pt));
draw((10.3, y0_tunnelling_curve_b) .. (12.5,y_height_tunnelling_turve+ y0_tunnelling_curve_b) .. (14.7, y0_tunnelling_curve_b), p=rgb(colour4)+linewidth(1pt));
draw((15.3, y0_tunnelling_curve_b) .. (17.5,y_height_tunnelling_turve+ y0_tunnelling_curve_b) .. (19.7,y0_tunnelling_curve_b));


label("$b$", (10, y_b_label+ arrow_height)+fig_shift);
