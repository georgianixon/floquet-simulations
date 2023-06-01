settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);
//size(7cm);

// ~~~~~~~~~~ First Lattice
label("\textbf{(a)}", (-1,4));

dot((0,0));
dot((5,0));
dot((10,0));
dot((15,0));
dot((20,0));
dot((25,0));

real tunnelling_line_height = 1.2;
//draw((15,tunnelling_line_height) -- (15,-tunnelling_line_height), p=rgb("006F63")+linewidth(1pt), Arrows);
//draw((20, tunnelling_line_height) -- (20,-tunnelling_line_height), p=rgb("006F63")+linewidth(1pt), Arrows);
//draw((25,tunnelling_line_height) -- (25,-tunnelling_line_height), p=rgb("006F63")+linewidth(1pt), Arrows);
draw((15,tunnelling_line_height) -- (15,-tunnelling_line_height), p=rgb("006F63")+linewidth(1pt), arrow=ArcArrows());
draw((20, tunnelling_line_height) -- (20,-tunnelling_line_height), p=rgb("006F63")+linewidth(1pt), arrow=ArcArrows());
draw((25,tunnelling_line_height) -- (25,-tunnelling_line_height), p=rgb("006F63")+linewidth(1pt), arrow=ArcArrows());


real y_shift_tunnelling_curve = 0.3;
draw((0.3,1+y_shift_tunnelling_curve) .. (2.5,2.1+y_shift_tunnelling_curve) .. (4.7,1+y_shift_tunnelling_curve));
draw((5.3,1+y_shift_tunnelling_curve) .. (7.5,2.1+y_shift_tunnelling_curve) .. (9.7,1+y_shift_tunnelling_curve));
draw((10.3,1+y_shift_tunnelling_curve) .. (12.5,2.1+y_shift_tunnelling_curve) .. (14.7,1+y_shift_tunnelling_curve));
draw((15.3,1+y_shift_tunnelling_curve) .. (17.5,2.1+y_shift_tunnelling_curve) .. (19.7,1+y_shift_tunnelling_curve));
draw((20.3,1+y_shift_tunnelling_curve) .. (22.5,2.1+y_shift_tunnelling_curve) .. (24.7,1+y_shift_tunnelling_curve));

// label
label("$J$", (12.5,2.9+y_shift_tunnelling_curve),  black);
label("$J$", (7.5,2.9+y_shift_tunnelling_curve), black);
label("$J$", (2.5,2.9+y_shift_tunnelling_curve), black);
label("$J$", (17.5,2.9+y_shift_tunnelling_curve), black);
label("$J$", (22.5,2.9+y_shift_tunnelling_curve), black);

label("$b$", (15,-1.8));

// ~~~~~~ Second Lattice

pair fig_shift = (0,-7);
label("\textbf{(b)}", (-1,4)+fig_shift);

dot((0,0)+fig_shift);
dot((5,0)+fig_shift);
dot((10,0)+fig_shift);
dot((15,0)+fig_shift);
dot((20,0)+fig_shift);
dot((25,0)+fig_shift);

label("$J'$", (12.5,2.9)+fig_shift,  p=rgb("C30934"));
label("$J$", (7.5,2.9)+fig_shift, black);
label("$J$", (2.5,2.9)+fig_shift, black);
label("$J$", (17.5,2.9)+fig_shift, black);
label("$J$", (22.5,2.9)+fig_shift, black);

draw((0.3,1)+fig_shift .. (2.5,2.1)+fig_shift .. (4.7,1)+fig_shift);
draw((5.3,1)+fig_shift .. (7.5,2.1)+fig_shift .. (9.7,1)+fig_shift);
draw((10.3,1)+fig_shift .. (12.5,2.1)+fig_shift .. (14.7,1)+fig_shift, p=rgb("C30934"));
draw((15.3,1)+fig_shift .. (17.5,2.1)+fig_shift .. (19.7,1)+fig_shift);
draw((20.3,1)+fig_shift .. (22.5,2.1)+fig_shift .. (24.7,1)+fig_shift);

label("$b$", (15,-1)+fig_shift);

