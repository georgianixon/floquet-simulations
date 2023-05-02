settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);
//size(7cm);

// ~~~~~~~~~~ First Lattice
label("(a)", (-1,4));

dot((0,0));
dot((5,0));
dot((10,0));
dot((15,0));
dot((20,0));
dot((25,0));

draw((15,1) -- (15,-1), p=rgb("006F63")+linewidth(1pt), Arrows);
draw((20, 1) -- (20,-1), p=rgb("006F63")+linewidth(1pt), Arrows);
draw((25,1) -- (25,-1), p=rgb("006F63")+linewidth(1pt), Arrows);


draw((0.3,1) .. (2.5,2.1) .. (4.7,1));
draw((5.3,1) .. (7.5,2.1) .. (9.7,1));
draw((10.3,1) .. (12.5,2.1) .. (14.7,1));
draw((15.3,1) .. (17.5,2.1) .. (19.7,1));
draw((20.3,1) .. (22.5,2.1) .. (24.7,1));

// label
label("$J$", (12.5,2.9),  black);
label("$J$", (7.5,2.9), black);
label("$J$", (2.5,2.9), black);
label("$J$", (17.5,2.9), black);
label("$J$", (22.5,2.9), black);

// ~~~~~~ Second Lattice

pair fig_shift = (0,-7);
label("(b)", (-1,4)+fig_shift);

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

