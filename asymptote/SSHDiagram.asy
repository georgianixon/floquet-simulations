settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);
//size(7cm);

// ################## FIRST ONE
label("(a)", (-3,4));
//shakes
real large_shake_height = 2;
real small_shake_height = 1.4;
draw((0,large_shake_height) -- (0,-large_shake_height), p=rgb("006F63")+linewidth(1pt), arrow=ArcArrows());
draw((5,small_shake_height ) -- (5,-small_shake_height ), p=rgb("F78320")+linewidth(1pt), arrow=ArcArrows());
draw((10,large_shake_height) -- (10,-large_shake_height), p=rgb("006F63")+linewidth(1pt), arrow=ArcArrows());
draw((15,small_shake_height ) -- (15,-small_shake_height ), p=rgb("F78320")+linewidth(1pt), arrow=ArcArrows());
draw((20,large_shake_height) -- (20,-large_shake_height), p=rgb("006F63")+linewidth(1pt), arrow=ArcArrows());


//dots
dot((0,0));
dot((5,0));
dot((10,0));
dot((15,0));
dot((20,0));


//tunnellings
real label_height = 3.5;
real tunnelling_line_height = 1.9;
draw((0.3,tunnelling_line_height) .. (2.5,tunnelling_line_height+1.1) .. (4.7,tunnelling_line_height));
draw((0.3,tunnelling_line_height) .. (2.5,tunnelling_line_height+1.1) .. (4.7,tunnelling_line_height));
draw((5.3,tunnelling_line_height) .. (7.5,tunnelling_line_height+1.1) .. (9.7,tunnelling_line_height));
draw((10.3,tunnelling_line_height) .. (12.5,tunnelling_line_height+1.1) .. (14.7,tunnelling_line_height));
draw((15.3,tunnelling_line_height) .. (17.5,tunnelling_line_height+1.1) .. (19.7,tunnelling_line_height));
label("$J$", (2.5,label_height) );
label("$J$", (7.5,label_height));
label("$J$", (12.5,label_height));
label("$J$", (17.5,label_height));



// ################## second ONE
pair fig_shift = (0,-7);
label("(b)", (-3,4)+fig_shift);

//dots
dot((0,0)+fig_shift );
dot((5,0)+fig_shift );
dot((10,0)+fig_shift );
dot((15,0)+fig_shift );
dot((20,0)+fig_shift );


//tunnellings
draw((0.3,1.3)+fig_shift  .. (2.5,2.4)+fig_shift  .. (4.7,1.3)+fig_shift , p=rgb("006F63")+linewidth(1.1pt));
draw((5.3,1.3)+fig_shift  .. (7.5,2.4)+fig_shift  .. (9.7,1.3)+fig_shift , p=rgb("F78320")+linewidth(1.1pt));
draw((10.3,1.3) +fig_shift .. (12.5,2.4)+fig_shift  .. (14.7,1.3)+fig_shift , p=rgb("006F63")+linewidth(1.1pt));
draw((15.3,1.3)+fig_shift  .. (17.5,2.4)+fig_shift  .. (19.7,1.3)+fig_shift , p=rgb("F78320")+linewidth(1.1pt));
label("$J'$", (2.5,label_height)+fig_shift , p=rgb("006F63"));
label("$J''$", (7.5,label_height)+fig_shift , p=rgb("F78320"));
label("$J'$", (12.5,label_height)+fig_shift , p=rgb("006F63"));
label("$J''$", (17.5,label_height)+fig_shift , p=rgb("F78320"));

