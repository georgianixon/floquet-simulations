settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);

size(7cm);
dot((0,0));
dot((5,0));
dot((10,0));
dot((15,0));
dot((20,0));


draw((10,1) -- (10,-1), p=rgb("006F63")+linewidth(1pt), Arrows);


draw((0.3,1) .. (2.5,2.1) .. (4.7,1));
draw((5.3,1) .. (7.5,2.1) .. (9.7,1),  p=rgb("C30934"));
draw((10.3,1) .. (12.5,2.1) .. (14.7,1),  p=rgb("C30934"));
draw((15.3,1) .. (17.5,2.1) .. (19.7,1));


// label
label("$J'$", (12.5,2.9),  p=rgb("C30934"));
label("$J'$", (7.5,2.9),  p=rgb("C30934"));
label("$J$", (2.5,2.9), black);
label("$J$", (17.5,2.9), black);
