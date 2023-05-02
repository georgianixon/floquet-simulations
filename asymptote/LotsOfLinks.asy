settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);
//size(7cm);

//dots
dot((0,0));
dot((5,0));
dot((10,0));
dot((15,0));
dot((20,0));
dot((25,0));
dot((30,0));
dot((35,0));
dot((40,0));

//shakes
//red


draw((9.7,1) -- (9.7,-1), heavycyan, Arrows);
draw((14.7,1) -- (14.7,-1), heavycyan, Arrows);
draw((19.7,1) -- (19.7,-1), heavycyan, Arrows);
draw((24.7,1) -- (24.7,-1), heavycyan, Arrows);
draw((29.7,1) -- (29.7,-1), heavycyan, Arrows);
draw((34.7,1) -- (34.7,-1), heavycyan, Arrows);
draw((39.7,1) -- (39.7,-1), heavycyan, Arrows);

//blue

draw((20.3,0.6) -- (20.3,-0.6), orange, Arrows);

//green
draw((35.3,1.2) -- (35.3,-1.2), deepmagenta, Arrows);
draw((40.3,1.2) -- (40.3,-1.2), deepmagenta, Arrows);


// normal hops
draw((0.3,1.3) .. (2.5,2.4) .. (4.7,1.3));
draw((10.3,1.3) .. (12.5,2.4) .. (14.7,1.3));
draw((25.3,1.3) .. (27.5,2.4) .. (29.7,1.3));
draw((35.3,1.3) .. (37.5,2.4) .. (39.7,1.3));


//hopping labels
draw((5.3,1.3) .. (7.5,2.4) .. (9.7,1.3), red);
label("$J'$", (7.5,3.1), red);

draw((15.3,1.3) .. (17.5,2.4) .. (19.7,1.3), red);
label("$J''$", (17.5,3.1), red);

draw((20.3,1.3) .. (22.5,2.4) .. (24.7,1.3), red);
label("$J''$", (22.5,3.1), red);

draw((30.3,1.3) .. (32.5,2.4) .. (34.7,1.3), red);
label("$J'''$", (32.5,3.1), red);