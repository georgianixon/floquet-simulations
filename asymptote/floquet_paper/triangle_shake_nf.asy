settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);

// string colour1 = "B76FB3"; // pink
// string colour2 = "6BB5C6"; // light blue
// string colour3 = "006F63"; // green
// string colour4 = "F57F17"; //orange
// string colour5 = "0F1980"; //purple



string colour1 = "AD7A99"; // pink
string colour2 = "27BED2"; // light blue
string colour4 = "006F63"; // green
string colour3 = "F57F17"; //orange
string colour5 = "0F1980"; //purple
string colour6 = "C30934"; //red


// ###################### First Triangle
//shaking arrows
real arrow_height2 = 2.2;
real arrow_height3 = 1.6;

real lattice_space=6.2;

draw((0,0) -- (0,arrow_height2), p=rgb(colour6)+linewidth(2.3pt), arrow=ArcArrow(SimpleHead, size=5));
draw((0,0) -- (0,-arrow_height2), p=rgb(colour6)+linewidth(2.3pt)+linetype("2 2"), arrow=ArcArrow(SimpleHead, size=5));


// draw((5,1.4) -- (5,-1.4), p=rgb(colour1)+linewidth(1pt), arrow=ArcArrows());
// draw((5,0.5) -- (5,-0.5), p=rgb(colour1)+linewidth(4pt));

draw((lattice_space,0) -- (lattice_space,arrow_height3), p=rgb(colour5)+linewidth(1pt)+linetype("1 2"), arrow=ArcArrow(SimpleHead, size=4));
draw((lattice_space,0) -- (lattice_space,-arrow_height3), p=rgb(colour5)+linewidth(1pt), arrow=ArcArrow(SimpleHead, size=4));

// draw((i_d*5,0) -- (i_d*5,-arrow_height), p=rgb(colour5)+linewidth(0.9pt)+linetype("2 2"), arrow=ArcArrow(SimpleHead, size=4));

// red hopping
real line_gap = 0.6;
draw((0,0)+line_gap*(Cos(60),Sin(60)) .. lattice_space*(Cos(60),Sin(60))-line_gap*(Cos(60),Sin(60)));
draw(lattice_space*(Cos(60),Sin(60))-line_gap*(Cos(120), Sin(120)) .. (lattice_space,0)+line_gap*(Cos(120), Sin(120))  );
draw((line_gap,0) .. (lattice_space-line_gap,0));



dot((0,0));
dot((lattice_space,0));
dot(lattice_space*(Cos(60),Sin(60)));


// labels
real label_gap = 1;
label("$J$", lattice_space/2*(Cos(60),Sin(60))+label_gap*(Cos(150),Sin(150)));
label("$J$", (lattice_space,0)+lattice_space/2*(Cos(120),Sin(120))+label_gap*(Cos(30), Sin(30)));
label("$J$", (lattice_space/2,0)-label_gap*(0,1));

//a label
pair label_loc = (-1.5, lattice_space*1.01);
label("(a)", label_loc);

//site labels
label("$1$", lattice_space*(0.5, Sin(60))+(0,1));
label("$3$", lattice_space*(1, 0)+(1,0));
label("$2$", (0, 0)+(-1,0));


// ######################### Second Triangle
pair triangle_shift = (2.2*lattice_space,0);
//shaking arrows

// red hopping
real line_gap = 0.6;
draw((0,0)+line_gap*(Cos(60),Sin(60))+ triangle_shift .. lattice_space*(Cos(60),Sin(60))-line_gap*(Cos(60),Sin(60))+ triangle_shift, p=rgb(colour3)+linewidth(1pt));
draw(lattice_space*(Cos(60),Sin(60))-line_gap*(Cos(120), Sin(120))+ triangle_shift .. (lattice_space,0)+line_gap*(Cos(120), Sin(120))+ triangle_shift  ,p=rgb(colour4)+linewidth(1pt));
draw((line_gap,0)+ triangle_shift .. (lattice_space-line_gap,0)+ triangle_shift,p=rgb(colour2)+linewidth(1pt));
 



dot((0,0)+ triangle_shift);
dot((lattice_space,0)+ triangle_shift);
dot(lattice_space*(Cos(60),Sin(60))+ triangle_shift);


// labels
real label_gap = 1.8;
label("$J_{12}$", lattice_space/2*(Cos(60),Sin(60))+label_gap*(Cos(150),Sin(150))+ triangle_shift, p=rgb(colour3)+linewidth(1.1pt));
label("$J_{31}$", (lattice_space,0)+lattice_space/2*(Cos(120),Sin(120))+label_gap*(Cos(30), Sin(30))+ triangle_shift, p=rgb(colour4));
label("$J_{23}$", (lattice_space/2,0)-label_gap*(0,1)+ triangle_shift+(0,0.7), p=rgb(colour2));

//b label
label("(b)", label_loc+triangle_shift);

// flux label
pair tri_centre = (lattice_space/2, lattice_space*0.3);
label("$\Phi$", tri_centre+triangle_shift);


real flux_circle_line_rad = 1;

path flux_curve = tri_centre+(0,flux_circle_line_rad)+triangle_shift..tri_centre+(-flux_circle_line_rad,0)+triangle_shift..tri_centre+(0,-flux_circle_line_rad)+triangle_shift..tri_centre+(flux_circle_line_rad,0)+triangle_shift; 
draw(flux_curve,ArcArrow()); 

