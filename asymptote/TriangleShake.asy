settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);


// ###################### First Triangle
//shaking arrows
draw((0,2) -- (0,-2), p=rgb("006F63")+linewidth(1pt), arrow=ArcArrows());
draw((5,1.4) -- (5,-1.4), p=rgb("F78320")+linewidth(1pt), arrow=ArcArrows());
draw((5,0.5) -- (5,-0.5), p=rgb("F78320")+linewidth(4pt));



// red hopping
real line_gap = 0.6;
draw((0,0)+line_gap*(Cos(60),Sin(60)) .. 5*(Cos(60),Sin(60))-line_gap*(Cos(60),Sin(60)));
draw(5*(Cos(60),Sin(60))-line_gap*(Cos(120), Sin(120)) .. (5,0)+line_gap*(Cos(120), Sin(120))  );
draw((line_gap,0) .. (5-line_gap,0));



dot((0,0));
dot((5,0));
dot(5*(Cos(60),Sin(60)));


// labels
real label_gap = 1.8;
label("$J$", 2.5*(Cos(60),Sin(60))+label_gap*(Cos(150),Sin(150)));
label("$J$", (5,0)+2.5*(Cos(120),Sin(120))+label_gap*(Cos(30), Sin(30)));
label("$J$", (2.5,0)-label_gap*(0,1));

//a label
label("(a)", (-3,5));


// ######################### Second Triangle
pair triangle_shift = (13,0);
//shaking arrows

// red hopping
real line_gap = 0.6;
draw((0,0)+line_gap*(Cos(60),Sin(60))+ triangle_shift .. 5*(Cos(60),Sin(60))-line_gap*(Cos(60),Sin(60))+ triangle_shift, p=rgb("006F63")+linewidth(1.1pt));
draw(5*(Cos(60),Sin(60))-line_gap*(Cos(120), Sin(120))+ triangle_shift .. (5,0)+line_gap*(Cos(120), Sin(120))+ triangle_shift  ,p=rgb("F78320")+linewidth(1.1pt));
draw((line_gap,0)+ triangle_shift .. (5-line_gap,0)+ triangle_shift,p=rgb("C30934")+linewidth(1.1pt));



dot((0,0)+ triangle_shift);
dot((5,0)+ triangle_shift);
dot(5*(Cos(60),Sin(60))+ triangle_shift);


// labels
real label_gap = 1.8;
label("$J_{3,1}$", 2.5*(Cos(60),Sin(60))+label_gap*(Cos(150),Sin(150))+ triangle_shift, p=rgb("006F63")+linewidth(1.1pt));
label("$J_{1,2}$", (5,0)+2.5*(Cos(120),Sin(120))+label_gap*(Cos(30), Sin(30))+ triangle_shift, p=rgb("F78320"));
label("$J_{2,3}$", (2.5,0)-label_gap*(0,1)+ triangle_shift, p=rgb("C30934"));

//b label
label("(b)", (-3,5)+triangle_shift);
