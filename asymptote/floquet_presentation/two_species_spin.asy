settings.outformat = "png";
settings.render=20;
defaultpen(fontsize(16pt));
unitsize(3mm);
usepackage("amsfonts");


import graph;
real parabol_width = 4;
string colour = "000000"; 
pen lw = linewidth(2.1pt);
pen dw = linewidth(7.5pt);
pen sw = linewidth(1.5pt);

pen coloura = rgb("006F63");
pen colourb = rgb("F78320");
pen colourc = rgb("#9E7AFF");

//function
real x1 = 0;
real x2 = 6;
real A = 4;
real omega = 2*pi;
real wavelength = 5;
real f(real x) { return A*cos(x*omega/wavelength); }
path g = graph(f ,x1*wavelength , x2*wavelength, n=200);
draw(g, p=rgb(colour)+lw);

real spin_length = 1.2;
for (int i =0; i<=5; ++i)
{
    dot(((i+0.5)*wavelength,0), p=colourc+dw);
    draw(((i+0.5)*wavelength,-spin_length ) -- ((i+0.5)*wavelength, spin_length), p=colourc+sw, arrow=ArcArrow(SimpleHead, size=4));
}





