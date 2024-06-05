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
pen coloura = rgb("006F63");
pen colourb = rgb("F78320");
pen colourhop = rgb("6C6C6C");

//function
real x1 = 0;
real x2 = 6;
real A = 4;
real omega = 2*pi;
real wavelength = 5;
real f(real x) { return A*cos(x*omega/wavelength); }
path g = graph(f ,x1*wavelength , x2*wavelength, n=200);
draw(g, p=rgb(colour)+lw);

for (int i =0; i<=5; ++i)
{
    dot(((i+0.4)*wavelength,0), p=coloura+dw);
    dot(((i+0.6)*wavelength,0), p=colourb+dw);
}

real t_width = 0.3;
real arrow_head_size = 4;
real t_ygap = 1.6;
pen tw = linewidth(1.5pt);

real ft(real x) { return A*cos((x-wavelength)*2*pi/(wavelength*1.5))+t_ygap; }
path ta = graph(ft, (1+t_width)*wavelength, (1-t_width)*wavelength, n=20);
draw(ta, p=colourhop+tw, arrow=ArcArrows(SimpleHead, size=arrow_head_size));

// real fb(real x) { return A*cos((x-2*wavelength)*2*pi/(wavelength*1.5))+t_ygap; }
// path ta = graph(fb, (2+t_width)*wavelength, (2-t_width)*wavelength, n=20);
// draw(ta, p=colourb+tw, arrow=ArcArrows(SimpleHead, size=arrow_head_size));

label("$J$", (wavelength-1.4,A+1.8), p=colourhop);
// label("$J$", (2*wavelength-1.8,A+1.8), p=colourb);

dot((4.5*wavelength,A*1.2), p=coloura+dw);
label("a", (4.76*wavelength,A*1.2));
dot((5.5*wavelength,A*1.2), p=colourb+dw);
label("b", (5.76*wavelength,A*1.2));
