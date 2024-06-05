settings.outformat = "png";
settings.render=20;
defaultpen(fontsize(16pt));
unitsize(3mm);
usepackage("amsfonts");

pen lw = linewidth(1.7pt);
pen dw = linewidth(7.5pt);
pen sw = linewidth(1pt);

pen coloura = rgb("006F63");
pen colourb = rgb("F78320");
pen colourc = rgb("#9E7AFF");

real rad = 5;
draw(circle((0,0),rad),p=sw);
draw(ellipse((0,0),rad,2), p=sw+dashed);


real spread = 0.6;
real text_rad =6.2;
dot((-spread,text_rad), p=coloura+dw);
label("a", (spread,text_rad));
dot((-spread,-text_rad), p=colourb+dw);
label("b", (spread,-text_rad));
draw((0,0) -- (sqrt((rad**2)/2),sqrt((rad**2)/2)), p=colourc+lw, arrow=ArcArrow(SimpleHead, size=4));