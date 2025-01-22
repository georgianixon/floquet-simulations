settings.outformat = "pdf";
// settings.render=30;
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex";
size(7cm);


string colour1 = "AD7A99"; // pink
string colour2 = "7CDEDC"; // light blue
string colour3 = "006F63"; // green
string colour4 = "F57F17"; // orange
string colour5 = "0F1980"; // purple
string colour6 = "C30934"; // red
string blue1 = "#273A54";
string blue2 = "#40618C";
string blue3 = "#6589B8";
string yellow1 = "#C96480";
string flux1 = "#D8F2EB";
string flux2 = "#F2EAD8";
string flux3 = "#D8E0F2";
string darkflux1 = "#57C7A9";
string darkflux2 = "#C7A557";
string darkflux3 = "#5778C7";

int lat_Ly = 4; 
int lat_Lx = 4;
real lat_space = 5.5;

// [[1.75 1.75 0.75 0.75]
//  [0.25 1.25 1.25 0.25]
//  [0.75 0.75 1.75 1.75]
//  [1.25 0.25 0.25 1.25]]

pen tun_pen = linewidth(1.8pt);
// label("$J_x \mathcal{J}_0\!\left(\!\frac{A_{ij}}{\omega}\!\right)\!$", lat_space*(0.5, 0.14), p=fontsize(7pt));
// frame f;
// label(f,"$J_y \mathcal{J}_1(A_x)\!e^{i \xi_{ij}}$",lat_space*(-3, 1.4),p=fontsize(7pt));
// add(rotate(90)*f);

//shading
// tunnellings vertical
real flux_label_y = -0.5;
pen arrow_pen = linewidth(1.4pt);
real flux_arrow_rad = lat_space*0.27;
// draw things in the middle of the plaquette
string[] horiz_flux_colours = {flux1, flux2, flux3};
string[] horiz_darkflux_colours = {darkflux1, darkflux2, darkflux3};
string[] horiz_fluxes = {"$g$", "$2g$", "$3g$"};
for (int i_y = -lat_Ly +2; i_y <=0; ++i_y)
{
    for (int i_x = 0; i_x < lat_Lx-1; ++ i_x)
    {
        fill(lat_space*(i_x,i_y) -- lat_space*(i_x,i_y-1) -- lat_space*(i_x + 1,i_y -1) -- lat_space*(i_x + 1,i_y) -- cycle, p=rgb(horiz_flux_colours[i_x]));
        draw(arc(lat_space*(i_x + 0.5, i_y - 0.5 ), r=flux_arrow_rad, angle1=240, angle2=-60), arrow=Arrow(TeXHead), p=arrow_pen+rgb(horiz_darkflux_colours[i_x]));
        label(horiz_fluxes[i_x], lat_space*(i_x + 0.5, i_y - 0.5 ));
    }
}

real horiz_tunnel_label_ygap = -0.1;
// vertical tunnelling labels
pen fnt_size_sides = fontsize(8pt);
real y_sides_secondline = -0.12;
real y_side_labels = -0.7;


// draw dots
for (int i_y=-(lat_Ly-1); i_y<=0; ++i_y)
{ 
    for (int i_x = 0; i_x < lat_Lx; ++i_x)
    {
        if (i_y % 2 ==1)
        {
            dot(lat_space*(i_x, i_y));//, p=rgb(colour4));
        }
        else
        {
             dot(lat_space*(i_x, i_y));
        }
    }
}

// draw tunnellings
real tun_gap = 0.6;
// tunnellings horizontal
string[] horiz_tunnel_colors={blue1, blue2, blue3};
for (int i_x = 0; i_x < lat_Lx - 1; ++i_x)
{
    for (int i_y = 0; i_y > -lat_Ly; --i_y)
    {
        draw((i_x,i_y)*lat_space+ (tun_gap, 0) -- (i_x+1, i_y)*lat_space - (tun_gap,0), p=rgb(horiz_tunnel_colors[i_x])+tun_pen);
    }
}

for (int i_x = 0; i_x < lat_Lx; ++i_x)
{
    for (int i_y = 0; i_y > -lat_Ly + 1; --i_y)
    {
        draw((i_x, i_y-1)*lat_space + (0, tun_gap) -- (i_x,i_y)*lat_space - (0, tun_gap), p=rgb(yellow1)+tun_pen);
    }
}

