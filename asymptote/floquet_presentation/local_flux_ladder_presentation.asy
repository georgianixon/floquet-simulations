settings.outformat = "png";
settings.render=30;
defaultpen(fontsize(8pt));
unitsize(3mm);
usepackage("amsfonts");

settings.tex="pdflatex" ;


size(7cm);


string colour1 = "AD7A99"; // pink
string colour2 = "7CDEDC"; // light blue
string colour3 = "006F63"; // green
string colour4 = "F57F17"; //orange
string colour5 = "0F1980"; //purple
string colour6 = "C30934"; //red


string blue1 = "#273A54";
string blue2 = "#40618C";
string blue3 = "#6589B8";


string yellow1 = "#C96480";

string flux1 = "#D8F2EB";
string flux2 = "#F2EAD8";
string flux3 = "#D8E0F2";

int lat_Ly = 2; 
int lat_Lx = 4;
real lat_space = 8.6;

string[] horiz_tun_col = {blue1, blue2, blue3};
pen tun_pen = linewidth(1.8pt);

//shading
fill((0,0) -- lat_space*(0,-1) -- lat_space*(1,-1) -- lat_space*(1,0) -- cycle, p=rgb(flux1));
fill(lat_space*(1,0) -- lat_space*(1,-1) -- lat_space*(2,-1) -- lat_space*(2,0) -- cycle, p=rgb(flux2));
fill(lat_space*(2,0) -- lat_space*(2,-1) -- lat_space*(3,-1) -- lat_space*(3,0) -- cycle, p=rgb(flux3));

real horiz_tunnel_label_ygap = -0.1;

// vertical tunnelling labels
pen fnt_size_sides = fontsize(8pt);
real y_sides_secondline = -0.12;
real y_side_labels = -0.7;


label(Label("$\epsilon_h$", Rotate((0,90))), lat_space*(0,y_side_labels) , p = fnt_size_sides, right);
label(Label("$\epsilon_h e^{-ig}$", Rotate((0,90))), lat_space*(1,y_side_labels) , p = fnt_size_sides, left);
label(Label("$\epsilon_h e^{-i3g}$", Rotate((0,90))), lat_space*(2,y_side_labels) , p = fnt_size_sides, left);
label(Label("$\epsilon_h e^{-i6g}$", Rotate((0,90))), lat_space*(3,y_side_labels) , left,  p = fnt_size_sides);


// draw dots
// int i_dot = 1;
pair label_gap_up = 0.5*(1,1);
pair label_gap_down = 0.5*(1,-1);

for (int i_y=-1; i_y<=0; ++i_y)
{ 
    for (int i_x = 0; i_x < lat_Lx; ++i_x)
    {
        dot(lat_space*(i_x, i_y));
        // if (i_dot >=5)
        // {
        //     label((string) i_dot, (i_x, i_y)*lat_space + label_gap_up);
        // }
        // else
        // {
        //     label((string) i_dot, (i_x, i_y)*lat_space + label_gap_down);
        // }
        
        // i_dot = i_dot + 1;

    }
}

// draw tunnellings

real tun_gap = 0.6;

// tunnellings horizontal
for (int i_x = 0; i_x < lat_Lx - 1; ++i_x)
{
    for (int i_y = 0; i_y > -lat_Ly; --i_y)
    {
        draw((i_x,i_y)*lat_space+ (tun_gap, 0) -- (i_x+1, i_y)*lat_space - (tun_gap,0), p=rgb(horiz_tun_col[i_x])+tun_pen);

    }
}

// tunnellings vertical
for (int i_x = 0; i_x < lat_Lx; ++i_x)
{
    for (int i_y = 0; i_y > -lat_Ly + 1; --i_y)
    {
        draw((i_x, i_y-1)*lat_space + (0, tun_gap) -- (i_x,i_y)*lat_space - (0, tun_gap), p=rgb(yellow1)+tun_pen, arrow=ArcArrow(SimpleHead, size=3));

    }
}

// fluxes
real flux_label_y = -0.5;
label("$\Phi_1 = g $", lat_space*(0.5, flux_label_y ));
label("$\Phi_2 = 2g $", lat_space*(1.5,flux_label_y ));
label("$\Phi_3 = 3g $", lat_space*(2.5, flux_label_y ));

// arrows
string darkflux1 = "#57C7A9";
string darkflux2 = "#C7A557";
string darkflux3 = "#5778C7";
pen arrow_pen = linewidth(1.4pt);
real flux_arrow_rad = lat_space*0.27;
draw(arc(lat_space*(0.5, flux_label_y ), r=flux_arrow_rad, angle1=240, angle2=-60), arrow=Arrow(TeXHead), p=arrow_pen+rgb(darkflux1));
draw(arc(lat_space*(1.5, flux_label_y ), r=flux_arrow_rad, angle1=240, angle2=-60), arrow=Arrow(TeXHead), p=arrow_pen+rgb(darkflux2));
draw(arc(lat_space*(2.5, flux_label_y ), r=flux_arrow_rad, angle1=240, angle2=-60), arrow=Arrow(TeXHead), p=arrow_pen+rgb(darkflux3));



label("$\nu_j=$", lat_space*(-0.2,0.2));
label("$\varphi_j=$", lat_space*(-0.2,0.1));
string[] g_vals_top = {"$0$","g", "3g", "6g"};
for(int i=0; i<lat_Lx; ++i)
{
    label("$0$", lat_space*(i, 0.2));
    label(g_vals_top[i], lat_space*(i,0.1));
}

label("$\nu_j=$", lat_space*(-0.2,-1.1));
label("$\varphi_j=$", lat_space*(-0.2,-1.2));
string[] g_vals_bottom = {"$h$","g+h", "3g+h", "6g+h"};
for(int i=0; i<lat_Lx; ++i)
{
    label("$1$", lat_space*(i, -1.1));
    label(g_vals_bottom[i], lat_space*(i,-1.2));
}
