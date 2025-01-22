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

pen tun_pen = linewidth(0.7pt);

//shading
// tunnellings vertical

real horiz_tunnel_label_ygap = -0.1;
// vertical tunnelling labels
pen fnt_size_sides = fontsize(8pt);
real y_sides_secondline = -0.12;
real y_side_labels = -0.7;

label("$J_x$", lat_space*(0.5, 0.14));
label("$J_y$", lat_space*(-0.14, -0.5));

// draw tunnellings
real tun_gap = 0.6;
// tunnellings horizontal
for (int i_x = 0; i_x < lat_Lx - 1; ++i_x)
{
    for (int i_y = 0; i_y > -lat_Ly; --i_y)
    {
        draw((i_x,i_y)*lat_space+ (tun_gap, 0) -- (i_x+1, i_y)*lat_space - (tun_gap,0), p=tun_pen);
    }
}

for (int i_x = 0; i_x < lat_Lx; ++i_x)
{
    for (int i_y = 0; i_y > -lat_Ly + 1; --i_y)
    {
        draw((i_x, i_y-1)*lat_space + (0, tun_gap) -- (i_x,i_y)*lat_space - (0, tun_gap), p=tun_pen);
    }
}


// draw dots
int i_dot = 0;
string varphi_vals[] =  {"$0\!+\!3h$","$g\!+\!3h$","$3g\!+\!3h$", "$6g\!+\!3h$","$0\!+\!2h$","$g\!+\!2h$","$3g\!+\!2h$", "$6g\!+\!2h$","$0\!+\!h$","$g\!+\!h$","$3g\!+\!h$", "$6g\!+\!h$", "$0$","$g$","$3g$", "$6g$"};
pair label_gap = 0.3*(0.6,-1.8);
real arrow_size = 1.2;
pen lw_arrows = linewidth(1.2pt);
pen lw_tunnellings = linewidth(0.1pt);
real arrow_head_size = 4;
pen varphi_fontsize=fontsize(7pt);
for (int i_y=-(lat_Ly-1); i_y<=0; ++i_y)
{ 
    for (int i_x = 0; i_x < lat_Lx; ++i_x)
    {
        pair spot = lat_space*(i_x, i_y);
        if (i_y % 2 ==1)
        {
            draw(spot - arrow_size*(0,-1)  -- spot+arrow_size*(0,-1), p=rgb(colour6)+lw_arrows, arrow=ArcArrows(SimpleHead, size=arrow_head_size));
            dot(spot);//, p=rgb(colour4));
        }
        else
        {
            draw(spot - arrow_size*(0,-1)  -- spot+arrow_size*(0,-1), p=rgb(colour6)+lw_arrows, arrow=ArcArrows(SimpleHead, size=arrow_head_size));
            dot(lat_space*(i_x, i_y));
        }
        label(varphi_vals[i_dot], (i_x, i_y)*lat_space + label_gap, p=varphi_fontsize, E);
        i_dot = i_dot + 1;
    }
}



