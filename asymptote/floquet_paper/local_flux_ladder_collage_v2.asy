settings.outformat = "pdf";
// settings.render=15;/
defaultpen(fontsize(10pt));
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
label("$\mathcal{J}_0\! \left(\! A'\! \sin\! \left(\! \frac{g}{2}\! \right)\! \right)$", lat_space*(0+0.5,horiz_tunnel_label_ygap) );
label("$\mathcal{J}_0\! \left(\! A'\! \sin\! \left(g \right)\! \right)$", lat_space*(1+0.5,horiz_tunnel_label_ygap) );
label("$\mathcal{J}_0\! \left(\! A'\! \sin\! \left(\! \frac{3g}{2}\! \right)\! \right)$", lat_space*(2+0.5,horiz_tunnel_label_ygap) );

// vertical tunnelling labels
pen fnt_size_sides = fontsize(8pt);
real y_sides_secondline = -0.12;
real y_side_labels = -0.7;

// label("$\epsilon_h$", lat_space*(0,y_side_labels) , p = fnt_size_sides, left);
// label("$\epsilon_h \times$", lat_space*(1,y_side_labels) , p = fnt_size_sides, left);
// label("$  e^{-ig}$", lat_space*(1,y_side_labels + y_sides_secondline), left,p = fnt_size_sides);
// label("$\epsilon_h \times$", lat_space*(2,y_side_labels) , p = fnt_size_sides, left);
// label("$ e^{-i3g }$", lat_space*(2,y_side_labels + y_sides_secondline), left, p = fnt_size_sides);
// label(Label("$\epsilon_h e^{-i6g}$", Rotate((0,90))), lat_space*(3,y_side_labels) , left,  p = fnt_size_sides);
// // label("$ $", lat_space*(3,y_side_labels + y_sides_secondline), left, p = fnt_size_sides);

label(Label("$\epsilon_h$", Rotate((0,90))), lat_space*(0,y_side_labels) , p = fnt_size_sides, right);
label(Label("$\epsilon_h e^{-ig}$", Rotate((0,90))), lat_space*(1,y_side_labels) , p = fnt_size_sides, left);
label(Label("$\epsilon_h e^{-i3g}$", Rotate((0,90))), lat_space*(2,y_side_labels) , p = fnt_size_sides, left);
label(Label("$\epsilon_h e^{-i6g}$", Rotate((0,90))), lat_space*(3,y_side_labels) , left,  p = fnt_size_sides);


// draw dots
int i_dot = 1;
pair label_gap_up = 0.6*(0.9,1);
pair label_gap_down = 0.6*(0.9,-1);

for (int i_y=-1; i_y<=0; ++i_y)
{ 
    for (int i_x = 0; i_x < lat_Lx; ++i_x)
    {
        dot(lat_space*(i_x, i_y));
        if (i_dot >=5)
        {
            label((string) i_dot, (i_x, i_y)*lat_space + label_gap_up, p=fontsize(9pt));
        }
        else
        {
            label((string) i_dot, (i_x, i_y)*lat_space + label_gap_down, p=fontsize(9pt));
        }
        
        i_dot = i_dot + 1;

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




// table
string[][] headers = {{"site $j$",  "$\varphi_j$", "$\nu_j$"},
                         {"$1$",  "$0$", "$0$" },
                         {"$2$",  "$g$", "$0$" },
                          {"$3$", "$3g$", "$0$" },
                          {"$4$", "$6g$", "$0$" },
                          {"$5$",  "$h$", "$1$"},
                          {"$6$",  "$g\! +\! h$", "$1$"},
                          {"$7$", "$3g\! +\! h$", "$1$"},
                          {"$8$", "$6g\! +\! h$", "$1$"}};



real table_x_space = 2.9;
real table_y_space = 1.5;
pair table_start = lat_space*(0,-1.3) + (table_x_space/2,0);
pen fnt_size_table = fontsize(7pt);
real table_border_ygap = table_y_space/2;

//draw table box
for (int i_x=0; i_x<9; ++i_x)
{ 
    for (int i_y= 0; i_y > -3; --i_y)
    {
        label(headers[i_x][-i_y], table_start+table_x_space*(i_x, 0)+ table_y_space*(0,i_y), p=fnt_size_table);
    }
}
//draw table vertical lines
draw(table_start+ (-table_x_space/2,table_border_ygap) -- table_start+(table_x_space/2,table_border_ygap)+8*(table_x_space,0) -- table_start +8*(table_x_space,0) -2*(0,table_y_space)+(table_x_space/2,-table_border_ygap)  -- table_start -2*(0,table_y_space)+(-table_x_space/2,-table_border_ygap) -- cycle);
for (int i_x = 1; i_x < 9; ++i_x)
{
    draw(table_start + table_y_space*(0,0.5) + table_x_space*(i_x-0.5,0)-- table_start + table_x_space*(i_x-0.5,0) + table_y_space*(0,-2.5));
}
//draw table horizontal lines
for (int i_y = 0; i_y < 2; ++i_y)
{
    draw(table_start +  table_x_space*(-0.5,0) + table_y_space*(0,-i_y - 0.5)-- table_start +  table_x_space*(8.5,0) +  table_y_space*(0,-i_y - 0.5));
}




// figure labels
real fig_label_x = -2.6;
real fig_label_y = 1;

pair a_label_loc = (fig_label_x,fig_label_y) + (table_x_space/2,0);
pair b_label_loc =  table_start + (fig_label_x,fig_label_y);
pair c_label_loc = b_label_loc - (0,table_y_space*4);
pair d_label_loc = c_label_loc + (14.6,0);
pair e_label_loc = c_label_loc + (0,-10.6 );


pair c_d_fig_shift = (-1,-0.6);

label("(a)", a_label_loc);
label("(b)",b_label_loc);


label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/grad_mag_field_flux_5x20_v2.pdf"), e_label_loc + (-0.5,0.5), SE);

label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/flux_ladder_flux_vals_v2.pdf"), c_label_loc + c_d_fig_shift + (0.2,0), SE);
label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/flux_ladder_abs_vals.pdf"), d_label_loc + c_d_fig_shift, SE);

label("(c)", c_label_loc);
label("(d)", d_label_loc);
label("(e)", e_label_loc);

