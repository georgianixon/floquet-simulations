settings.outformat = "pdf";
defaultpen(fontsize(9pt));
unitsize(3mm);
usepackage("amsfonts");

size(7cm);


string colour1 = "AD7A99"; // pink
string colour2 = "7CDEDC"; // light blue
string colour3 = "006F63"; // green
string colour4 = "F57F17"; //orange
string colour5 = "0F1980"; //purple
string colour6 = "C30934"; //red

int lat_Ly = 2; 
int lat_Lx = 4;
real lat_space = 15;
// draw dots
int i_dot = 1;
pair label_gap = 0.4*(1,-1);
for (int i_y=0; i_y>-lat_Ly; --i_y)
{ 
    for (int i_x = 0; i_x < lat_Lx; ++i_x)
    {
        dot(lat_space*(i_x, i_y));
        int index = -i_y*lat_Lx + i_x;
        label((string) i_dot, (i_x, i_y)*lat_space + label_gap);
        i_dot = i_dot + 1;

    }
}


//shading
fill((0,0) -- lat_space*(0,-1) -- lat_space*(1,-1) -- lat_space*(1,0) -- cycle, p=rgb("#F2D8DD"));
fill(lat_space*(1,0) -- lat_space*(1,-1) -- lat_space*(2,-1) -- lat_space*(2,0) -- cycle, p=rgb("#D8DEF2"));
fill(lat_space*(2,0) -- lat_space*(2,-1) -- lat_space*(3,-1) -- lat_space*(3,0) -- cycle, p=rgb("#F2ECD8"));


// label("$-\mathcal{J}_1 (B_{21}) \mathrm{e}^{-i \xi_{21}}$", lat_space*(0+0.5,0) );
// label("$\mathcal{J}_1 (B_{32}) \mathrm{e}^{i \xi_{32}}$", lat_space*(1+0.5,0) );
// label("$-\mathcal{J}_1 (B_{43}) \mathrm{e}^{-i \xi_{43}}$", lat_space*(2+0.5,0) );

label("$\mathcal{J}_0 \left( \frac{2 A}{\omega} \sin \left(\frac{g}{2}\right) \right)$", lat_space*(0+0.5,0) );
label("$\mathcal{J}_0 \left( \frac{2 A}{\omega} \sin \left(g \right) \right)$", lat_space*(1+0.5,0) );
label("$\mathcal{J}_0 \left( \frac{2 A}{\omega} \sin \left(\frac{3g}{2}\right) \right)$", lat_space*(2+0.5,0) );

label("$\mathcal{J}_0 \left( \frac{2 A}{\omega} \sin \left(\frac{g}{2}\right) \right)$", lat_space*(0+0.5,-1) );
label("$\mathcal{J}_0 \left( \frac{2 A}{\omega} \sin \left(g \right) \right)$", lat_space*(1+0.5,-1) );
label("$\mathcal{J}_0 \left( \frac{2 A}{\omega} \sin \left(\frac{3g}{2}\right) \right)$", lat_space*(2+0.5,-1) );



pen fnt_size_sides = fontsize(8pt);
real y_sides_secondline = -0.07;

label("$\mathcal{J}_1 \left(  \frac{2 A}{\omega} \sin \left(\frac{h}{2} \right) \right)$", lat_space*(0,-0.5) , p = fnt_size_sides);
label("$ \times e^{-i(h - \pi)/2}$", lat_space*(0,-0.5 + y_sides_secondline) , p = fnt_size_sides);
label("$\mathcal{J}_1 \left(  \frac{2 A}{\omega} \sin \left(\frac{h}{2} \right) \right)$", lat_space*(1,-0.5) , p = fnt_size_sides);
label("$ \times  e^{-i(2g + h - \pi)/2}$", lat_space*(1,-0.5 + y_sides_secondline), p = fnt_size_sides);
label("$\mathcal{J}_1 \left(  \frac{2 A}{\omega} \sin \left(\frac{h}{2} \right) \right)$", lat_space*(2,-0.5) , p = fnt_size_sides);
label("$ \times e^{-i(6g + h - \pi)/2}$", lat_space*(2,-0.5 + y_sides_secondline), p = fnt_size_sides);
label("$\mathcal{J}_1 \left(  \frac{2 A}{\omega} \sin \left(\frac{h}{2} \right) \right)$", lat_space*(3,-0.5) , p = fnt_size_sides);
label("$ \times e^{-i(12g + h - \pi)/2}$", lat_space*(3,-0.5 + y_sides_secondline), p = fnt_size_sides);




// table
string[][] headers = {{"site",  "$\varphi$", "$\nu$"},
                         {"$1$",  "$0$", "$0$" },
                         {"$2$",  "$g$", "$0$" },
                          {"$3$", "$3g$", "$0$" },
                          {"$4$", "$6g$", "$0$" },
                          {"$5$",  "$h$", "$1$"},
                          {"$6$",  "$g+h$", "$1$"},
                          {"$7$", "$3g+h$", "$1$"},
                          {"$8$", "$6g+h$", "$1$"}};



real table_x_space = 4;
real table_y_space = 2;
pair table_start = lat_space*(3.5,0);

for (int i_y=0; i_y>-9; --i_y)
{ 
    for (int i_x = 0; i_x < 3; ++i_x)
    {
        label(headers[-i_y][i_x], table_start+table_x_space*(i_x, 0)+ table_y_space*(0,i_y));
    }
}
draw(table_start+ (-1,1) -- table_start+ (1,1)+3*(table_x_space,0) -- table_start +3*(table_x_space,0) -8*(0,table_y_space)+(1,-1)  -- table_start -8*(0,table_y_space)+(-1,-1) -- cycle);


// fluxes
label("$\Phi_1 = g $", lat_space*(0.5, -0.3));
label("$\Phi_2 = 2g $", lat_space*(1.5, -0.3));
label("$\Phi_3 = 3g $", lat_space*(2.5, -0.3));
