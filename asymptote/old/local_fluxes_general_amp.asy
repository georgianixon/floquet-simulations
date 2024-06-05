settings.outformat = "pdf";
defaultpen(fontsize(10pt));
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
string[] As = {"$A_1$", "$A_3$", "$A_1$", "$A_3$", "$A_2$", "$A_4$", "$A_2$", "$A_4$"};
string[] nus = {"$0$", "$1$", "$0$", "$1$", "$0$", "$1$", "$0$", "$1$"};
string[] varphis = {"$0$", "$0$", "$0$", "$0$", "$0$", "$\varphi_1$", "$\varphi_2$", "$\varphi_3$"};

int i_dot = 1;
pair label_gap = 0.4*(1,-1);
for (int i_y=0; i_y>-lat_Ly; --i_y)
{ 
    for (int i_x = 0; i_x < lat_Lx; ++i_x)
    {
        dot(lat_space*(i_x, i_y));
        int index = -i_y*lat_Lx + i_x;
        // label("("+As[index]+","+nus[index]+","+varphis[index]+")",lat_space*(i_x, i_y+0.1) );
        label((string) i_dot, (i_x, i_y)*lat_space + label_gap);
        i_dot = i_dot + 1;

    }
}


// for (int i_y=0; i_y>-lat_Ly; --i_y)
// { 
//     for (int i_x = 0; i_x < lat_Lx-1; ++i_x)
//     {
//         label("("+As[index]+","+nus[index]+","+varphis[index]+")",lat_space*(i_x, i_y+0.1) );
//     }
// }

//shading
fill((0,0) -- lat_space*(0,-1) -- lat_space*(1,-1) -- lat_space*(1,0) -- cycle, p=rgb("#F2D8DD"));
fill(lat_space*(1,0) -- lat_space*(1,-1) -- lat_space*(2,-1) -- lat_space*(2,0) -- cycle, p=rgb("#D8DEF2"));
fill(lat_space*(2,0) -- lat_space*(2,-1) -- lat_space*(3,-1) -- lat_space*(3,0) -- cycle, p=rgb("#F2ECD8"));


label("$-\mathcal{J}_1 (B_{21}) \mathrm{e}^{-i \xi_{21}}$", lat_space*(0+0.5,0) );
label("$\mathcal{J}_1 (B_{32}) \mathrm{e}^{i \xi_{32}}$", lat_space*(1+0.5,0) );
label("$-\mathcal{J}_1 (B_{43}) \mathrm{e}^{-i \xi_{43}}$", lat_space*(2+0.5,0) );

label("$-\mathcal{J}_1 (B_{65}) \mathrm{e}^{-i \xi_{65}}$", lat_space*(0+0.5,-1) );
label("$\mathcal{J}_1 (B_{76}) \mathrm{e}^{i \xi_{76}}$", lat_space*(1+0.5,-1) );
label("$-\mathcal{J}_1 (B_{87}) \mathrm{e}^{-i \xi_{87}}$", lat_space*(2+0.5,-1) );

label("$\mathcal{J}_0 (B_{51})$", lat_space*(0,-0.5) );
label("$\mathcal{J}_0 (B_{62})$", lat_space*(1,-0.5) );
label("$\mathcal{J}_0 (B_{73})$", lat_space*(2,-0.5) );
label("$\mathcal{J}_0 (B_{84})$", lat_space*(3,-0.5) );



// table
string[][] headers = {{"site", "$A$", "$\varphi$", "$\nu$"},
                         {"$1$", "$A_1$", "$0$", "$0$" },
                         {"$2$", "$A_2$", "$0$", "$1$" },
                          {"$3$", "$A_1$", "$0$", "$0$" },
                          {"$4$", "$A_2$", "$0$", "$1$" },
                          {"$5$", "$A_3$", "$0$", "$0$"},
                          {"$6$", "$A_4$", "$\varphi_1$", "$1$"},
                          {"$7$", "$A_3$", "$\varphi_2$", "$0$"},
                          {"$8$", "$A_4$", "$\varphi_3$", "$1$"}};



real table_x_space = 4;
real table_y_space = 2;
pair table_start = lat_space*(3.5,0);

for (int i_y=0; i_y>-9; --i_y)
{ 
    for (int i_x = 0; i_x < 4; ++i_x)
    {
        label(headers[-i_y][i_x], table_start+table_x_space*(i_x, 0)+ table_y_space*(0,i_y));
    }
}
draw(table_start+ (-1,1) -- table_start+ (1,1)+3*(table_x_space,0) -- table_start +3*(table_x_space,0) -8*(0,table_y_space)+(1,-1)  -- table_start -8*(0,table_y_space)+(-1,-1) -- cycle);



//equations
string A_eq = "$B_{ij} = \sqrt{A_i^2 + A_j^2 - 2 A_i A_j \cos (\varphi_j - \varphi_i)}$";
string xi_eq = "$\cos (\xi_{ij}) = \frac{-A_j \cos (\varphi_j) + A_i \cos (\varphi_i)}{ B_{ij}}$";
string xi_eq_sin = "$\sin (\xi_{ij}) = \frac{-A_j \sin (\varphi_j) + A_i \sin (\varphi_i)}{ B_{ij}}$";

label(A_eq, lat_space*(1.5, -1.5));
// label(xi_eq, lat_space*(1.5, -1.7));
label("$\xi_{ij} = - i \log \left( \frac{-A_j e^{i\varphi_j} + A_i e^{i \varphi_i}}{\sqrt{A_i^2 + A_j^2 - 2 A_i A_j \cos (\varphi_j - \varphi_i)}} \right)$", lat_space*(1.5,-1.7));
// label(xi_eq_sin, lat_space*(1.5, -1.9));


draw(lat_space*(0.5,-1.3) -- lat_space*(2.5,-1.3) -- lat_space*(2.5,-1.9) -- lat_space*(0.5,-1.9) -- cycle);

// fluxes
label("$\Phi_1 = -\xi_{65}$", lat_space*(0.5, -0.5));
label("$\Phi_2 = \xi_{76}$", lat_space*(1.5, -0.5));
label("$\Phi_3 = -\xi_{87}$", lat_space*(2.5, -0.5));

// label("$\cos(\xi_{56}) = \frac{-A_4 \cos (\varphi_1) + A_3 }{ \sqrt{A_3^2 + A_4^2 - 2 A_3 A_4 \cos (\varphi_1)}}$", lat_space*(0,-2.2));
// label("$\cos(\xi_{67}) \frac{-A_3 \cos (\varphi_2) + A_4 \cos (\varphi_1)}{ \sqrt{A_4^2 + A_3^2 - 2 A_3 A_4 \cos (\varphi_2 - \varphi_1)}}$", lat_space*(0,-2.4));
// label("$\sin(\xi_{56}) = \frac{-A_4 \sin (\varphi_1) }{ \sqrt{A_3^2 + A_4^2 - 2 A_3 A_4 \cos (\varphi_1)}}$", lat_space*(1.8,-2.2));
// label("$\sin(\xi_{67}) \frac{-A_3 \sin (\varphi_2) + A_4 \sin (\varphi_1)}{ \sqrt{A_4^2 + A_3^2 - 2 A_4 A_3 \cos (\varphi_2 - \varphi_1)}}$", lat_space*(1.8,-2.4));
label("$\xi_{56} = -i \log \left(  \frac{-A_4 e^{i\varphi_1} + A_3               }{\sqrt{A_3^2 + A_4^2 - 2 A_3 A_4 \cos (\varphi_1)}} \right)$", lat_space*(1.8,-2.2));
label("$\xi_{67} = - i \log \left( \frac{-A_3 e^{i\varphi_2} + A_4 e^{i \varphi_1}}{\sqrt{A_4^2 + A_3^2 - 2 A_4 A_3 \cos (\varphi_2 - \varphi_1)}} \right)$", lat_space*(1.8,-2.4));


real AieAj_text_x = 0.5;
real AieAj_text_y0 = -2.6;
real AieAj_text_ygap = -0.15;
label("If $A_i = A_j = A$", lat_space*(AieAj_text_x,AieAj_text_y0), right);
label("$A_{ij} = 2A\sin \left(\frac{\varphi_i - \varphi_j}{2} \right) $ ", lat_space*(AieAj_text_x, AieAj_text_y0+ AieAj_text_ygap), right);
label("$\xi_{ij} = \left(\frac{\varphi_i + \varphi_j}{2}\right) - \frac{\pi}{2}$", lat_space*(AieAj_text_x, AieAj_text_y0 + 2*AieAj_text_ygap), right);
