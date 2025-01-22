settings.outformat = "pdf";
defaultpen(fontsize(9pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;


string pale_blue = "#BBE3F2"; 
string pale_red = "#F6DFEB"  ;
string pale_orange = "#FDDDC3";

string colour1 = "1565C0";
string colour2 = "C30934";
string colour3 = "006F63";
string colour4 = "F57F17";
string colour5 = "8E24AA";


real lattice_space = 3.7;
pair v1 = lattice_space*(1,0);
pair v2 = lattice_space*(0.5, -sin(pi/3));
string[] horiz_labels = {"1", "2"};
string[] horiz_labels2 = {"3"};

pen lw = linewidth(0.8pt);
pen line_col = grey;
pen phase1_pen = rgb(pale_blue);
pen phase2_pen = rgb(pale_red);

pen fnt_size_dot_labels = fontsize(7pt);


pen lw_unitcell = linewidth(2.1pt);
pen colour_unitcell = rgb("#A91401");

// lines
for (int i_x = 0; i_x <3; ++i_x)
{
    for (int i_y = 0; i_y <3; ++i_y)
    {
        
        
        // lines
        pair loc1 = i_x*v1*2 + 2*i_y*v2; // top left position of top left triangle
        pair loc2 = loc1 + v1; // top right position of triangle;
        pair loc3 = loc1 + v2; // bottom position of triangle
        pair loc4 = loc1 + 2*v2 + v1; // bottom left position of lower right triangle
        pair loc5 = loc4 + v1; // bottom right ..
        pair loc6 = loc5 - v2; // top ...

        // colour fills
        fill(loc1 -- loc2 -- loc3 -- cycle, p=phase1_pen);
        fill(loc4 -- loc5 -- loc6 -- cycle, p=phase1_pen);
        fill(loc2 -- loc2 + v1 -- loc6 -- loc4 -- loc4 - v1 --loc3 -- cycle, p=phase2_pen);

        // lines
        draw(loc1 -- loc2, p=line_col+lw);
        draw(loc1 -- loc3, p=line_col+lw);
        draw(loc3 -- loc2, p=line_col+lw);
        draw(loc4 -- loc5, p=line_col+lw);
        draw(loc4 -- loc6, p=line_col+lw);
        draw(loc5 -- loc6, p=line_col+lw);

        
        // phase labels
        label("$\phi$", (loc1+loc2+loc3)/3);
        label("$\phi$", (loc4+loc5+loc6)/3);

        label("$-2\phi$", (loc2 + v2));


        
    }
}
int n_tunnel_fullgraph = 6;
draw((0,0)--n_tunnel_fullgraph*v2 -- n_tunnel_fullgraph*v2 + n_tunnel_fullgraph*v1 -- n_tunnel_fullgraph*v1 -- cycle, p=lw+line_col);
int x_n_tunnel_unitcell = 2;
int y_n_tunnel_unitcell = 2;
draw((0,0)  --y_n_tunnel_unitcell*v2 -- y_n_tunnel_unitcell*v2 + x_n_tunnel_unitcell*v1 -- x_n_tunnel_unitcell*v1 -- cycle, p=lw_unitcell+colour_unitcell+linetype("2 2"));

real floquet_unit_cell_gap = 0.2;
draw((0,0) +floquet_unit_cell_gap*(-v1-v2)--y_n_tunnel_unitcell*v2 +floquet_unit_cell_gap*(v2-v1)-- y_n_tunnel_unitcell*v2 + x_n_tunnel_unitcell*v1 +floquet_unit_cell_gap*(v1+v2)-- x_n_tunnel_unitcell*v1 +floquet_unit_cell_gap*(v1-v2)-- cycle, p=rgb(colour4)+linetype("2 2")+linewidth(0.8pt));

// numbering
int i_dot = 1;
pair label_shift_odd = lattice_space*(0.25,-0.15);
pair label_shift_even = lattice_space*(0.2,0);
for (int i_y=0; i_y<7; ++i_y)
{ 
    for (int i_x = 0; i_x < 7; ++i_x)
    {
        pair loc = v1*i_x + v2*i_y;
        if(i_y %2== 0) {
        // do all labels
            dot(loc);
            // label((string) i_dot, loc- label_shift, p=fnt_size_dot_labels);
            label(horiz_labels[(i_x+2*i_y)%2], loc + label_shift_odd);
            i_dot = i_dot + 1;

        } else {
            if(i_x %2 ==0) {
                dot(loc);
                // label((string) i_dot, loc-label_shift , p=fnt_size_dot_labels);
                label(horiz_labels2[(i_x+2*i_y)%1], loc + label_shift_even);
                i_dot = i_dot + 1;
            }
        }
    }
}