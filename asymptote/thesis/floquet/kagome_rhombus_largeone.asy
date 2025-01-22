settings.outformat = "pdf";
defaultpen(fontsize(8pt));
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


real lattice_space = 2.3;
pair v1 = lattice_space*(1,0);
pair v2 = lattice_space*(0.5, -sin(pi/3));
string[] horiz_labels_odd = {"1", "2", "1", "2", "3", "1", "3", "1", "2", "3", "2", "3"};
string[] horiz_labels_even = {"3", "3", "2", "2", "1", "1"};


string[] row1 = {"2","3","1","2","1","2","3","1","3","1","2","3"};
string[] row2 = {"1","" ,"3","" ,"3","" ,"2","" ,"2","" ,"1","" };
string[] row3 = {"3","1","2","1","2","3","1","3","1","2","3","2"};
string[] row4 = {"2","" ,"3","" ,"1","" ,"2","" ,"3","" ,"1","" };
string[] row5 = {"3","1","2","3","2","3","1","2","1","2","3","1"};
string[] row6 = {"2","" ,"1","" ,"1","" ,"3","" ,"3","" ,"2","" };
string[] row7 = {"1","2","3","2","3","1","2","1","2","3","1","3"};
string[] row8 = {"3","" ,"1","" ,"2","" ,"3","" ,"1","" ,"2","" };
string[] row9 = {"1","2","3","1","3","1","2","3","2","3","1","2"};
string[] row10 = {"3","","2","" ,"2","" ,"1","" ,"1","" ,"3","" };
string[] row11 = {"2","3","1","3","1","2","3","2","3","1","2","1"};
string[] row12 = {"1","" ,"2","" ,"3","" ,"1","" ,"2","" ,"3","" };
string[] row13 = row1;


string[][] labels = {row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12, row13};

pen lw = linewidth(0.8pt);

pen line_col = grey;
pen phase1_pen = rgb(pale_blue);
pen phase2_pen = rgb(pale_orange);
pen[] phase_pens = {phase1_pen, phase2_pen};
pen fnt_size_dot_labels = fontsize(7pt);
pen fnt_size_phi = fontsize(7pt);
pair label_shift = lattice_space*(0.23,0);

pen lw_unitcell = linewidth(2.1pt);
pen colour_unitcell = rgb("#A91401");

// lines
int phase_int = 0;
string phase_labels[] = {"$\phi$", "$-\phi$"};
for (int i_y = 0; i_y <6; ++i_y)
{
    for (int i_x = 0; i_x <6; ++i_x)
    {
        // corners
        pair loc1 = i_x*v1*2 + 2*i_y*v2; // top left position of top left triangle
        pair loc2 = loc1 + v1; // top right position of triangle;
        pair loc3 = loc1 + v2; // bottom position of triangle
        pair loc4 = loc1 + 2*v2 + v1; // bottom left position of lower right triangle
        pair loc5 = loc4 + v1; // bottom right ..
        pair loc6 = loc5 - v2; // top ...

        

        if (i_y%2==1){
            phase_int +=1;
        }

        // colour fills
        fill(loc1 -- loc2 -- loc3 -- cycle, p=phase_pens[phase_int%2]);
        fill(loc4 -- loc5 -- loc6 -- cycle, p=phase_pens[(phase_int+1)%2]);

        //lines
        draw(loc1 -- loc2, p=line_col+lw);
        draw(loc1 -- loc3, p=line_col+lw);
        draw(loc3 -- loc2, p=line_col+lw);
        draw(loc4 -- loc5, p=line_col+lw);
        draw(loc4 -- loc6, p=line_col+lw);
        draw(loc5 -- loc6, p=line_col+lw);
        // phase labels
        // label(phase_labels[(phase_int)%2], (loc1+loc2+loc3)/3);
        // label(phase_labels[(phase_int+1)%2], (loc4+loc5+loc6)/3);


        
    }
    phase_int +=1;
}
int n_tunnel_fullgraph = 12;
draw((0,0)--n_tunnel_fullgraph*v2 -- n_tunnel_fullgraph*v2 + n_tunnel_fullgraph*v1 -- n_tunnel_fullgraph*v1 -- cycle, p=lw+line_col);
int n_tunnel_unitcell = 4;
draw((0,0)--n_tunnel_unitcell*v2 -- n_tunnel_unitcell*v2 + n_tunnel_unitcell*v1 -- n_tunnel_unitcell*v1 -- cycle, p=lw_unitcell+colour_unitcell+linetype("2 2"));

int x_n_tunnel_unitcell = 12;
int y_n_tunnel_unitcell = 12;
real floquet_unit_cell_gap = 0.2;
draw((0,0) +floquet_unit_cell_gap*(-v1-v2)--y_n_tunnel_unitcell*v2 +floquet_unit_cell_gap*(v2-v1)-- y_n_tunnel_unitcell*v2 + x_n_tunnel_unitcell*v1 +floquet_unit_cell_gap*(v1+v2)-- x_n_tunnel_unitcell*v1 +floquet_unit_cell_gap*(v1-v2)-- cycle, p=rgb(colour4)+linetype("2 2")+linewidth(0.8pt));


// numbering
int i_dot = 1;
phase_int = 0;
for (int i_y=0; i_y<13; ++i_y)
{ 
    for (int i_x = 0; i_x < 13; ++i_x)
    {
        pair loc = v1*i_x + v2*i_y; // position of dot
        if(i_y %2== 0) {
        // do all labels
            dot(loc);
            // label((string) i_dot, loc- label_shift, p=fnt_size_dot_labels); // number
            label(labels[i_y%12][i_x%12], loc + label_shift);
            i_dot = i_dot + 1;

        } else {
            if(i_x %2 ==0) {
                dot(loc);
                // label((string) i_dot, loc-label_shift , p=fnt_size_dot_labels);
                label(labels[i_y%12][i_x%12], loc + label_shift);
                i_dot = i_dot + 1;
            }
        }
    }
    if (i_y %2 == 0){
        phase_int +=1;
    }
    
}