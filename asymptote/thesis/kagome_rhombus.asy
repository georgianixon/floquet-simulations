settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;

string pale_green = "#D8F2EB";
string pale_yellow = "#F2EAD8";
string pale_blue2 = "#D8E0F2";
string pale_blue = "BBDDFA"; 

real lattice_space = 4;
pair v1 = lattice_space*(1,0);
pair v2 = lattice_space*(0.5, -sin(pi/3));
string[] horiz_labels = {"A", "B", "C"};
string[] horiz_labels2 = {"C", "B", "A"};
pair label_shift = lattice_space*(0.2,0);

pen lw = linewidth(0.8pt);
pen line_col = grey;
pen phase1_pen = rgb(pale_blue);
pen phase2_pen = rgb(pale_yellow);

pen fnt_size_dot_labels = fontsize(7pt);

// lines
for (int i_x = 0; i_x <3; ++i_x)
{
    for (int i_y = 0; i_y <3; ++i_y)
    {
        // lines
        
        pair loc1 = i_x*v1*2 + 2*i_y*v2;
        pair loc2 = loc1 + v1;
        pair loc3 = loc1 + v2;
        pair loc4 = loc1 + 2*v2 + v1;
        pair loc5 = loc4 + v1;
        pair loc6 = loc5 - v2;

        draw(loc1 -- loc2, p=line_col+lw);
        draw(loc1 -- loc3, p=line_col+lw);
        draw(loc3 -- loc2, p=line_col+lw);
        draw(loc4 -- loc5, p=line_col+lw);
        draw(loc4 -- loc6, p=line_col+lw);
        draw(loc5 -- loc6, p=line_col+lw);

        // colour fills
        fill(loc1 -- loc2 -- loc3 -- cycle, p=phase1_pen);
        fill(loc4 -- loc5 -- loc6 -- cycle, p=phase2_pen);

        // phase labels
        label("$\phi$", (loc1+loc2+loc3)/3);
        label("$-\phi$", (loc4+loc5+loc6)/3);


        
    }
}
draw((0,0)--6*v2 -- 6*v2 + 6*v1 -- 6*v1 -- cycle, p=lw+line_col);

// numbering
int i_dot = 1;
for (int i_y=0; i_y<7; ++i_y)
{ 
    for (int i_x = 0; i_x < 7; ++i_x)
    {
        pair loc = v1*i_x + v2*i_y;
        if(i_y %2== 0) {
        // do all labels
            dot(loc);
            label((string) i_dot, loc- label_shift, p=fnt_size_dot_labels);
            label(horiz_labels[(i_x+2*i_y)%3], loc + label_shift);
            i_dot = i_dot + 1;

        } else {
            if(i_x %2 ==0) {
                dot(loc);
                label((string) i_dot, loc-label_shift , p=fnt_size_dot_labels);
                label(horiz_labels[(i_x+2*i_y)%3], loc + label_shift);
                i_dot = i_dot + 1;
            }
        }
    }
}