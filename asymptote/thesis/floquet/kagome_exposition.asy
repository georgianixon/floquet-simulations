settings.outformat = "pdf";
// settings.render = 10;
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

real lattice_space = 4.5;
pair v1 = lattice_space*(1,0);
pair v2 = lattice_space*(0.5, -sin(pi/3));
pair v3 = v2 - v1;
string[] horiz_labels = {"A", "B", "C"};
string[] horiz_labels2 = {"C", "B", "A"};

pen lw = linewidth(0.7pt);
pen line_col = grey;
pen phase1_pen = rgb(pale_blue);
pen phase2_pen = rgb(pale_red);

pen fnt_size_dot_labels = fontsize(7pt);


pen lw_unitcell = linewidth(2.1pt);
pen colour_unitcell = rgb("#A91401");
real arrow_loc = 0.54;
real arrow_size = 4;
pen arrow_pen = rgb(colour1);
pen dotsize=linewidth(3.5pt);


label("(a)",(0,0) - v2 - v1);
pair b_loc =  2.7*v1 + 2.7*v2;
draw(b_loc+ 0.5*(v1+2*v2) - (0,4)--b_loc+ 0.5*(v1+2*v2) + (0,4), arrow=ArcArrow(SimpleHead, size=arrow_size),p=rgb("B7b7b7")+linetype("2 2") );
draw(b_loc+ 0.5*(v1+2*v2) - (4,0)--b_loc+ 0.5*(v1+2*v2) + (4,0), arrow=ArcArrow(SimpleHead, size=arrow_size),p=rgb("B7b7b7")+linetype("2 2") );

label("(b)",b_loc);
draw(b_loc + 0.5*(v1+v2) -- b_loc + 0.5*(2*v1+v2) -- b_loc + 0.5*(2*v1+2*v2) -- b_loc + 0.5*(2*v1+2*v2+v3) -- b_loc + 0.5*(v1+2*v2 + v3) -- b_loc + 0.5*(v1+v2+ v3) -- cycle,p=rgb(colour1));
label ("$\Gamma$", b_loc+ 0.5*(v1+2*v2));
// dot(b_loc+ 0.5*(v1+2*v2));
// draw(b_loc+ 0.5*(v1+2*v2) -- b_loc+ 0.5*(3*v1+2*v2), arrow=ArcArrow(SimpleHead, size=arrow_size),p=rgb(colour4));
// draw(b_loc+ 0.5*(v1+2*v2) -- b_loc+ 0.5*(v1+4*v2), arrow=ArcArrow(SimpleHead, size=arrow_size),p=rgb(colour4));
// draw(b_loc+ 0.5*(v1+2*v2) -- b_loc+ 0.5*(v1+4*v2), arrow=ArcArrow(SimpleHead, size=arrow_size),p=rgb(colour4));
label("K", b_loc  + 0.5*(v1+v2) -0.15*v2 );
label("K'", b_loc + 0.5*(2*v1+ v2) - 0.15*v3);
label("$k_x$", b_loc+ 0.5*(v1+2*v2) + (4,0)+(0.4,-0.6));
label("$k_y$", b_loc+ 0.5*(v1+2*v2) + (0,4)+(0.6,0.4));

// lines
for (int i_star_array = 0; i_star_array < 1; ++i_star_array)
{
    for (int i_x = 0; i_x <2; ++i_x)
    {
        for (int i_y = 0; i_y >-2; --i_y)
        {
            
            
            // lines
            pair loc1 = i_x*v1*2 - 2*i_y*(v2-v1) + i_star_array*2*(v2); // central pos
            pair loc2 = loc1 + v1; // top right position of triangle;
            pair loc3 = loc1 + v1 - v2; // bottom position of triangle
            pair loc12 = loc1 - v1; // bottom left position of lower right triangle
            pair loc10 = loc1 - v1 + v2; // bottom right ..
            pair loc5 = loc2 + v1;
            pair loc4 = loc2 + v2;
            pair loc6 = loc1 + 2*v2;
            pair loc7 = loc6 + v1;
            pair loc8 = loc10 + v2;
            pair loc9 = loc6 - v1 + v2;
            pair loc11 = loc8 - v1;
            // colour fills
            // fill(loc1 -- loc2 -- loc3 -- cycle, p=phase1_pen);
            // fill(loc1 -- loc12 -- loc10 -- cycle, p=phase1_pen);
            // fill(loc2 -- loc5 -- loc4 -- cycle, p=phase1_pen);
            // fill(loc6 -- loc7 -- loc4 -- cycle,p=phase1_pen);
            // fill(loc8 -- loc6 -- loc9 -- cycle,p=phase1_pen);
            // fill(loc10 -- loc8 -- loc11 -- cycle, p=phase1_pen);
            // fill(loc1 -- loc2-- loc4 -- loc6 -- loc8 --loc10 -- cycle, p=phase2_pen);

        }
    }
}


// lines

for (int i_star_array = 0; i_star_array < 1; ++i_star_array)
{
    for (int i_x = 0; i_x <2; ++i_x)
    {
        for (int i_y = 0; i_y >-2; --i_y)
        {    
            // lines
            pair loc1 = i_x*v1*2 - 2*i_y*(v2-v1) + i_star_array*2*(v2); // central pos
            pair loc2 = loc1 + v1; // top right position of triangle;
            pair loc3 = loc1 + v1 - v2; // bottom position of triangle
            pair loc12 = loc1 - v1; // bottom left position of lower right triangle
            pair loc10 = loc1 - v1 + v2; // bottom right ..
            pair loc5 = loc2 + v1;
            pair loc4 = loc2 + v2;
            pair loc6 = loc1 + 2*v2;
            pair loc7 = loc6 + v1;
            pair loc8 = loc10 + v2;
            pair loc9 = loc6 - v1 + v2;
            pair loc11 = loc8 - v1;
    

            // lines
            draw(loc1-- loc2, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc2-- loc3, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc3-- loc1, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc2-- loc4, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc4-- loc5, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc5-- loc2, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc6-- loc7, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc7-- loc4, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc4-- loc6, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc6-- loc8, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc8-- loc9, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc9-- loc6, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc8-- loc10, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc10-- loc11, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc11-- loc8, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc10-- loc1, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc1-- loc12, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
            draw(loc12-- loc10, p=arrow_pen+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));


            //labels
            // label("$\phi$", (loc1+loc2+loc3)/3);
            // label("$\phi$", (loc2+loc4+loc5)/3);
            // label("$\phi$", (loc4+loc6+loc7)/3);
            // label("$\phi$", (loc6+loc8+loc9)/3);
            // label("$\phi$", (loc8+loc10+loc11)/3);
            // label("$\phi$", (loc10+loc1+loc12)/3);
            // label("$-2\phi$", (loc1 + loc2 + loc4 + loc6 + loc8 + loc10 )/6);

            //dots
            dot(loc1, p=dotsize);
            dot(loc2, p=dotsize);
            dot(loc3, p=dotsize);
            dot(loc4, p=dotsize);
            dot(loc5, p=dotsize);
            dot(loc6, p=dotsize);
            dot(loc7, p=dotsize);
            dot(loc8, p=dotsize);
            dot(loc9, p=dotsize);
            dot(loc10, p=dotsize);
            dot(loc11, p=dotsize);
            dot(loc12, p=dotsize);
        }
    }
}


pair dot6 =  2*v2;
pair dot7 = dot6 + v1;
pair dot4 = dot7-v2;

draw(dot6-- dot7, p=rgb("FFFFFF")+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
draw(dot7 --dot4, p=rgb("FFFFFF")+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));
draw(dot4-- dot6, p=rgb("FFFFFF")+lw, arrow=ArcArrow(SimpleHead, Relative(arrow_loc), size=arrow_size));


// label("A", dot6 + (0.5,-0.7));
// label("B", dot6 + v1+ (-0.5,-0.7));
// label("C", dot6 + v1-v2 , (1.3,0));

real gap_x = 0.1;
real space_y = 0.3;
string vec_colour = "000000";
draw(dot6  -- dot7, p=rgb(vec_colour)+lw, arrow=ArcArrow( size=arrow_size));
draw(dot7--dot4 , p=rgb(vec_colour)+lw, arrow=ArcArrow(size=arrow_size));
draw(dot4--dot6 , p=rgb(vec_colour)+lw, arrow=ArcArrow(size=arrow_size));

// draw dots A, B, and C
dot(dot6,p=linewidth(5pt)+rgb(colour4));
dot(dot4,p=linewidth(5pt)+rgb(colour5));
dot(dot7,p=linewidth(5pt)+rgb(colour2));

real label_shift_ABC = 0.22;
label("$A$", dot6 + v2*label_shift_ABC);
label("$C$", dot4 + v1*label_shift_ABC);
label("$B$", dot7 + v3*label_shift_ABC);

label("$\delta_{AB}$", dot6 + v1/2 - 2*(0,space_y));
label("$\delta_{BC}$", dot7 - v2/2 + 4*space_y*(sin(pi/3),0.5));
label("$\delta_{CA}$", dot4 + v3/2 + 1.2*(-sin(pi/3),0.5));

pair dot8 =  dot6 - v1;
pair dot9 = dot8 + v2;
label("$J\mathrm{e}^{i \varphi}$", dot6 - v1/2 + 3*(0,space_y), p=arrow_pen);
label("$J\mathrm{e}^{i \varphi}$", dot8 + v2/2 - 4*space_y*(sin(pi/3),0.5), p=arrow_pen);
label("$J\mathrm{e}^{i \varphi}$", dot9 - v3/2 - 1.2*(-sin(pi/3),0.5),p=arrow_pen);
// int n_tunnel_fullgraph = 4;
// draw((0,0)--n_tunnel_fullgraph*v2 -- n_tunnel_fullgraph*v2 + n_tunnel_fullgraph*v1 -- n_tunnel_fullgraph*v1 -- cycle, p=lw+line_col);
// int x_n_tunnel_unitcell = 2;
// int y_n_tunnel_unitcell = 2;
// draw((0,0)--y_n_tunnel_unitcell*v2 -- y_n_tunnel_unitcell*v2 + x_n_tunnel_unitcell*v1 -- x_n_tunnel_unitcell*v1 -- cycle, p=lw_unitcell+colour_unitcell+linetype("2 2"));
// // numbering
// int i_dot = 1;
// pair label_shift_odd = lattice_space*(0.25,-0.15);
// pair label_shift_even = lattice_space*(0.2,0);
// for (int i_y=0; i_y<5; ++i_y)
// { 
//     for (int i_x = 0; i_x < 5; ++i_x)
//     {
//         pair loc = v1*i_x + v2*i_y;
//         if(i_y %2== 0) {
//         // do all labels
//             dot(loc);
//             // label((string) i_dot, loc- label_shift, p=fnt_size_dot_labels);
//             label(horiz_labels[(i_x+2*i_y)%3], loc + label_shift_odd);
//             i_dot = i_dot + 1;

//         } else {
//             if(i_x %2 ==0) {
//                 dot(loc);
//                 // label((string) i_dot, loc-label_shift , p=fnt_size_dot_labels);
//                 label(horiz_labels[(i_x+2*i_y)%3], loc + label_shift_even);
//                 i_dot = i_dot + 1;
//             }
//         }
//     }
// }