settings.outformat = "pdf";
// settings.render=10;
defaultpen(fontsize(9pt));
//defaultpen(arrowsize(9));
//defaultpen(arrowsize(5bp));
unitsize(3mm);
settings.tex="pdflatex" ;


import graph;
//size(7cm);

//-0.37242316  0.85520254 -0.51623059  1.        
// string colour1 = "1565C0";
// string colour2 = "C30934";
// string colour3 = "006F63";
// string colour4 = "F57F17";
// string colour5 = "8E24AA";

// #F8923B
// #FAB173
// #FBCFAA
// #FDEEE2
// #E2EEED
// #A2CAC6
// #64A7A0
// #228277

string dot_colour1 = "#10050F";
string dot_colour2 = "52194F";
string dot_colour3 = "B0228C";
string dot_colour4 = "#EA426C";
string dot_colour4 = "E11246";
string dot_colour4 = "E311CA";

string dot_colour5 = "#E9297F";
string dot_colour6 = "#EF62A1";
string dot_colour7 = "#F07997";
string dot_colour8 = "#F17B79";
string dot_colour9 = "#E3A2A1";

// string pos_tunnelling1 = "E2123F";
// string pos_tunnelling2 = "F47B95";
// string neg_tunnelling1 = "B4C5F8";
// string neg_tunnelling2 = "577DEF";
// string neg_tunnelling3 = "1342CD";
// string neg_tunnelling4 = "0A2470";
string tunnelling1 = "#D96908";
string tunnelling2 = "#F78826";
string tunnelling3 = "FAB173";
string tunnelling4 = "FBCFAA";

string tunnelling5 = "A2CAC6";
string tunnelling6 = "64A7A0";
string tunnelling7 = "228277";
string tunnelling8 = "1D3432";


real num_lines = 6;
real line_x0 = 0;  // top left x0
real line_y0 = 0;  // top left y0
real line_space = 2.2;
real line_y1 = line_y0 - num_lines*line_space;
real line_x1 = line_x0 + num_lines*line_space;

pen pl = linewidth(1.6pt);

pair centre = (line_x0 + 3*line_space,line_y0 - 3*line_space);


// DRAW LINES ############################################################################################################
for  (int i_d=-1; i_d<=1; i_d=i_d+2)
{
    //first lines
    draw(centre -- centre+ i_d*(0, line_space), p=rgb(tunnelling1)+pl);
    draw(centre -- centre + i_d*(line_space,0),  p=rgb(tunnelling1)+pl);

    //second tunnelling lines
    draw(centre + (0, i_d*line_space) -- centre+ (0, i_d*2*line_space), p=rgb(tunnelling2)+pl);
    draw(centre + (0, i_d*line_space) -- centre+ (i_d*line_space, i_d*line_space), p=rgb(tunnelling2)+pl);
    draw(centre + (0, i_d*line_space) -- centre+ (-i_d*line_space, i_d*line_space), p=rgb(tunnelling2)+pl);

    draw(centre + ( i_d*line_space, 0)-- centre + (i_d*2*line_space,0),  p=rgb(tunnelling2)+pl);
    draw(centre + ( i_d*line_space, 0)-- centre + (i_d*line_space,i_d*line_space),  p=rgb(tunnelling2)+pl);
    draw(centre + ( i_d*line_space, 0)-- centre + (i_d*line_space,-i_d*line_space),  p=rgb(tunnelling2)+pl);

    //third tunnelling lines
    draw(centre+ (0, i_d*2*line_space) -- centre+ (0, i_d*3*line_space), p=rgb(tunnelling3)+pl);
    draw(centre+ (0, i_d*2*line_space) -- centre+ (i_d*line_space, i_d*2*line_space), p=rgb(tunnelling3)+pl);
    draw(centre+ (0, i_d*2*line_space) -- centre+ (-i_d*line_space, i_d*2*line_space), p=rgb(tunnelling3)+pl);
    draw(centre+ (i_d*line_space, i_d*line_space) -- centre + (i_d*line_space, i_d*2*line_space), p=rgb(tunnelling3)+pl);
    draw(centre+ (i_d*line_space, i_d*line_space) -- centre + (i_d*2*line_space, i_d*line_space), p=rgb(tunnelling3)+pl);
    draw( centre+ (-i_d*line_space, i_d*line_space) -- centre + (-i_d*line_space, i_d*2*line_space), p=rgb(tunnelling3)+pl);
    draw( centre+ (-i_d*line_space, i_d*line_space) -- centre + (-i_d*2*line_space, i_d*line_space), p=rgb(tunnelling3)+pl);

    draw( centre+ (i_d*2*line_space, 0) -- centre + (i_d*3*line_space, 0), p=rgb(tunnelling3)+pl);
    draw( centre+ (i_d*2*line_space, 0) -- centre + (i_d*2*line_space, i_d*line_space), p=rgb(tunnelling3)+pl);
    draw( centre+ (i_d*2*line_space, 0) -- centre + (i_d*2*line_space, -i_d*line_space), p=rgb(tunnelling3)+pl);
    draw( centre + (i_d*line_space,-i_d*line_space) -- centre + (i_d*2*line_space,-i_d*line_space),  p=rgb(tunnelling3)+pl);
    draw( centre + (i_d*line_space,-i_d*line_space) -- centre + (i_d*line_space,-i_d*2*line_space),  p=rgb(tunnelling3)+pl);


    // fourth tunnelling lines
    // draw(centre+  i_d*line_space*(0, 3) -- centre+ i_d*line_space*(1,3), p=rgb(tunnelling4)+pl);
    // draw(centre+  i_d*line_space*(0, 3) -- centre+ i_d*line_space*(-1,3), p=rgb(tunnelling4)+pl);
    // draw(centre+  i_d*line_space*(3, 0) -- centre+ i_d*line_space*(3,1), p=rgb(tunnelling4)+pl);
    // draw(centre+  i_d*line_space*(3, 0) -- centre+ i_d*line_space*(3,-1), p=rgb(tunnelling4)+pl);
    // draw(centre+  i_d*line_space*(3, 0) -- centre+ i_d*line_space*(4,0), p=rgb(tunnelling4)+pl);
    // draw(centre+  i_d*line_space*( 0,3) -- centre+ i_d*line_space*(0,4), p=rgb(tunnelling4)+pl);

    // draw(centre+  i_d*line_space*(1, 2) -- centre+ i_d*line_space*(1,3), p=rgb(tunnelling4)+pl);
    // draw(centre+  i_d*line_space*(1, 2) -- centre+ i_d*line_space*(2,2), p=rgb(tunnelling4)+pl);
    // draw(centre+  i_d*line_space*(1, -2) -- centre+ i_d*line_space*(1,-3), p=rgb(tunnelling4)+pl);
    // draw(centre+  i_d*line_space*(1, -2) -- centre+ i_d*line_space*(2,-2), p=rgb(tunnelling4)+pl);
    // draw(centre+  i_d*line_space*(2, 1) -- centre+ i_d*line_space*(2,2), p=rgb(tunnelling4)+pl);
    // draw(centre+  i_d*line_space*(2, 1) -- centre+ i_d*line_space*(3,1), p=rgb(tunnelling4)+pl);
    // draw(centre+  i_d*line_space*(2, -1) -- centre+ i_d*line_space*(2,-2), p=rgb(tunnelling4)+pl);
    // draw(centre+  i_d*line_space*(2, -1) -- centre+ i_d*line_space*(3,-1), p=rgb(tunnelling4)+pl);

    int n_tot_l = 3;
    for  (int i_l=0; i_l<=n_tot_l; i_l=i_l+1)
    {   
        int j_l = n_tot_l - i_l;
        draw(centre+  i_d*line_space*(i_l, j_l) -- centre+ i_d*line_space*(i_l,j_l +1), p=rgb(tunnelling4)+pl);
        draw(centre+  i_d*line_space*(i_l, j_l) -- centre+ i_d*line_space*(i_l+1,j_l ), p=rgb(tunnelling4)+pl);
        draw(centre+  i_d*line_space*(i_l, -j_l) -- centre+ i_d*line_space*(i_l,-j_l -1), p=rgb(tunnelling4)+pl);
        draw(centre+  i_d*line_space*(i_l, -j_l) -- centre+ i_d*line_space*(i_l+1,-j_l), p=rgb(tunnelling4)+pl);
    }
    
    

    //fifth tunnelling
    // draw(centre+  i_d*line_space*(3, 1) -- centre+ i_d*line_space*(3,2), p=rgb(tunnelling5)+pl);
    // draw(centre+  i_d*line_space*(3, -1) -- centre+ i_d*line_space*(3,-2), p=rgb(tunnelling5)+pl);
    // draw(centre+  i_d*line_space*(1,3) -- centre+ i_d*line_space*(2,3), p=rgb(tunnelling5)+pl);
    // draw(centre+  i_d*line_space*( 1,-3) -- centre+ i_d*line_space*(2, -3), p=rgb(tunnelling5)+pl);
    // draw(centre+  i_d*line_space*(2, 2) -- centre+ i_d*line_space*(3,2), p=rgb(tunnelling5)+pl);
    // draw(centre+  i_d*line_space*(2, 2) -- centre+ i_d*line_space*(2, 3), p=rgb(tunnelling5)+pl);
    // draw(centre+  i_d*line_space*(2, -2) -- centre+ i_d*line_space*(3,-2), p=rgb(tunnelling5)+pl);
    // draw(centre+  i_d*line_space*(2, -2) -- centre+ i_d*line_space*(2, -3), p=rgb(tunnelling5)+pl);

    int n_tot_l = 4;
    for  (int i_l=0; i_l<=n_tot_l; i_l=i_l+1)
    {   
        int j_l = n_tot_l - i_l;
        draw(centre+  i_d*line_space*(i_l, j_l) -- centre+ i_d*line_space*(i_l,j_l +1), p=rgb(tunnelling5)+pl);
        draw(centre+  i_d*line_space*(i_l, j_l) -- centre+ i_d*line_space*(i_l+1,j_l ), p=rgb(tunnelling5)+pl);
        draw(centre+  i_d*line_space*(i_l, -j_l) -- centre+ i_d*line_space*(i_l,-j_l -1), p=rgb(tunnelling5)+pl);
        draw(centre+  i_d*line_space*(i_l, -j_l) -- centre+ i_d*line_space*(i_l+1,-j_l), p=rgb(tunnelling5)+pl);
    }
    

    //sixth tunnelling
    // draw(centre+  i_d*line_space*(3, 2) -- centre+ i_d*line_space*(3, 3), p=rgb(tunnelling6)+pl);
    // draw(centre+  i_d*line_space*( 2, 3) -- centre+ i_d*line_space*(3, 3), p=rgb(tunnelling6)+pl);
    // draw(centre+  i_d*line_space*( 2,-3) -- centre+ i_d*line_space*(3,-3), p=rgb(tunnelling6)+pl);
    // draw(centre+  i_d*line_space*(3,-2) -- centre+ i_d*line_space*(3, -3), p=rgb(tunnelling6)+pl);

    int n_tot_l = 5;
    for  (int i_l=0; i_l<=n_tot_l; i_l=i_l+1)
    {   
        int j_l = n_tot_l - i_l;
        draw(centre+  i_d*line_space*(i_l, j_l) -- centre+ i_d*line_space*(i_l,j_l +1), p=rgb(tunnelling6)+pl);
        draw(centre+  i_d*line_space*(i_l, j_l) -- centre+ i_d*line_space*(i_l+1,j_l ), p=rgb(tunnelling6)+pl);
        draw(centre+  i_d*line_space*(i_l, -j_l) -- centre+ i_d*line_space*(i_l,-j_l -1), p=rgb(tunnelling6)+pl);
        draw(centre+  i_d*line_space*(i_l, -j_l) -- centre+ i_d*line_space*(i_l+1,-j_l), p=rgb(tunnelling6)+pl);
    }

    int n_tot_l = 6;
    for  (int i_l=0; i_l<=n_tot_l; i_l=i_l+1)
    {   
        int j_l = n_tot_l - i_l;
        draw(centre+  i_d*line_space*(i_l, j_l) -- centre+ i_d*line_space*(i_l,j_l +1), p=rgb(tunnelling7)+pl);
        draw(centre+  i_d*line_space*(i_l, j_l) -- centre+ i_d*line_space*(i_l+1,j_l ), p=rgb(tunnelling7)+pl);
        draw(centre+  i_d*line_space*(i_l, -j_l) -- centre+ i_d*line_space*(i_l,-j_l -1), p=rgb(tunnelling7)+pl);
        draw(centre+  i_d*line_space*(i_l, -j_l) -- centre+ i_d*line_space*(i_l+1,-j_l), p=rgb(tunnelling7)+pl);
    }

    int n_tot_l = 7;
    for  (int i_l=0; i_l<=n_tot_l; i_l=i_l+1)
    {   
        int j_l = n_tot_l - i_l;
        draw(centre+  i_d*line_space*(i_l, j_l) -- centre+ i_d*line_space*(i_l,j_l +1), p=rgb(tunnelling8)+pl);
        draw(centre+  i_d*line_space*(i_l, j_l) -- centre+ i_d*line_space*(i_l+1,j_l ), p=rgb(tunnelling8)+pl);
        draw(centre+  i_d*line_space*(i_l, -j_l) -- centre+ i_d*line_space*(i_l,-j_l -1), p=rgb(tunnelling8)+pl);
        draw(centre+  i_d*line_space*(i_l, -j_l) -- centre+ i_d*line_space*(i_l+1,-j_l), p=rgb(tunnelling8)+pl);
    }
}

dot(centre, grey);

// for (int i_d=1; i_d<=num_lines-1; ++i_d)
// {
//     draw((line_x0, line_y0-i_d*line_space) -- (line_x1, line_y0-i_d*line_space), p=pl);
//     draw((line_x0+i_d*line_space, line_y0) -- (line_x0+ i_d*line_space, line_y1),  p=pl);
// }



// DRAW DOTS ############################################################################################################
real poly_size = 0.45;
filldraw(circle(centre,poly_size-0.1), rgb(dot_colour1), rgb(dot_colour1));

for  (int i_d=-1; i_d<=1; i_d=i_d+2)
{
    //first dot
    fill(shift(centre + line_space*i_d*(0,1))*scale(poly_size)*polygon(3), p = rgb(dot_colour2));
    fill(shift(centre + line_space*i_d*(1,0))*scale(poly_size)*polygon(3), p = rgb(dot_colour2));

    // second dots
    fill(shift(centre + line_space*i_d*(1,1))*scale(poly_size)*polygon(4), p = rgb(dot_colour3));
    fill(shift(centre + line_space*i_d*(-1,1))*scale(poly_size)*polygon(4), p = rgb(dot_colour3));
    fill(shift(centre + line_space*i_d*(2,0))*scale(poly_size)*polygon(4), p = rgb(dot_colour3));
    fill(shift(centre + line_space*i_d*(0,2))*scale(poly_size)*polygon(4), p = rgb(dot_colour3));

    //third dots
    fill(shift(centre + line_space*i_d*(3,0))*scale(poly_size)*polygon(5), p = rgb(dot_colour4));
    fill(shift(centre + line_space*i_d*(0,3))*scale(poly_size)*polygon(5), p = rgb(dot_colour4));
    fill(shift(centre + line_space*i_d*(1,2))*scale(poly_size)*polygon(5), p = rgb(dot_colour4));
    fill(shift(centre + line_space*i_d*(2,1))*scale(poly_size)*polygon(5), p = rgb(dot_colour4));
    fill(shift(centre + line_space*i_d*(2,-1))*scale(poly_size)*polygon(5), p = rgb(dot_colour4));
    fill(shift(centre + line_space*i_d*(1,-2))*scale(poly_size)*polygon(5), p = rgb(dot_colour4));

    //fourth dots
    filldraw(circle(centre+(i_d*line_space,i_d*3*line_space),poly_size-0.1),rgb(dot_colour5),rgb(dot_colour5));
    filldraw(circle(centre+(i_d*2*line_space,i_d*2*line_space),poly_size-0.1),rgb(dot_colour5),rgb(dot_colour5));
    filldraw(circle(centre+(i_d*3*line_space,i_d*line_space),poly_size-0.1),rgb(dot_colour5),rgb(dot_colour5));
    filldraw(circle(centre+(i_d*3*line_space,-i_d*line_space),poly_size-0.1),rgb(dot_colour5),rgb(dot_colour5));
    filldraw(circle(centre+(i_d*2*line_space,-i_d*2*line_space),poly_size-0.1),rgb(dot_colour5),rgb(dot_colour5));
    filldraw(circle(centre+(i_d*line_space,-i_d*3*line_space),poly_size-0.1),rgb(dot_colour5),rgb(dot_colour5));
    filldraw(circle(centre+i_d*line_space*(0,4),poly_size-0.1),rgb(dot_colour5),rgb(dot_colour5));
    filldraw(circle(centre+i_d*line_space*(4,0),poly_size-0.1),rgb(dot_colour5),rgb(dot_colour5));



    //fifth dots
    fill(shift(centre + line_space*i_d*(2,3))*scale(poly_size)*polygon(3), p = rgb(dot_colour6));
    fill(shift(centre + line_space*i_d*(3,2))*scale(poly_size)*polygon(3), p = rgb(dot_colour6));
    fill(shift(centre + line_space*i_d*(3,-2))*scale(poly_size)*polygon(3), p = rgb(dot_colour6));
    fill(shift(centre + line_space*i_d*(2,-3))*scale(poly_size)*polygon(3), p = rgb(dot_colour6));
    fill(shift(centre + line_space*i_d*(4,1))*scale(poly_size)*polygon(3), p = rgb(dot_colour6));
    fill(shift(centre + line_space*i_d*(4,-1))*scale(poly_size)*polygon(3), p = rgb(dot_colour6));
    fill(shift(centre + line_space*i_d*(1,4))*scale(poly_size)*polygon(3), p = rgb(dot_colour6));
    fill(shift(centre + line_space*i_d*(1,-4))*scale(poly_size)*polygon(3), p = rgb(dot_colour6));

    fill(shift(centre + line_space*i_d*(5,0))*scale(poly_size)*polygon(3), p = rgb(dot_colour6));
    fill(shift(centre + line_space*i_d*(0,5))*scale(poly_size)*polygon(3), p = rgb(dot_colour6));


    //sixth dots
    fill(shift(centre + line_space*i_d*(3,-3))*scale(poly_size)*polygon(4), p = rgb(dot_colour7));
    fill(shift(centre + line_space*i_d*(3,3))*scale(poly_size)*polygon(4), p = rgb(dot_colour7));
    fill(shift(centre + line_space*i_d*(4,2))*scale(poly_size)*polygon(4), p = rgb(dot_colour7));
    fill(shift(centre + line_space*i_d*(4,-2))*scale(poly_size)*polygon(4), p = rgb(dot_colour7));
    fill(shift(centre + line_space*i_d*(2,4))*scale(poly_size)*polygon(4), p = rgb(dot_colour7));
    fill(shift(centre + line_space*i_d*(2,-4))*scale(poly_size)*polygon(4), p = rgb(dot_colour7));

    fill(shift(centre + line_space*i_d*(5,1))*scale(poly_size)*polygon(4), p = rgb(dot_colour7));
    fill(shift(centre + line_space*i_d*(5,-1))*scale(poly_size)*polygon(4), p = rgb(dot_colour7));
    fill(shift(centre + line_space*i_d*(1,5))*scale(poly_size)*polygon(4), p = rgb(dot_colour7));
    fill(shift(centre + line_space*i_d*(1,-5))*scale(poly_size)*polygon(4), p = rgb(dot_colour7));
    fill(shift(centre + line_space*i_d*(6,0))*scale(poly_size)*polygon(4), p = rgb(dot_colour7));
    fill(shift(centre + line_space*i_d*(0,6))*scale(poly_size)*polygon(4), p = rgb(dot_colour7));

    //seventh dots
    fill(shift(centre + line_space*i_d*(3,4))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
    fill(shift(centre + line_space*i_d*(3,-4))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
    fill(shift(centre + line_space*i_d*(4,3))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
    fill(shift(centre + line_space*i_d*(4,-3))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));

    fill(shift(centre + line_space*i_d*(5,2))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
    fill(shift(centre + line_space*i_d*(5,-2))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
    fill(shift(centre + line_space*i_d*(2,5))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
    fill(shift(centre + line_space*i_d*(2,-5))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
    fill(shift(centre + line_space*i_d*(6,1))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
    fill(shift(centre + line_space*i_d*(6,-1))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
    fill(shift(centre + line_space*i_d*(1,6))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
    fill(shift(centre + line_space*i_d*(1,-6))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
    fill(shift(centre + line_space*i_d*(0,7))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
    fill(shift(centre + line_space*i_d*(7,0))*scale(poly_size)*polygon(5), p = rgb(dot_colour8));

    //ninth dots
    filldraw(circle(centre+i_d*line_space*(4,4),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(4,-4),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));

    filldraw(circle(centre+i_d*line_space*(5,3),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(5,-3),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(3,5),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(3,-5),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(6,2),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(6,-2),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(2,6),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(2,-6),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(7,1),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(7,-1),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(1,7),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(1,-7),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(8,0),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));
    filldraw(circle(centre+i_d*line_space*(0,8),poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));



}




// // DRAW LEGEND1 ############################################################################################################

pair label_shift = (0.3,0.4);
pair legend_gaps = (0, 2);
pair legend_start_pos = centre +  line_space*(-7.5, 8);
pair legend1_loc = (line_x0 ,line_y0) + legend_start_pos;

filldraw(circle(legend1_loc,poly_size-0.1),rgb(dot_colour1),rgb(dot_colour1));
fill(shift(legend1_loc - legend_gaps)*scale(poly_size)*polygon(3), p = rgb(dot_colour2));
fill(shift(legend1_loc - 2*legend_gaps)*scale(poly_size)*polygon(4), p = rgb(dot_colour3));
fill(shift(legend1_loc - 3*legend_gaps)*scale(poly_size)*polygon(5), p = rgb(dot_colour4));
filldraw(circle(legend1_loc - 4*legend_gaps,poly_size-0.1),rgb(dot_colour4),rgb(dot_colour5));
pair legend1_loc_second_row_shift = (3,-1);
fill(shift(legend1_loc + legend1_loc_second_row_shift - 0*legend_gaps)*scale(poly_size)*polygon(3), p = rgb(dot_colour6));
fill(shift(legend1_loc + legend1_loc_second_row_shift - 1*legend_gaps)*scale(poly_size)*polygon(4), p = rgb(dot_colour7));
fill(shift(legend1_loc +legend1_loc_second_row_shift - 2*legend_gaps)*scale(poly_size)*polygon(5), p = rgb(dot_colour8));
filldraw(circle(legend1_loc +legend1_loc_second_row_shift - 3*legend_gaps,poly_size-0.1),rgb(dot_colour9),rgb(dot_colour9));


pair legend1_A_loc = legend1_loc + (1,0);
label("$A_1$", legend1_A_loc);
label("$A_2$", legend1_A_loc - legend_gaps);
label("$A_3$", legend1_A_loc - 2*legend_gaps);
label("$A_4$", legend1_A_loc - 3*legend_gaps);
label("$A_5$", legend1_A_loc - 4*legend_gaps);

label("$A_6$", legend1_A_loc + legend1_loc_second_row_shift- 0*legend_gaps);
label("$A_7$", legend1_A_loc + legend1_loc_second_row_shift- 1*legend_gaps);
label("$A_8$", legend1_A_loc + legend1_loc_second_row_shift- 2*legend_gaps);
label("$A_9$", legend1_A_loc + legend1_loc_second_row_shift- 3*legend_gaps);








// // DRAW LEGEND2 ############################################################################################################

pair legend2_loc =  centre +  line_space*(3.5, 8);
real line_len = 1;

draw(legend2_loc -- legend2_loc+ line_len*(1,0), p=rgb(tunnelling1)+pl);
draw(legend2_loc - legend_gaps -- legend2_loc-legend_gaps +line_len*(1,0), p=rgb(tunnelling2)+pl);
draw(legend2_loc - 2*legend_gaps -- legend2_loc- 2*legend_gaps +line_len*(1,0), p=rgb(tunnelling3)+pl);
draw(legend2_loc - 3*legend_gaps -- legend2_loc- 3*legend_gaps +line_len*(1,0), p=rgb(tunnelling4)+pl);

pair legend2_loc_second_row_shift = (6,0);
draw(legend2_loc + legend2_loc_second_row_shift - 0*legend_gaps -- legend2_loc + legend2_loc_second_row_shift - 0*legend_gaps +line_len*(1,0), p=rgb(tunnelling5)+pl);
draw(legend2_loc + legend2_loc_second_row_shift - 1*legend_gaps -- legend2_loc + legend2_loc_second_row_shift - 1*legend_gaps +line_len*(1,0), p=rgb(tunnelling6)+pl);
draw(legend2_loc + legend2_loc_second_row_shift - 2*legend_gaps -- legend2_loc + legend2_loc_second_row_shift - 2*legend_gaps +line_len*(1,0), p=rgb(tunnelling7)+pl);
draw(legend2_loc + legend2_loc_second_row_shift - 3*legend_gaps -- legend2_loc + legend2_loc_second_row_shift - 3*legend_gaps +line_len*(1,0), p=rgb(tunnelling8)+pl);


pair legend2_Jloc = legend2_loc + (1.8,0);
label("$J_1$", legend2_Jloc);
label("$J_2$", legend2_Jloc - legend_gaps);
label("$J_3$", legend2_Jloc - 2*legend_gaps);
label("$J_4$", legend2_Jloc - 3*legend_gaps);
label("$J_5$", legend2_Jloc + legend2_loc_second_row_shift - 0*legend_gaps);
label("$J_6$", legend2_Jloc + legend2_loc_second_row_shift - 1*legend_gaps);
label("$J_7$", legend2_Jloc + legend2_loc_second_row_shift - 2*legend_gaps);
label("$J_8$", legend2_Jloc + legend2_loc_second_row_shift - 3*legend_gaps);


pair brace_loc = legend2_Jloc + (0.8,0);
draw(brace(brace_loc,brace_loc - 3*legend_gaps, 0.5));
draw(brace(brace_loc + legend2_loc_second_row_shift- 0*legend_gaps,brace_loc  + legend2_loc_second_row_shift - 3*legend_gaps, 0.5));

pair ge_loc = brace_loc + (1.5,0);
label("$<0$", ge_loc - 1.5*legend_gaps );
label("$>0$", ge_loc + legend2_loc_second_row_shift - 1.5*legend_gaps);

