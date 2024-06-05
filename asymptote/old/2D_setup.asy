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

string dot_colour1 = "361134";
string dot_colour2 = "B0228C";
string dot_colour3 = "#EA426C";
string dot_colour4 = "#EA8286";
string dot_colour5 = "#ED9CC6";
string dot_colour6 = "#DCB9F7";

// string pos_tunnelling1 = "E2123F";
// string pos_tunnelling2 = "F47B95";
// string neg_tunnelling1 = "B4C5F8";
// string neg_tunnelling2 = "577DEF";
// string neg_tunnelling3 = "1342CD";
// string neg_tunnelling4 = "0A2470";
string pos_tunnelling1 = "FAB173";
string pos_tunnelling2 = "FBCFAA";
// string pos_tunnelling2 = "FDEEE2";
// string neg_tunnelling1 = "E2EEED";
string neg_tunnelling1 = "A2CAC6";
string neg_tunnelling2 = "64A7A0";
string neg_tunnelling3 = "228277";
string neg_tunnelling4 = "1D3432";


real num_lines = 6;
real line_x0 = 0;  // top left x0
real line_y0 = 0;  // top left y0
real line_space = 3;
real line_y1 = line_y0 - num_lines*line_space;
real line_x1 = line_x0 + num_lines*line_space;

pen pl = linewidth(1.6pt);

pair centre = (line_x0 + 3*line_space,line_y0 - 3*line_space);


// DRAW LINES ############################################################################################################
for  (int i_d=-1; i_d<=1; i_d=i_d+2)
{
    //first lines
    draw(centre -- centre+ i_d*(0, line_space), p=rgb(pos_tunnelling1)+pl);
    draw(centre -- centre + i_d*(line_space,0),  p=rgb(pos_tunnelling1)+pl);

    //second tunnelling lines
    draw(centre + (0, i_d*line_space) -- centre+ (0, i_d*2*line_space), p=rgb(pos_tunnelling2)+pl);
    draw(centre + (0, i_d*line_space) -- centre+ (i_d*line_space, i_d*line_space), p=rgb(pos_tunnelling2)+pl);
    draw(centre + (0, i_d*line_space) -- centre+ (-i_d*line_space, i_d*line_space), p=rgb(pos_tunnelling2)+pl);

    draw(centre + ( i_d*line_space, 0)-- centre + (i_d*2*line_space,0),  p=rgb(pos_tunnelling2)+pl);
    draw(centre + ( i_d*line_space, 0)-- centre + (i_d*line_space,i_d*line_space),  p=rgb(pos_tunnelling2)+pl);
    draw(centre + ( i_d*line_space, 0)-- centre + (i_d*line_space,-i_d*line_space),  p=rgb(pos_tunnelling2)+pl);

    //third tunnelling lines
    draw(centre+ (0, i_d*2*line_space) -- centre+ (0, i_d*3*line_space), p=rgb(neg_tunnelling1)+pl);
    draw(centre+ (0, i_d*2*line_space) -- centre+ (i_d*line_space, i_d*2*line_space), p=rgb(neg_tunnelling1)+pl);
    draw(centre+ (0, i_d*2*line_space) -- centre+ (-i_d*line_space, i_d*2*line_space), p=rgb(neg_tunnelling1)+pl);
    draw(centre+ (i_d*line_space, i_d*line_space) -- centre + (i_d*line_space, i_d*2*line_space), p=rgb(neg_tunnelling1)+pl);
    draw(centre+ (i_d*line_space, i_d*line_space) -- centre + (i_d*2*line_space, i_d*line_space), p=rgb(neg_tunnelling1)+pl);
    draw( centre+ (-i_d*line_space, i_d*line_space) -- centre + (-i_d*line_space, i_d*2*line_space), p=rgb(neg_tunnelling1)+pl);
    draw( centre+ (-i_d*line_space, i_d*line_space) -- centre + (-i_d*2*line_space, i_d*line_space), p=rgb(neg_tunnelling1)+pl);

    draw( centre+ (i_d*2*line_space, 0) -- centre + (i_d*3*line_space, 0), p=rgb(neg_tunnelling1)+pl);
    draw( centre+ (i_d*2*line_space, 0) -- centre + (i_d*2*line_space, i_d*line_space), p=rgb(neg_tunnelling1)+pl);
    draw( centre+ (i_d*2*line_space, 0) -- centre + (i_d*2*line_space, -i_d*line_space), p=rgb(neg_tunnelling1)+pl);
    draw( centre + (i_d*line_space,-i_d*line_space) -- centre + (i_d*2*line_space,-i_d*line_space),  p=rgb(neg_tunnelling1)+pl);
    draw( centre + (i_d*line_space,-i_d*line_space) -- centre + (i_d*line_space,-i_d*2*line_space),  p=rgb(neg_tunnelling1)+pl);


    // fourth tunnelling lines
    draw(centre+  i_d*line_space*(0, 3) -- centre+ i_d*line_space*(1,3), p=rgb(neg_tunnelling2)+pl);
    draw(centre+  i_d*line_space*(0, 3) -- centre+ i_d*line_space*(-1,3), p=rgb(neg_tunnelling2)+pl);
    draw(centre+  i_d*line_space*(3, 0) -- centre+ i_d*line_space*(3,1), p=rgb(neg_tunnelling2)+pl);
    draw(centre+  i_d*line_space*(3, 0) -- centre+ i_d*line_space*(3,-1), p=rgb(neg_tunnelling2)+pl);
    draw(centre+  i_d*line_space*(1, 2) -- centre+ i_d*line_space*(1,3), p=rgb(neg_tunnelling2)+pl);
    draw(centre+  i_d*line_space*(1, 2) -- centre+ i_d*line_space*(2,2), p=rgb(neg_tunnelling2)+pl);
    draw(centre+  i_d*line_space*(1, -2) -- centre+ i_d*line_space*(1,-3), p=rgb(neg_tunnelling2)+pl);
    draw(centre+  i_d*line_space*(1, -2) -- centre+ i_d*line_space*(2,-2), p=rgb(neg_tunnelling2)+pl);
    draw(centre+  i_d*line_space*(2, 1) -- centre+ i_d*line_space*(2,2), p=rgb(neg_tunnelling2)+pl);
    draw(centre+  i_d*line_space*(2, 1) -- centre+ i_d*line_space*(3,1), p=rgb(neg_tunnelling2)+pl);
    draw(centre+  i_d*line_space*(2, -1) -- centre+ i_d*line_space*(2,-2), p=rgb(neg_tunnelling2)+pl);
    draw(centre+  i_d*line_space*(2, -1) -- centre+ i_d*line_space*(3,-1), p=rgb(neg_tunnelling2)+pl);

    //fifth tunnelling
    draw(centre+  i_d*line_space*(3, 1) -- centre+ i_d*line_space*(3,2), p=rgb(neg_tunnelling3)+pl);
    draw(centre+  i_d*line_space*(3, -1) -- centre+ i_d*line_space*(3,-2), p=rgb(neg_tunnelling3)+pl);
    draw(centre+  i_d*line_space*(1,3) -- centre+ i_d*line_space*(2,3), p=rgb(neg_tunnelling3)+pl);
    draw(centre+  i_d*line_space*( 1,-3) -- centre+ i_d*line_space*(2, -3), p=rgb(neg_tunnelling3)+pl);
    draw(centre+  i_d*line_space*(2, 2) -- centre+ i_d*line_space*(3,2), p=rgb(neg_tunnelling3)+pl);
    draw(centre+  i_d*line_space*(2, 2) -- centre+ i_d*line_space*(2, 3), p=rgb(neg_tunnelling3)+pl);
    draw(centre+  i_d*line_space*(2, -2) -- centre+ i_d*line_space*(3,-2), p=rgb(neg_tunnelling3)+pl);
    draw(centre+  i_d*line_space*(2, -2) -- centre+ i_d*line_space*(2, -3), p=rgb(neg_tunnelling3)+pl);

    //sixth tunnelling
    draw(centre+  i_d*line_space*(3, 2) -- centre+ i_d*line_space*(3, 3), p=rgb(neg_tunnelling4)+pl);
    draw(centre+  i_d*line_space*( 2, 3) -- centre+ i_d*line_space*(3, 3), p=rgb(neg_tunnelling4)+pl);
    draw(centre+  i_d*line_space*( 2,-3) -- centre+ i_d*line_space*(3,-3), p=rgb(neg_tunnelling4)+pl);
    draw(centre+  i_d*line_space*(3,-2) -- centre+ i_d*line_space*(3, -3), p=rgb(neg_tunnelling4)+pl);
}

dot(centre, grey);

// for (int i_d=1; i_d<=num_lines-1; ++i_d)
// {
//     draw((line_x0, line_y0-i_d*line_space) -- (line_x1, line_y0-i_d*line_space), p=pl);
//     draw((line_x0+i_d*line_space, line_y0) -- (line_x0+ i_d*line_space, line_y1),  p=pl);
// }



// DRAW DOTS ############################################################################################################
real poly_size = 0.45;
filldraw(circle(centre,poly_size-0.1),grey,grey);

for  (int i_d=-1; i_d<=1; i_d=i_d+2)
{
    //first dot
    // dot(centre + (0,i_d*line_space), p=rgb(dot_colour1));
    // dot(centre + (i_d*line_space, 0), p=rgb(dot_colour1));
    fill(shift(centre + line_space*i_d*(0,1))*scale(poly_size)*polygon(3), p = rgb(dot_colour1));
    fill(shift(centre + line_space*i_d*(1,0))*scale(poly_size)*polygon(3), p = rgb(dot_colour1));

    // second dots
    fill(shift(centre + line_space*i_d*(1,1))*scale(poly_size)*polygon(4), p = rgb(dot_colour2));
    fill(shift(centre + line_space*i_d*(-1,1))*scale(poly_size)*polygon(4), p = rgb(dot_colour2));
    fill(shift(centre + line_space*i_d*(2,0))*scale(poly_size)*polygon(4), p = rgb(dot_colour2));
    fill(shift(centre + line_space*i_d*(0,2))*scale(poly_size)*polygon(4), p = rgb(dot_colour2));
    // dot(centre+(-i_d*line_space,i_d*line_space), p=rgb(dot_colour2));
    // dot(centre+(i_d*line_space,i_d*line_space), p=rgb(dot_colour2));
    // dot(centre+(i_d*2*line_space,0), p=rgb(dot_colour2));
    // dot(centre+(0,i_d*2*line_space), p=rgb(dot_colour2));

    //third dots
    // dot(centre+(i_d*3*line_space,0), p=rgb(dot_colour3));
    // dot(centre+(0,i_d*3*line_space), p=rgb(dot_colour3));
    // dot(centre+(i_d*line_space,i_d*2*line_space), p=rgb(dot_colour3));
    // dot(centre+(i_d*2*line_space,i_d*line_space), p=rgb(dot_colour3));
    // dot(centre+(i_d*2*line_space,-i_d*line_space), p=rgb(dot_colour3));
    // dot(centre+(i_d*line_space,-i_d*2*line_space), p=rgb(dot_colour3));
    fill(shift(centre + line_space*i_d*(3,0))*scale(poly_size)*polygon(5), p = rgb(dot_colour3));
    fill(shift(centre + line_space*i_d*(0,3))*scale(poly_size)*polygon(5), p = rgb(dot_colour3));
    fill(shift(centre + line_space*i_d*(1,2))*scale(poly_size)*polygon(5), p = rgb(dot_colour3));
    fill(shift(centre + line_space*i_d*(2,1))*scale(poly_size)*polygon(5), p = rgb(dot_colour3));
    fill(shift(centre + line_space*i_d*(2,-1))*scale(poly_size)*polygon(5), p = rgb(dot_colour3));
    fill(shift(centre + line_space*i_d*(1,-2))*scale(poly_size)*polygon(5), p = rgb(dot_colour3));

    //fourth dots
    // dot(centre+(0,i_d*4*line_space), p=rgb(dot_colour4));
    // dot(centre+(i_d*4*line_space,0), p=rgb(dot_colour4));
    // dot(centre+(i_d*line_space,i_d*3*line_space), p=rgb(dot_colour4));
    // dot(centre+(i_d*2*line_space,i_d*2*line_space), p=rgb(dot_colour4));
    // dot(centre+(i_d*3*line_space,i_d*line_space), p=rgb(dot_colour4));
    // dot(centre+(i_d*3*line_space,-i_d*line_space), p=rgb(dot_colour4));
    // dot(centre+(i_d*line_space,-i_d*3*line_space), p=rgb(dot_colour4));
    // dot(centre+(i_d*2*line_space,-i_d*2*line_space), p=rgb(dot_colour4));

    // filldraw(circle(centre+(0,i_d*4*line_space),poly_size),rgb(dot_colour4),rgb(dot_colour4));
    // filldraw(circle(centre+(i_d*4*line_space,0),poly_size),rgb(dot_colour4),rgb(dot_colour4));
    filldraw(circle(centre+(i_d*line_space,i_d*3*line_space),poly_size-0.1),rgb(dot_colour4),rgb(dot_colour4));
    filldraw(circle(centre+(i_d*2*line_space,i_d*2*line_space),poly_size-0.1),rgb(dot_colour4),rgb(dot_colour4));
    filldraw(circle(centre+(i_d*3*line_space,i_d*line_space),poly_size-0.1),rgb(dot_colour4),rgb(dot_colour4));
    filldraw(circle(centre+(i_d*3*line_space,-i_d*line_space),poly_size-0.1),rgb(dot_colour4),rgb(dot_colour4));
    filldraw(circle(centre+(i_d*2*line_space,-i_d*2*line_space),poly_size-0.1),rgb(dot_colour4),rgb(dot_colour4));
    filldraw(circle(centre+(i_d*line_space,-i_d*3*line_space),poly_size-0.1),rgb(dot_colour4),rgb(dot_colour4));


    //fifth dots
    // dot(centre+(i_d*2*line_space,i_d*3*line_space), p=rgb(dot_colour5));
    // dot(centre+(i_d*3*line_space,i_d*2*line_space), p=rgb(dot_colour5));
    // dot(centre+(i_d*3*line_space,-i_d*2*line_space), p=rgb(dot_colour5));
    // dot(centre+(i_d*2*line_space,-i_d*3*line_space), p=rgb(dot_colour5));
    fill(shift(centre + line_space*i_d*(2,3))*scale(poly_size)*polygon(3), p = rgb(dot_colour5));
    fill(shift(centre + line_space*i_d*(3,2))*scale(poly_size)*polygon(3), p = rgb(dot_colour5));
    fill(shift(centre + line_space*i_d*(3,-2))*scale(poly_size)*polygon(3), p = rgb(dot_colour5));
    fill(shift(centre + line_space*i_d*(2,-3))*scale(poly_size)*polygon(3), p = rgb(dot_colour5));

    //sixth dots
    // dot(centre+(i_d*3*line_space,i_d*3*line_space), p=rgb(dot_colour6));
    // dot(centre+(i_d*3*line_space,-i_d*3*line_space), p=rgb(dot_colour6));
    fill(shift(centre + line_space*i_d*(3,-3))*scale(poly_size)*polygon(4), p = rgb(dot_colour6));
    fill(shift(centre + line_space*i_d*(3,3))*scale(poly_size)*polygon(4), p = rgb(dot_colour6));
}




// DRAW LEGEND1 ############################################################################################################

pair label_shift = (0.3,0.4);
// label("$A_1$", centre+label_shift+(0,line_space),  p=rgb(colour1)+fontsize(5pt));
// label("$A_2$", centre+label_shift+(line_space, line_space),  p=rgb(colour2)+fontsize(5pt));
// label("$A_3$", centre+label_shift+(line_space, 2*line_space),  p=rgb(colour3)+fontsize(5pt));
// label("$A_4$", centre+label_shift+(2*line_space, 2*line_space),  p=rgb(colour4)+fontsize(5pt));
// label("$A_5$", centre+label_shift+(2*line_space, 3*line_space),  p=rgb(colour5)+fontsize(5pt));
// label("$A_6$", centre+label_shift+(3*line_space, 3*line_space),  p=rgb(colour6)+fontsize(5pt));


pair legend_gaps = (0, 2);
pair legend_start_pos = (-3, -1);
pair legend1_loc = (line_x0 ,line_y0) + legend_start_pos;

// dot(legend1_loc, grey);
filldraw(circle(legend1_loc,poly_size-0.1),grey,grey);

// dot(legend1_loc - legend_gaps, p = rgb(dot_colour1));
fill(shift(legend1_loc - legend_gaps)*scale(poly_size)*polygon(3), p = rgb(dot_colour1));
// dot(legend1_loc - 2*legend_gaps, p = rgb(dot_colour2));
fill(shift(legend1_loc - 2*legend_gaps)*scale(poly_size)*polygon(4), p = rgb(dot_colour2));
// dot(legend1_loc - 3*legend_gaps, p = rgb(dot_colour3));
fill(shift(legend1_loc - 3*legend_gaps)*scale(poly_size)*polygon(5), p = rgb(dot_colour3));
// dot(legend1_loc - 4*legend_gaps, p = rgb(dot_colour4));
filldraw(circle(legend1_loc - 4*legend_gaps,poly_size-0.1),rgb(dot_colour4),rgb(dot_colour4));

// dot(legend1_loc - 5*legend_gaps, p = rgb(dot_colour5));
fill(shift(legend1_loc - 5*legend_gaps)*scale(poly_size)*polygon(3), p = rgb(dot_colour5));
// dot(legend1_loc - 6*legend_gaps, p = rgb(dot_colour6));
fill(shift(legend1_loc - 6*legend_gaps)*scale(poly_size)*polygon(4), p = rgb(dot_colour6));


pair legend1_A_loc = legend1_loc + (1,0);
label("$A_0$", legend1_A_loc);
label("$A_1$", legend1_A_loc - legend_gaps);
label("$A_2$", legend1_A_loc - 2*legend_gaps);
label("$A_3$", legend1_A_loc - 3*legend_gaps);
label("$A_4$", legend1_A_loc - 4*legend_gaps);
label("$A_5$", legend1_A_loc - 5*legend_gaps);
label("$A_6$", legend1_A_loc - 6*legend_gaps);








// DRAW LEGEND2 ############################################################################################################

pair legend2_loc =  (line_x1, line_y0) +  (1,-1);
real line_len = 1;

draw(legend2_loc -- legend2_loc+ line_len*(1,0), p=rgb(pos_tunnelling1)+pl);
draw(legend2_loc - legend_gaps -- legend2_loc-legend_gaps +line_len*(1,0), p=rgb(pos_tunnelling2)+pl);

draw(legend2_loc - 3*legend_gaps -- legend2_loc- 3*legend_gaps +line_len*(1,0), p=rgb(neg_tunnelling1)+pl);
draw(legend2_loc - 4*legend_gaps -- legend2_loc- 4*legend_gaps +line_len*(1,0), p=rgb(neg_tunnelling2)+pl);
draw(legend2_loc - 5*legend_gaps -- legend2_loc- 5*legend_gaps +line_len*(1,0), p=rgb(neg_tunnelling3)+pl);
draw(legend2_loc - 6*legend_gaps -- legend2_loc- 6*legend_gaps +line_len*(1,0), p=rgb(neg_tunnelling4)+pl);

pair legend2_Jloc = legend2_loc + (1.8,0);
label("$J_1$", legend2_Jloc);
label("$J_2$", legend2_Jloc - legend_gaps);

label("$J_3$", legend2_Jloc - 3*legend_gaps);
label("$J_4$", legend2_Jloc - 4*legend_gaps);
label("$J_5$", legend2_Jloc - 5*legend_gaps);
label("$J_6$", legend2_Jloc - 6*legend_gaps);

pair brace_loc = legend2_Jloc + (0.8,0);
draw(brace(brace_loc,brace_loc - legend_gaps, 0.5));
draw(brace(brace_loc- 3*legend_gaps,brace_loc - 6*legend_gaps, 0.5));

pair ge_loc = brace_loc + (1.5,0);
label("$<0$", ge_loc - 0.5*legend_gaps );
label("$>0$", ge_loc - 4.5*legend_gaps);


// picture pic1;
// real size=3;
// size(pic1,size);
// fill(pic1,(0,0)--(50,100)--(100,0)--cycle,red);

// picture pic2;
// size(pic2,size);
// fill(pic2,unitcircle,green);

// picture pic3;
// size(pic3,size);
// fill(pic3,unitsquare,blue);

// picture pic;
// add(pic,pic1.fit(),(0,0),N);


