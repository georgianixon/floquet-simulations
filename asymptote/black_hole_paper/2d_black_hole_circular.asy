// settings.outformat = "pdf";

settings.outformat = "png";
settings.render=10;

defaultpen(fontsize(9pt));

//defaultpen(arrowsize(9));
//defaultpen(arrowsize(5bp));
unitsize(3mm);
settings.tex="pdflatex" ;


pair centre = (0,0);

string colour1 = "#CC222C";
string colour2 = "#FB7936";
string colour3 = "#FEA500";
string colour4 = "#EDD69E";
string colour5 = "#E9A799";
string colour6 = "#897396";
string colour7 = "#0F4C83";
string colour8 = "#57C9E2";
string colour9 = "#A4E0BE";
string colour10 = "#4A5335";
string colour11 = "#788DA6";
string colour12 = "#272E3C";
string colour13 = "#A19897";
string colour14 = "#9C4721";


string colour1 = "#FF1212";
string colour2 = "#EB7317";
string colour3 = "#F7BA00";
string colour4 = "#F7EB00";
string colour5 = "#A0F76B";
string colour6 = "#65EF9A";
string colour7 = "#5BF3DF";
string colour8 = "#0F87FF";
string colour9 = "#0230D9";
string colour10 = "#9933FF";
string colour11 = "#FF40F9";
string colour12 = "#FF458C";
string colour13 = "#FFCCE0";
string colour14 = "#FEA764";
string colour15 = "#FFCB30";
string colour16 = "#FCF519";
string colour17 = "#9EFF66";

// // DOTS
real dot_spacing = 3;

//box
draw(box(dot_spacing*(-0.9,-0.9), dot_spacing*10.5*(1,1)));

// ARC
// draw(arc(centre, 6*dot_spacing, 0, 90), p=rgb("FF2722"));
draw(arc(centre, 8*dot_spacing, 0, 90), p=rgb("#E0E0E0")+linewidth(1.5pt)+linetype("2 2"));

// AXES
pen axes_pen = rgb("#424242")+linewidth(0.4pt);
real axes_arrow_head_size = 6;
real axes_origin_x = -0.5;
real axes_origin_y = -0.5;
pair axes_origin = (axes_origin_x,axes_origin_y);
real axes_length = 1;
draw(axes_origin*dot_spacing -- (axes_origin+(0,axes_length))*dot_spacing,  p=axes_pen, arrow=ArcArrow(SimpleHead, size=axes_arrow_head_size));
draw(axes_origin*dot_spacing -- (axes_origin+(axes_length, 0))*dot_spacing,  p=axes_pen, arrow=ArcArrow(SimpleHead, size=axes_arrow_head_size));
label("$x$", (axes_origin+axes_length*(1,0)+(0.17,0))*dot_spacing, p=fontsize(7pt));
label("$y$", (axes_origin+axes_length*(0,1)+(0,0.17))*dot_spacing, p=fontsize(7pt));
real axes_tick_length = 0.1;
draw((0,axes_origin_y)*dot_spacing -- (0,axes_origin_y - axes_tick_length)*dot_spacing, p=axes_pen);
draw((axes_origin_x, 0)*dot_spacing -- (axes_origin_x - axes_tick_length, 0)*dot_spacing, p=axes_pen);
label("$0$", (0,axes_origin_y - 2.5*axes_tick_length)*dot_spacing, p=fontsize(7pt));
label("$0$", (axes_origin_y - 2.5*axes_tick_length, 0)*dot_spacing, p=fontsize(7pt));



// Tunnelling stepwise 45^*
pair[] diagonal_tunnelling_indices= {(0,0), (0,1), (1, 1), (1, 2), (2,2),  (2,3), (3,3), (3,4), (4,4), (4,5), (5,5), (5,6), (6,6), (6, 7), (7,7), (7,8), (8,8)};
for (int i_h=0; i_h<diagonal_tunnelling_indices.length - 1; ++i_h)
{
    draw(diagonal_tunnelling_indices[i_h]*dot_spacing -- diagonal_tunnelling_indices[i_h+1]*dot_spacing, p=rgb("#9BA2FF")+linewidth(1.5pt));
}
// Tunnelling axes
for (int i_ax=0; i_ax<10; ++i_ax)
{   
    draw((i_ax,0)*dot_spacing -- (i_ax+1, 0)*dot_spacing, p=rgb("#FFB69C")+linewidth(1.5pt));
}

// // for  (int i_x=0; i_x<=10; i_x=i_x+1)
// // {
// //     for (int i_y=0; i_y<=10; i_y=i_y+1)
// //     {
// //         dot(centre + i_x*dot_spacing*(1,0)+ i_y*dot_spacing*(0,1));

// //     }
// // }

// //SHAKES
// pair[] l1={(0,0)};
// pair[] l2 = {(1,0), (0,1)};
// pair[] l3 = {(2,0), (1,1), (0,2)};
// pair[] l4 = {(3,0), (2,1), (1,2), (0,3)};
// pair[] l5 = {(3,1), (2,2), (1,3)};
// pair[] l6 = {(5,0), (4,1), (3,2), (2,3), (1,4), (0,5)};
// pair[] l7 = {(4,0), (4,2), (3,3), (2,4), (0,4)};
// pair[] l8 = {(6,1), (5,2), (4,3), (3,4), (2,5), (1,6)};
// pair[] l9 = {(6,0), (5,1), (5,3), (4,4), (3,5), (1,5), (0,6)};
// pair[] l10 = {(7,0), (7, 2), (6,3), (5,4), (4,5), (3,6), (2,7), (0,7)};
// pair[] l11 = {(7,1), (6, 2), (6,4), (5,5), (4,6), (2,6), (1,7)};
// pair[] l12 = {(8,1), (8,3), (7,4), (6,5), (5,6), (4,7), (3,8), (1,8)};
// pair[] l13 = {(8,0), (9,1), (8,2),(9,3), (7,3), (8,4), (7,5), (6,6), (5, 7), (4,8), (3,7), (3,9), (2,8), (1,9), (0,8)};
// pair[] l14 = {(9,0), (9,2), (9,4), (8,5), (7,6), (6,7), (5,8), (4,9), (2,9), (0,9)};
// pair[] l15 = {(9, 5), (8,6), (7,7), (6,8), (5,9)};
// pair[] l16 = {(10,1), (10, 3), (10,5), (9,6), (8,7), (7,8), (6,9), (5,10), (3,10), (1,10)};
// pair[] l17 = {(10,0), (10,2), (10,4), (10,6), (9,7), (8,8), (7,9), (6,10), (4,10), (2,10), (0,10)};

//new shakes
pair[] l1={(0,0)};
pair[] l2 = {(1,0), (0,1)};
pair[] l3 = {(2,0), (1,1), (0,2)};
pair[] l4 = { (2,1), (1,2)};
pair[] l5 = {(3,1), (2,2), (1,3)};
pair[] l6 = {(3,0),  (4,1), (3,2), (2,3), (1,4),  (0,3)};
pair[] l7 = {(4,0), (4,2), (3,3), (2,4), (0,4)};
pair[] l8 = {(5,0), (5,2), (4,3), (3,4), (2,5),(0,5)};
pair[] l9 = {(6,0), (5,1), (5,3), (4,4), (3,5), (1,5), (0,6)};
pair[] l10 = {(7,0), (7, 2), (6,1), (6,3), (5,4), (4,5), (3,6), (2,7), (1,6), (0,7)};
pair[] l11 = {(7,1), (6, 2), (6,4), (5,5), (4,6), (2,6), (1,7)};
pair[] l12 = {(8,1), (8,3), (7,4), (6,5), (5,6), (4,7), (3,8), (1,8)};
pair[] l13 = { (8,0),(9,1), (8,2),(9,3), (7,3), (8,4), (7,5), (6,6), (5,7), (4,8), (3,7), (3,9), (2,8), (1,9),(0,8)};
pair[] l14 = {(9,0), (9,2), (9,4), (8,5), (7,6), (6,7), (5,8), (4,9), (2,9), (0,9)};
pair[] l15 = {(10,0), (10,2), (9, 5), (8,6), (7,7), (6,8), (5,9), (2,10), (0,10)};
pair[] l16 = {(10,1), (10, 3), (10,5),  (9,6), (8,7), (7,8), (6,9), (5,10), (3,10), (1,10)};
pair[] l17 = {  (10,4), (9,7), (8,8), (7,9), (4,10)};


pair[][] lst = {l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17}; 
string[] drive_colours = {colour1, colour2, colour3, colour4, colour5, colour6, colour7, colour8, colour9, colour10, colour11, colour12, colour13, colour14, colour15, colour16, colour17};
real shake_height = 0.9;
pen lw = linewidth(2.1pt);
real arrow_head_size = 6;

for (int i_l=0; i_l<lst.length; ++i_l)
{
    real lst_size = lst[i_l].length;
    for (int j_l=0; j_l <lst_size; ++j_l)
    {
        pair spot = dot_spacing*lst[i_l][j_l];
        dot(spot);


        if(i_l %2== 1) {
            draw(spot -- spot+shake_height*(0,1), p=rgb(drive_colours[i_l])+lw);
            draw(spot -- spot-shake_height*(0,1), p=rgb(drive_colours[i_l])+lw, arrow=ArcArrow(SimpleHead, size=arrow_head_size));
        }   else {
            draw(spot -- spot+shake_height*(0,1), p=rgb(drive_colours[i_l])+lw, arrow=ArcArrow(SimpleHead, size=arrow_head_size));
            draw(spot -- spot-shake_height*(0,1), p=rgb(drive_colours[i_l])+lw);
        }
    }
}

pair up_drive_label = (1,0.5);
pair down_drive_label = (-1,0.5); 

label("$A_1$", centre + up_drive_label);
label("$A_2$", centre + dot_spacing*(0,1)+down_drive_label);
label("$A_3$", centre + dot_spacing*(1,1) + up_drive_label);
label("$A_4$", centre + dot_spacing*(1,2) +  down_drive_label);
label("$A_5$", centre + dot_spacing*(2,2) + up_drive_label);
label("$A_6$", centre + dot_spacing*(2,3) +  down_drive_label);
label("$A_7$", centre + dot_spacing*(3,3) + up_drive_label);
label("$A_8$", centre + dot_spacing*(3,4) +  down_drive_label);
label("$A_9$", centre + dot_spacing*(4,4) +  up_drive_label);
label("$A_{10}$", centre + dot_spacing*(4,5) + down_drive_label);
label("$A_{11}$", centre + dot_spacing*(5,5) + up_drive_label);
label("$A_{12}$", centre + dot_spacing*(5,6) + down_drive_label);
label("$A_{13}$", centre + dot_spacing*(6,6) +  up_drive_label);
label("$A_{14}$", centre + dot_spacing*(6,7) +  down_drive_label);
label("$A_{15}$", centre + dot_spacing*(7,7) + up_drive_label);
label("$A_{16}$", centre + dot_spacing*(7,8) + down_drive_label);
label("$A_{17}$", centre + dot_spacing*(8,8)+ up_drive_label);


// circles
real circle_size = 0.4;
string circle_colour = "#616161";
draw(circle(dot_spacing*(1,9),dot_spacing*circle_size), p=rgb(circle_colour));
draw(circle(dot_spacing*(3,9),dot_spacing*circle_size), p=rgb(circle_colour));
draw(circle(dot_spacing*(9,1),dot_spacing*circle_size), p=rgb(circle_colour));
draw(circle(dot_spacing*(9,3),dot_spacing*circle_size), p=rgb(circle_colour));
draw(circle(dot_spacing*(4,8),dot_spacing*circle_size), p=rgb(circle_colour));
draw(circle(dot_spacing*(8,4),dot_spacing*circle_size), p=rgb(circle_colour));


real b_c_label_x_coord = 11;
real b_c_fig_x_coord = 14.3;
real a_b_label_y_coord = 10.4;

pair b_label_loc = dot_spacing*(b_c_label_x_coord,a_b_label_y_coord);
pair b_fig_loc = dot_spacing*(b_c_fig_x_coord+0.1,7.6);
pair a_label_loc = dot_spacing*(-1.3,a_b_label_y_coord);
label(graphic("/home/gnixon/floquet-simulations/figures/black_hole_paper/a_vals_alternating_2D.pdf"),b_fig_loc);
label("(b)", b_label_loc);
label("(a)", a_label_loc);


pair c_label_loc = dot_spacing*(b_c_label_x_coord,4.5);
pair c_fig_loc = dot_spacing*(b_c_fig_x_coord, 1.9);
label(graphic("/home/gnixon/floquet-simulations/figures/black_hole_paper/linear_tunnelling_2D.pdf"),c_fig_loc);
label("(c)", c_label_loc);
