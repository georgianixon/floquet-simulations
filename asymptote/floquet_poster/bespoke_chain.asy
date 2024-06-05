settings.outformat = "png";
settings.render=20;
defaultpen(fontsize(10pt));
unitsize(3mm);
//size(7cm);

string colour1 = "AD7A99"; // pink
string colour2 = "7CDEDC"; // light blue
string colour3 = "006F63"; // green
string colour4 = "F57F17"; //orange
string colour5 = "0F1980"; //purple

string colour6 = "1565C0";
string colour7 = "C30934";
string colour8 = "8E24AA";



// ~~~~~~~~~~ First Lattice

real lattice_space = 4.8;

for (int i_d = 0; i_d<7; ++i_d)
{
    dot(lattice_space*(i_d, 0));
}

// arrows
real tunnelling_line_height = 1.2;


// shaking arrow
real arrow_height1 = 1.8;
real arrow_height2 = 0.6;
real arrow_height3 = 1;
real arrow_height4 = 2.4;

real shake_halfgap = 0.3;
real[] arrow1_xshifts = {0, 0,-shake_halfgap,-2*shake_halfgap, 0, 0};
real[] arrow2_xshifts = {0,0,shake_halfgap,0,-shake_halfgap,-2*shake_halfgap, 0, 0};
real[] arrow3_xshifts= {0,0,0,2*shake_halfgap,shake_halfgap,0,-shake_halfgap};
real[] arrow4_xshifts = {0,0,0,0,0,2*shake_halfgap, shake_halfgap};

for (int i_d=1; i_d<=3; ++i_d)
{
    draw((i_d*lattice_space+arrow1_xshifts[i_d],0) -- (i_d*lattice_space+arrow1_xshifts[i_d],arrow_height1), p=rgb(colour6)+linewidth(0.9pt), arrow=ArcArrow(SimpleHead, size=4));
    draw((i_d*lattice_space+arrow1_xshifts[i_d],0) -- (i_d*lattice_space+arrow1_xshifts[i_d],-arrow_height1), p=rgb(colour6)+linewidth(0.9pt)+linetype("2 2"), arrow=ArcArrow(SimpleHead, size=4));
}
for (int i_d=2; i_d<=5; ++i_d)
{
    draw((i_d*lattice_space+arrow2_xshifts[i_d],0) -- (i_d*lattice_space+arrow2_xshifts[i_d],arrow_height2), p=rgb(colour7)+linewidth(0.9pt), arrow=ArcArrow(SimpleHead, size=4));
    draw((i_d*lattice_space+arrow2_xshifts[i_d],0) -- (i_d*lattice_space+arrow2_xshifts[i_d],-arrow_height2), p=rgb(colour7)+linewidth(0.9pt)+linetype("2 2"), arrow=ArcArrow(SimpleHead, size=4));
}
for (int i_d=3; i_d<=6; ++i_d)
{
    draw((i_d*lattice_space+arrow3_xshifts[i_d],0) -- (i_d*lattice_space+arrow3_xshifts[i_d],arrow_height3), p=rgb(colour8)+linewidth(0.9pt), arrow=ArcArrow(SimpleHead, size=4));
    draw((i_d*lattice_space+arrow3_xshifts[i_d],0) -- (i_d*lattice_space+arrow3_xshifts[i_d],-arrow_height3), p=rgb(colour8)+linewidth(0.9pt)+linetype("2 2"), arrow=ArcArrow(SimpleHead, size=4));
}
for (int i_d=5; i_d<=6; ++i_d)
{
    draw((i_d*lattice_space+arrow4_xshifts[i_d],0) -- (i_d*lattice_space+arrow4_xshifts[i_d],arrow_height4), p=rgb(colour3)+linewidth(0.9pt), arrow=ArcArrow(SimpleHead, size=4));
    draw((i_d*lattice_space+arrow4_xshifts[i_d],0) -- (i_d*lattice_space+arrow4_xshifts[i_d],-arrow_height4), p=rgb(colour3)+linewidth(0.9pt)+linetype("2 2"), arrow=ArcArrow(SimpleHead, size=4));
}



// tunnelling curves
real y0_tunnelling_curve = 0.5;
real y_height_tunnelling_turve = 0.9;
for (int i_t = 0; i_t <6; ++i_t)
{
    draw(((i_t+0.06)*lattice_space, y0_tunnelling_curve) .. ((i_t+0.5)*lattice_space,y_height_tunnelling_turve+ y0_tunnelling_curve) .. ((i_t+1-0.06)*lattice_space, y0_tunnelling_curve));

}

// label J
real y_j_label =y0_tunnelling_curve+y_height_tunnelling_turve+0.7;
for (int i_t = 0; i_t <6; ++i_t)
{
    label("$J$", ((i_t +0.5)*lattice_space, y_j_label),  black);
}


// ~~~~~~ Second Lattice

real y_fig_shift = -7;
pair fig_shift = (0,y_fig_shift);

//dots

for (int i_d = 0; i_d<7; ++i_d)
{
    dot(lattice_space*(i_d, 0)+fig_shift);
}


//J labels
// label("$J'$", (2.5*lattice_space,y_j_label)+fig_shift,  p=rgb(colour4));
// for (int i_t = 0; i_t <2; ++i_t)
// {
//     label("$J$", ((i_t +0.5)*lattice_space, y_j_label)+fig_shift,  black);
// }
// for (int i_t = 3; i_t <5; ++i_t)
// {
//     label("$J$", ((i_t +0.5)*lattice_space, y_j_label)+fig_shift,  black);
// }



real y0_tunnelling_curve_b = y0_tunnelling_curve + y_fig_shift;
// draw((0.3, y0_tunnelling_curve_b) .. (2.5,y_height_tunnelling_turve+ y0_tunnelling_curve_b) .. (4.7, y0_tunnelling_curve_b));
// draw((5.3, y0_tunnelling_curve_b) .. (7.5,y_height_tunnelling_turve+ y0_tunnelling_curve_b) .. (9.7, y0_tunnelling_curve_b));
// draw((10.3, y0_tunnelling_curve_b) .. (12.5,y_height_tunnelling_turve+ y0_tunnelling_curve_b) .. (14.7, y0_tunnelling_curve_b), p=rgb(colour4)+linewidth(1pt));
// draw((15.3, y0_tunnelling_curve_b) .. (17.5,y_height_tunnelling_turve+ y0_tunnelling_curve_b) .. (19.7,y0_tunnelling_curve_b));
// draw((20.3, y0_tunnelling_curve_b) .. (22.5,y_height_tunnelling_turve+ y0_tunnelling_curve_b) .. (24.7,y0_tunnelling_curve_b));
string colours_tun[] = {colour6, colour7, colour8, colour6, colour3, colour7};
string label_tuns[] = {"$J'$","$J''$", "$J'''$", "$J'$", "$J''''$", "$J''$"};
for (int i_t = 0; i_t <=5; ++i_t)
{
    draw(((i_t+0.06)*lattice_space, y0_tunnelling_curve_b) .. ((i_t+0.5)*lattice_space,y_height_tunnelling_turve+ y0_tunnelling_curve_b) .. ((i_t+1-0.06)*lattice_space, y0_tunnelling_curve_b), p=rgb(colours_tun[i_t])+linewidth(1pt));
    label(label_tuns[i_t], ((i_t +0.5)*lattice_space, y_j_label)+fig_shift,  p=rgb(colours_tun[i_t]));
}
