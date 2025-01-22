settings.outformat = "png";
settings.render=10;
defaultpen(fontsize(9pt));
//defaultpen(arrowsize(9));
//defaultpen(arrowsize(5bp));
unitsize(3mm);
settings.tex="pdflatex" ;


import graph;
//size(7cm);

//-0.37242316  0.85520254 -0.51623059  1.        
string colour1 = "1565C0";
string colour2 = "C30934";
string colour3 = "006F63";
string colour4 = "F57F17";
string colour5 = "8E24AA";

// ################## FIRST ONE
real dot_separation_x = 4.5;
real centre_dot_x = 16;

// grey tweezer goes first to be behind
real optical_tweez_height = 3.4;
real optical_tweez_width_min = 0.5;
real optical_tweez_width_max = 1.7;
fill((centre_dot_x - optical_tweez_width_min,0){up} .. (centre_dot_x - optical_tweez_width_max,optical_tweez_height) -- (centre_dot_x + optical_tweez_width_max,optical_tweez_height) .. (centre_dot_x+optical_tweez_width_min,0){down} .. (centre_dot_x+optical_tweez_width_max,-optical_tweez_height) -- (centre_dot_x - optical_tweez_width_max, -optical_tweez_height) .. cycle, p=rgb("BBDDFA"));


//shakes
real first_shake_height = 1.1;
real shake_height_increase = 1.2;
real arrow_head_size_decrease = 1;
//dots
// dot((0,0));
//dot((centre_dot_x - 7*dot_separation_x,0));
int num_dots_onside = 3;
dot((centre_dot_x,0));

for (int i_d=1; i_d<=num_dots_onside; ++i_d)
{
 dot((centre_dot_x - i_d*dot_separation_x,0));
 dot((centre_dot_x + i_d*dot_separation_x,0));
}
draw((centre_dot_x-(num_dots_onside+1)*dot_separation_x, 0)--(centre_dot_x+(num_dots_onside+1)*dot_separation_x,0));

// arrow shakes
real first_dot_x = centre_dot_x + num_dots_onside*dot_separation_x;
for (int i_d=0; i_d<=num_dots_onside*2; ++i_d)
{
    if(i_d %2== 1) {
        draw((first_dot_x - i_d*dot_separation_x,0) -- (first_dot_x - i_d*dot_separation_x,-first_shake_height*shake_height_increase^i_d), p=rgb(colour1)+linewidth(0.7pt)+linetype("2 2"), arrow=ArcArrow(SimpleHead, size=3.5*arrow_head_size_decrease^i_d));
        draw((first_dot_x - i_d*dot_separation_x,0) -- (first_dot_x - i_d*dot_separation_x,+first_shake_height*shake_height_increase^i_d), p=rgb(colour1)+linewidth(0.7pt), arrow=ArcArrow(SimpleHead, size=3.5*arrow_head_size_decrease^i_d));

    } else {
        draw((first_dot_x - i_d*dot_separation_x,0) -- (first_dot_x - i_d*dot_separation_x,-first_shake_height*shake_height_increase^i_d), p=rgb(colour1)+linewidth(0.7pt), arrow=ArcArrow(SimpleHead, size=3.5*arrow_head_size_decrease^i_d));
        draw((first_dot_x - i_d*dot_separation_x,0) -- (first_dot_x - i_d*dot_separation_x,+first_shake_height*shake_height_increase^i_d), p=rgb(colour1)+linewidth(0.7pt)+linetype("2 2"), arrow=ArcArrow(SimpleHead, size=3.5*arrow_head_size_decrease^i_d));
    }
}

// Al labels being alternating
// epsilon bar


// e_i 
draw((centre_dot_x+0.45,0) -- (centre_dot_x+0.45,-first_shake_height*shake_height_increase^3),p=linewidth(0.8pt),bar=Bars(size=3));
label("$W_j(t)$", (centre_dot_x+1.9,-1.55));

//J_0 label
label("$\kappa J_0$", (centre_dot_x - dot_separation_x*2.5, 0.8));

real A_vals_height = first_shake_height*shake_height_increase^4+1;

label("$A_{j-3}$", (centre_dot_x - 3*dot_separation_x-1.5, -first_shake_height*shake_height_increase^4-0.5));
label("$A_{j-2}$", (centre_dot_x - 2*dot_separation_x, first_shake_height*shake_height_increase^5+0.5));
label("$A_{j-1}$", (centre_dot_x - dot_separation_x, -first_shake_height*shake_height_increase^4-0.5));
label("$A_{j}$", (centre_dot_x, first_shake_height*shake_height_increase^3+0.5));
label("$A_{j+1}$", (centre_dot_x+ dot_separation_x+1.2, -first_shake_height*shake_height_increase^2-0.5));
label("$A_{j+2}$", (centre_dot_x+2*dot_separation_x+0.3, first_shake_height*shake_height_increase+0.6));
label("$A_{j+3}$", (centre_dot_x+3*dot_separation_x+1, -first_shake_height-0.5));

// ################## second time-dependent pic

real y_row1_lab = 5.5;
real y_row2_lab = -4;
real y_row2_img = -5.5;

real x_col1_lab = -4.5;

pair b_image_loc = (-4.5,y_row2_img);
pair c_image_loc = (16.5,y_row2_img);

pair a_label_loc = (x_col1_lab,5);
pair b_label_loc = (x_col1_lab,y_row2_lab);
pair c_label_loc = (16.5,y_row2_lab);
