settings.outformat = "pdf";
// settings.render=20;
defaultpen(fontsize(10pt));
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

string[] colours = {colour3, colour2, colour1, colour4, colour5};

// ################## FIRST ONE
real dot_separation_x = 4;
real centre_dot_x = 16;

// grey tweezer goes first to be behind
real optical_tweez_height = 3.2;
real optical_tweez_width_min = 0.5;
real optical_tweez_width_max = 1.7;
fill((centre_dot_x - optical_tweez_width_min,0){up} .. (centre_dot_x - optical_tweez_width_max,optical_tweez_height) -- (centre_dot_x + optical_tweez_width_max,optical_tweez_height) .. (centre_dot_x+optical_tweez_width_min,0){down} .. (centre_dot_x+optical_tweez_width_max,-optical_tweez_height) -- (centre_dot_x - optical_tweez_width_max, -optical_tweez_height) .. cycle, p=rgb("BBDDFA"));

//shakes
real first_shake_height = 2.4;
real shake_height_increase = 1.2;
real arrow_head_size_decrease = 1;
//dots
// dot((0,0));
//dot((centre_dot_x - 7*dot_separation_x,0));
int num_dots_onside = 2;
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

    draw((first_dot_x - i_d*dot_separation_x,0) -- (first_dot_x - i_d*dot_separation_x,-first_shake_height), p=rgb(colours[i_d])+linewidth(0.7pt)+linetype("2 2"), arrow=ArcArrow(SimpleHead, size=3.5*arrow_head_size_decrease^i_d));
    draw((first_dot_x - i_d*dot_separation_x,0) -- (first_dot_x - i_d*dot_separation_x,+first_shake_height), p=rgb(colours[i_d])+linewidth(0.7pt), arrow=ArcArrow(SimpleHead, size=3.5*arrow_head_size_decrease^i_d));

}

// Al labels being alternating
// epsilon bar


// e_i 
draw((centre_dot_x+0.45,0) -- (centre_dot_x+0.45,-first_shake_height),p=linewidth(0.8pt),bar=Bars(size=3));
label("$W_l(t)$", (centre_dot_x+2,-1.55));

//J_0 label
label("$J$", (centre_dot_x - dot_separation_x*2.5, 0.6));


// ################## second time-dependent pic
real second_row_label_height = -4.1;
real second_row_image_heightb = second_row_label_height-6.2;
real second_row_image_heightc = second_row_label_height-6.65;
real first_column_label_x = 3;
real second_column_label_x = 16.9;
real second_column_fig_x = 23.5;


pair b_label_loc = (3,-4.6);
pair b_image_loc = (9.2,-11.7);
pair c_image_loc = (23.2,-10.9);
pair c_inset_image_loc = (20.9, -13.9);
pair c_label_loc = (16.9,-4.6);
