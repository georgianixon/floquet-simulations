settings.outformat = "pdf";
// settings.render=20;
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;

// size(7cm);


pair b_label_adjust = (0,0);

real row1_fig_y = 0;
real row2_fig_y = -14;
real col1_fig_x = 0;
real col2_fig_x = 9.9;
real col3_fig_x = 26.5;

real row1_label_y = 0.5;
real row2_label_y = -13.3;
real col1_label_x = 0;
real col2_label_x = 12;
real col3_label_x =27;


real col2_shift_y = -0.6;
real col3_shift_y = col2_shift_y;
// real e_shift_x = 0;
// real f_shift_x = e_shift_x;
// real col2__inset_x = 11.8;
// real b_inset_y = -4;



real inset_shift_x = -3.2;
real d_inset_x = col1_fig_x+inset_shift_x;
real e_inset_x = col2_fig_x+inset_shift_x;
real f_inset_x = col3_fig_x+inset_shift_x;
real row2_inset_y = row2_fig_y-2.7;

pair a_fig_loc = (col1_fig_x,row1_fig_y);
pair a_label_loc = (col1_label_x,row1_label_y);
pair b_fig_loc = (col2_fig_x,row1_fig_y+col2_shift_y);
pair b_label_loc = (col2_label_x,row1_label_y);
pair c_fig_loc = (col3_fig_x, row1_fig_y+col3_shift_y);
pair c_label_loc = (col3_label_x, row1_label_y);


pair d_fig_loc = (col1_fig_x,row2_fig_y);
pair d_label_loc = (col1_label_x,row2_label_y);
pair e_fig_loc = (col2_fig_x,row2_fig_y+col2_shift_y);
pair e_label_loc = (col2_label_x,row2_label_y);
pair f_fig_loc = (col3_fig_x, row2_fig_y+col3_shift_y);
pair f_label_loc = (col3_label_x, row2_label_y);

// pair d_inset_loc = (d_inset_x, row2_inset_y);
// pair e_inset_loc = (e_inset_x, row2_inset_y);
// pair f_inset_loc = (f_inset_x, row2_inset_y);

string fillsize= "width=4cm,height=3.18cm";
pair col2_fill_shift = (2.58,-0.26);

pair col3_fill_shift = (0.64,-0.26);




label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/triangle_parameters_beta=1_cosine.pdf"),a_fig_loc, SE);

label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/triangle_parameters_beta=1_A2_unfilled.pdf"),b_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/triangle_parameters_beta=1_A3_unfilled.pdf"),c_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/A2.png", options=fillsize), b_fig_loc+col2_fill_shift, SE);
label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/A3.png", options=fillsize),c_fig_loc+col3_fill_shift, SE);
// 

pair col2_fill_shift = (2.51,-0.26);
label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/capcut2.png", options=fillsize), e_fig_loc+col2_fill_shift, SE);

label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/triangle_parameters_beta=2_cosine.pdf"),d_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/triangle_parameters_beta=2_A2_unfilled.pdf"),e_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/triangle_parameters_beta=2_A3_unfilled.pdf"),f_fig_loc, SE);
// label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/2alpha=beta_A2_powerlaw.png", options=fillsize), e_fig_loc+col2_fill_shift, SE);

label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/2alpha=beta_A3.png", options=fillsize),f_fig_loc+col3_fill_shift, SE);

label("(a)", a_label_loc,SE);
label("(b)", b_label_loc,SE);

label("(c)", c_label_loc,SE);

label("(d)", d_label_loc,SE);

label("(e)", e_label_loc,SE);

label("(f)", f_label_loc,SE);


