settings.outformat = "pdf";
// settings.render=20;
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;

// size(7cm);


pair b_label_adjust = (0,0);

real row1_fig_y = 0;
real row2_fig_y = -16;
real col1_fig_x = 0;
real col2_fig_x = 15;
real col3_fig_x = 30;

real row1_label_y = 5;
real row2_label_y = -9.5;
real col1_label_x = 5.8;
real col2_label_x = 20.8;
real col3_label_x = 35.8;


real d_shift_x = -0.3;
real e_shift_x = 0.7;
real f_shift_x = e_shift_x;
// real col2__inset_x = 11.8;
// real b_inset_y = -4;



real inset_shift_x = -3.2;
real d_inset_x = col1_fig_x+inset_shift_x;
real e_inset_x = col2_fig_x+inset_shift_x;
real f_inset_x = col3_fig_x+inset_shift_x;
real row2_inset_y = row2_fig_y-2.7;

pair a_fig_loc = (col1_fig_x,row1_fig_y);
pair a_label_loc = (col1_label_x,row1_label_y);
pair b_fig_loc = (col2_fig_x,row1_fig_y);
pair b_label_loc = (col2_label_x,row1_label_y);
pair c_fig_loc = (col3_fig_x, row1_fig_y);
pair c_label_loc = (col3_label_x, row1_label_y);


pair d_fig_loc = (col1_fig_x+d_shift_x,row2_fig_y);
pair d_label_loc = (col1_label_x,row2_label_y);
pair e_fig_loc = (col2_fig_x+e_shift_x,row2_fig_y);
pair e_label_loc = (col2_label_x,row2_label_y);
pair f_fig_loc = (col3_fig_x+f_shift_x, row2_fig_y);
pair f_label_loc = (col3_label_x -2, row2_label_y);

pair d_inset_loc = (d_inset_x, row2_inset_y);
pair e_inset_loc = (e_inset_x, row2_inset_y);
pair f_inset_loc = (f_inset_x, row2_inset_y);

label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/lieb_lattice1.pdf"),a_fig_loc);
label("(a)", a_label_loc);

// label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/lieb_lattice_shake1.pdf"),b_fig_loc);
label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/lieb_lattice2.pdf"),b_fig_loc);
label("(b)", b_label_loc);

// label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/lieb_lattice_shake2.pdf"),c_fig_loc);
label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/lieb_lattice4.pdf"),c_fig_loc);
label("(c)", c_label_loc);


label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/lieb_lattice_shake1.pdf"),d_fig_loc);
label("(d)", d_label_loc);

label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/lieb_lattice_shake2.pdf"),e_fig_loc);
label("(e)", e_label_loc);

label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/lieb_lattice_shake4.pdf"),f_fig_loc);
label("(f)", f_label_loc);

label(graphic("/home/gnixon/floquet-simulations/asymptote/floquet_paper/lieb_renorm_1.pdf"),d_inset_loc);
label(graphic("/home/gnixon/floquet-simulations/asymptote/floquet_paper/lieb_renorm_2.pdf"),e_inset_loc);
label(graphic("/home/gnixon/floquet-simulations/asymptote/floquet_paper/lieb_renorm_3.pdf"),f_inset_loc);

