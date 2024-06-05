settings.outformat = "png";
settings.render=20;
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;

// size(7cm);

real a_b_label_y = 5.4;
real c_d_label_y = -8.6;
real a_c_label_x = 5.8;
real b_d_label_x = 20.8;

pair b_label_adjust = (-2,0);

real a_b_fig_y = 0;
real c_d_fig_y = -15;
real a_c_fig_x = 0;
real b_d_fig_x = 15.6;

pair a_adjust = (0,0.4);
pair b_adjust = (-0.7, 0);

real b_d_inset_x = 11.8;
real b_inset_y = -4;
real c_inset_x = -3.2;
real c_d_inset_y = -18;

pair a_fig_loc = (a_c_fig_x,a_b_fig_y) + a_adjust;
pair a_label_loc = (a_c_label_x,a_b_label_y);
pair b_fig_loc = (b_d_fig_x,a_b_fig_y) + b_adjust;
pair b_label_loc = (b_d_label_x,a_b_label_y) + b_label_adjust;
pair c_fig_loc = (a_c_fig_x,c_d_fig_y);
pair c_label_loc = (a_c_label_x,c_d_label_y);
pair d_fig_loc = (b_d_fig_x,c_d_fig_y);
pair d_label_loc = (b_d_label_x,c_d_label_y);
pair b_inset_loc = (b_d_inset_x, b_inset_y);
pair c_inset_loc = (c_inset_x, c_d_inset_y);
pair d_inset_loc = (b_d_inset_x, c_d_inset_y);



label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/lieb_lattice_shake1.pdf"),b_fig_loc);
label("(b)", b_label_loc);

label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/lieb_lattice_shake2.pdf"),c_fig_loc);
label("(c)", c_label_loc);

label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/lieb_lattice_shake3.pdf"),d_fig_loc);
label("(d)", d_label_loc);

label(graphic("/home/gnixon/floquet-simulations/asymptote/floquet_paper/lieb_lattice.pdf"),a_fig_loc);
label("(a)", a_label_loc);


label(graphic("/home/gnixon/floquet-simulations/asymptote/floquet_paper/lieb_renorm_1.pdf"),b_inset_loc);
label(graphic("/home/gnixon/floquet-simulations/asymptote/floquet_paper/lieb_renorm_2.pdf"),c_inset_loc);
label(graphic("/home/gnixon/floquet-simulations/asymptote/floquet_paper/lieb_renorm_3.pdf"),d_inset_loc);

