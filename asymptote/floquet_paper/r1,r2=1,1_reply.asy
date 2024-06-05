settings.outformat = "png";
settings.render=10;
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;

// size(7cm);

// real a_b_label_y = 0;
// real c_d_label_y = -8.6;
// real a_c_label_x = 0;
// real b_d_label_x = 20.8;

// pair b_label_adjust = (-2,0);

real a_b_fig_y = 0;
real c_d_fig_y = -16;
real a_c_fig_x = 0;
real b_d_fig_x = 28;




pair a_fig_loc = (a_c_fig_x,a_b_fig_y);
pair b_fig_loc = (b_d_fig_x,a_b_fig_y);
pair c_fig_loc = (a_c_fig_x,c_d_fig_y);
pair d_fig_loc = (b_d_fig_x,c_d_fig_y);
pair e_fig_loc = (12,-32);

pair label_offset = (4,0);

label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/find_minimum_Aparams_r1,r2=1,1.png"),a_fig_loc, SE);
label("(a)", a_fig_loc, SE);

label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/find_minimum_omegavals_r1,r2=1,1.png"),b_fig_loc, SE);
label("(b)", b_fig_loc, SE);

label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/find_minimum_fluxvals_r1,r2=1,1.png"),c_fig_loc, SE);
label("(c)", c_fig_loc, SE);

label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/find_minimum_rvals_r1,r2=1,1.png"),d_fig_loc, SE);
label("(d)", d_fig_loc, SE);

label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/find_minimum_Jparams_r1,r2=1,1.png"),e_fig_loc, SE);
label("(e)", e_fig_loc - (3,0), SE);