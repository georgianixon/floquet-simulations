settings.outformat = "png";
settings.render=12;
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");


settings.tex="pdflatex" ;

size(7cm);



// figure labels
pair a_fig_loc = (1.2,1.5);
pair b_fig_loc = (-1.5,0);
pair c_fig_loc = (19,0.4);

pair a_lab_loc = (-1.5,39);
pair b_lab_loc = (-1.5,1.5);
pair c_lab_loc = (19,1.5);

label(graphic("/home/gnixon/floquet-simulations/figures/thesis/black_hole/BH_a_vals_alternating_2D.pdf"),b_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/thesis/black_hole/BH_linear_tunnelling_2D.pdf"), c_fig_loc, SE);
// label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/black_hole/2d_black_hole_circular.pdf"), a_fig_loc, NE);
// label("(a)", a_lab_loc, SE);
label("(a)", b_lab_loc, SE);
label("(b)", c_lab_loc, SE);


