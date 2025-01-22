settings.outformat = "pdf";
defaultpen(fontsize(9pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;


// pair a_fig_loc = (0,0);
// pair b_fig_loc = (13.5,0);
// real c_fig_yshift = -1.7;
// pair c_fig_loc = (27,c_fig_yshift);

// pair label_shift = (0.5,-0.9);


// label(graphic("/home/gnixon/floquet-simulations/figures/thesis/black_hole/HSt0_linear_gradient.pdf"),a_fig_loc, SE);
// label("(a)", a_fig_loc + label_shift);


// label(graphic("/home/gnixon/floquet-simulations/figures/thesis/black_hole/HSt0_linear_gradient_lognorm.pdf"),b_fig_loc, SE);
// label("(b)", b_fig_loc + label_shift);

// label(graphic("/home/gnixon/floquet-simulations/figures/thesis/black_hole/HSt0_linear_gradient_tunnelling_terms.pdf"),c_fig_loc, SE);
// label("(c)", c_fig_loc + label_shift - (0,c_fig_yshift));



pair a_fig_loc = (0,0);
pair b_fig_loc = (19,0);


pair label_shift = (0.7,0);

label(graphic("/home/gnixon/floquet-simulations/figures/thesis/black_hole/bh_errors_tunnelling.pdf"),a_fig_loc + (0,-0.2), SE);
label("(a)", a_fig_loc + label_shift);


label(graphic("/home/gnixon/floquet-simulations/figures/thesis/black_hole/bh_errors_interactions.pdf"),b_fig_loc, SE);
label("(b)", b_fig_loc + label_shift);



