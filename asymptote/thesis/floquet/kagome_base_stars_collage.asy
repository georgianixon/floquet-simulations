settings.outformat = "pdf";
// settings.render=12;
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

real x_figshift = 16;
real y_figshift = -16;

pair a_fig_loc = (0,0);
pair b_fig_loc = (x_figshift,0);
pair c_fig_loc = (0,y_figshift);
pair d_fig_loc = (x_figshift,y_figshift);

pair label_shift = (0,0);

label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/kagome_star0.pdf"),a_fig_loc, SE);
label("(a)", a_fig_loc + label_shift, SE);


label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/kagome_star1.pdf"),b_fig_loc , SE);
label("(b)", b_fig_loc + label_shift, SE);

label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/kagome_star2.pdf"),c_fig_loc, SE);
label("(c)", c_fig_loc + label_shift, SE);


label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/kagome_star3.pdf"),d_fig_loc , SE);
label("(d)", d_fig_loc + label_shift, SE);



