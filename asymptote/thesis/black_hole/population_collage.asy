settings.outformat = "pdf";
defaultpen(fontsize(9pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;


pair a_fig_loc = (0,0);
pair b_fig_loc = (14.5,0);
pair c_fig_loc = (28,0);

pair label_shift = (0,1.4);


label(graphic("/home/gnixon/floquet-simulations/figures/thesis/black_hole/probabilities.pdf"),a_fig_loc-(0.2,0.2), SE);
label("(a)", a_fig_loc + label_shift, SE);


label(graphic("/home/gnixon/floquet-simulations/figures/thesis/black_hole/walks.pdf"),b_fig_loc, SE);
label("(b)", b_fig_loc + label_shift, SE);

label(graphic("/home/gnixon/floquet-simulations/figures/thesis/black_hole/measured_tfs.pdf"),c_fig_loc, SE);
label("(c)", c_fig_loc + label_shift, SE);
