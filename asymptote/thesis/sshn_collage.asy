settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;

// size(7cm);


real b_fig_yshift = -0;
pair a_fig_loc = (0,0);
pair b_fig_loc = (18,b_fig_yshift);


pair label_shift = (-0.3,-0.9);


label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/strob_ham_ssh3.pdf"),a_fig_loc, SE);
label("(a)", a_fig_loc + label_shift);


label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/strob_ham_ssh4.pdf"),b_fig_loc, SE);
label("(b)", b_fig_loc + label_shift - (0,b_fig_yshift));

