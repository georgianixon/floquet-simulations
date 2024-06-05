settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;


pair a_fig_loc = (0,0);
pair b_fig_loc = (13.5,0);
real c_fig_yshift = -1.7;
pair c_fig_loc = (27,c_fig_yshift);

pair label_shift = (0.5,-0.9);


label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/strob_ham_single_site_shake.pdf"),a_fig_loc, SE);
label("(a)", a_fig_loc + label_shift);


label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/strob_ham_single_site_shake_log.pdf"),b_fig_loc, SE);
label("(b)", b_fig_loc + label_shift);

label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/bessel_func_strob.pdf"),c_fig_loc, SE);
label("(c)", c_fig_loc + label_shift - (0,c_fig_yshift));
