settings.outformat = "pdf";
// settings.render=12;
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");


settings.tex="pdflatex" ;

size(7cm);




// figure labels
pair a_fig_loc = (0,0);
pair b_fig_loc = (14.4,0);
pair c_fig_loc = (0, -15);

pair a_lab_loc = (1.4,-2);
pair b_lab_loc = (15,-2);
pair c_lab_loc = (1.4, -16);




label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/sss_v2.pdf"),a_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/sss_log_v2.pdf"), b_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/bessel_func_strob_v2.pdf"), c_fig_loc, SE);

label("(a)", a_lab_loc, NW);
label("(b)", b_lab_loc, NW);
label("(c)", c_lab_loc, NW);


