settings.outformat = "pdf";
// settings.render=12;
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");


settings.tex="pdflatex" ;

size(7cm);



// figure labels
pair a_fig_loc = (-1,0);
pair b_fig_loc = (0,-12.1);

pair a_lab_loc = (0.5,-1);
pair b_lab_loc = (0.5,-13);

label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/flux_drive_params_findmin_nh=(0.5,0.3).pdf"),a_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/flux_J_params_findmin_nh=(0.5,0.3).pdf"), b_fig_loc, SE);

label("(a)", a_lab_loc, NW);
label("(b)", b_lab_loc, NW);


