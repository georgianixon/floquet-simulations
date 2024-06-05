settings.outformat = "pdf";
// settings.render=12;
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");


settings.tex="pdflatex" ;

size(7cm);



// figure labels
pair a_fig_loc = (0,0+0.6);
pair b_fig_loc = (14.1,0);

pair a_lab_loc = (1.4,-1.3);
pair b_lab_loc = (15.4,-1.3);

label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/ssh_model_v2.pdf"),a_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/local_mod_paper/zero_energy_edge_state_v2.pdf"), b_fig_loc, SE);

label("(a)", a_lab_loc, NW);
label("(b)", b_lab_loc, NW);


