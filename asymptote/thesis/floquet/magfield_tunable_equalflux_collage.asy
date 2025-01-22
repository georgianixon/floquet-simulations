settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;

// size(7cm);
pair a_fig_loc = (0,0);
pair b_fig_loc = (21,0);
pair label_shift = (1, 0.3);
pair b_shift = (1.4,-1.3);
// label(graphic("/home/gnixon/asymptote/thesis/local_flux_ladder_cartoon_equalflux_drives.pdf"),a_fig_loc, SE);
// label(graphic("/home/gnixon/asymptote/thesis/local_flux_ladder_cartoon_equalflux_fluxes.pdf"),b_fig_loc, SE);

label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/magfield_tunable_equalflux_drives.pdf"),a_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/magfield_tunable_equalflux_fluxes.pdf"),b_fig_loc+b_shift, SE);
label("(a)", a_fig_loc + label_shift);
label("(b)", b_fig_loc + label_shift);

