settings.outformat = "pdf";
defaultpen(fontsize(9pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;

// size(7cm);
pair a_fig_loc = (0,0);
pair b_fig_loc = (18,0);
pair c_fig_loc = (0,-18);
pair d_fig_loc = (18,-18);

pair label_shift = (0, 1.3);
// label(graphic("/home/gnixon/asymptote/thesis/local_flux_ladder_cartoon_equalflux_drives.pdf"),a_fig_loc, SE);
// label(graphic("/home/gnixon/asymptote/thesis/local_flux_ladder_cartoon_equalflux_fluxes.pdf"),b_fig_loc, SE);

label(graphic("/home/gnixon/floquet-simulations/figures/thesis/optical_lattices/free_particle.pdf"),a_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/thesis/optical_lattices/extended_free_particle.pdf"),b_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/thesis/optical_lattices/lattice_bands.pdf"),c_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/thesis/optical_lattices/gapped_lattice_bands.pdf"),d_fig_loc, SE);
label("(a)", a_fig_loc + label_shift, SE);
label("(b)", b_fig_loc + label_shift, SE);
label("(c)", c_fig_loc + label_shift, SE);
label("(d)", d_fig_loc + label_shift, SE);


