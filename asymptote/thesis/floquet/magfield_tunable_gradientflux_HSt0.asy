settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;

// size(7cm);
pair a_fig_loc = (0,0);
pair b_fig_loc = (21,0);
pair c_fig_loc = (0,-15);
pair c_fig_shift = (-0.1, 1);
pair label_shift = (1, 0.7);
pair b_shift = (0,0);
// label(graphic("/home/gnixon/asymptote/thesis/local_flux_ladder_cartoon_equalflux_drives.pdf"),a_fig_loc, SE);
// label(graphic("/home/gnixon/asymptote/thesis/local_flux_ladder_cartoon_equalflux_fluxes.pdf"),b_fig_loc, SE);

label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/grad_mag_field_linearflux_6x30.pdf"),a_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/grad_mag_field_abs_vals_6x30.pdf"),b_fig_loc, SE);
label(graphic("/home/gnixon/floquet-simulations/figures/thesis/floquet/grad_mag_field_colourbarflux_6x30.pdf"),c_fig_loc+c_fig_shift, SE);

label("(a)", a_fig_loc + label_shift);
label("(b)", b_fig_loc + label_shift);
label("(c)", c_fig_loc + label_shift);

