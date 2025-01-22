settings.outformat = "pdf";
// settings.render=20;
defaultpen(fontsize(9pt));
//defaultpen(arrowsize(9));
//defaultpen(arrowsize(5bp));
unitsize(3mm);
settings.tex="pdflatex" ;

pair a_loc = (0,0);
pair b_loc = (20,0);

label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/periodic_system_space.pdf"),a_loc);
label(graphic("/home/gnixon/floquet-simulations/asymptote/thesis/floquet/periodic_system_time.pdf"),b_loc);

