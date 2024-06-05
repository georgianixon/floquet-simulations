settings.outformat = "png";
settings.render=20;
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");
settings.tex="pdflatex" ;
string pale_green = "#D8F2EB";
string pale_yellow = "#F2EAD8";
string pale_blue2 = "#D8E0F2";
string pale_blue = "BBDDFA"; 

string[] inner_labs = {"C", "B", "A", "C", "A", "B"};
string[] outer_labs = {"A", "C", "B", "B", "C", "A"};
string[] tri_phases = {"$-\phi$", "$-\phi$", "$-\phi$", "$\phi$", "$\phi$", "$\phi$"};
pen[] tri_phase_pens = {rgb(pale_yellow), rgb(pale_yellow), rgb(pale_yellow), rgb(pale_blue), rgb(pale_blue), rgb(pale_blue)};
pen centre_phase_pen = white;
string cent_phase = "$0$";

real lattice_space = 4;
real kag_angle = pi/3;
pair label_shift = lattice_space*(0.2,0);
pen lw = linewidth(0.8pt);
pen line_col = grey;


fill(lattice_space*(cos(0), sin(0)) -- lattice_space*(cos(pi/3), sin(pi/3)) -- lattice_space*(cos(2*pi/3), sin(2*pi/3)) -- lattice_space*(cos(pi), sin(pi)) --lattice_space*(cos(4*pi/3), sin(4*pi/3)) --lattice_space*(cos(5*pi/3), sin(5*pi/3)) -- cycle, p=centre_phase_pen);
label((0,0), cent_phase);
for (int rot_int = 0; rot_int < 6; ++ rot_int)
{
    real angle = rot_int*kag_angle;
    pair inner_loc = lattice_space*(cos(angle), sin(angle));
    pair outer_loc = 2*lattice_space*sin(pi/3)*(cos(angle+pi/6), sin(angle+pi/6));
    pair next_inner_loc = lattice_space*(cos((rot_int+1)*kag_angle), sin((rot_int+1)*kag_angle));

    fill(inner_loc -- outer_loc -- next_inner_loc -- cycle, p=tri_phase_pens[rot_int]);
    draw(inner_loc -- outer_loc, p=line_col+lw);
    draw(inner_loc --  next_inner_loc, p=line_col+lw);
    draw(outer_loc --  next_inner_loc, p=line_col+lw);
    dot(inner_loc);
    dot(outer_loc);
    label(inner_labs[rot_int], inner_loc +label_shift);
    label(outer_labs[rot_int], outer_loc+ label_shift);
    label(tri_phases[rot_int], (inner_loc+next_inner_loc+outer_loc)/3);
}



