settings.outformat = "pdf";
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");

size(7cm);

string colour1 = "AD7A99"; // pink
string colour2 = "7CDEDC"; // light blue
string colour3 = "006F63"; // green
string colour4 = "F57F17"; //orange
string colour5 = "0F1980"; //purple

real y0_tunnelling_curve = 0.5;
real y_height_tunnelling_turve = 0.9;
real x_tunnelling_gap = 0.3;
real x_site_space = 5;
int N_sites = 3;
real x_label_loc_a = -1.2;
real x_label_loc_b = N_sites*x_site_space - x_label_loc_a;
real y_label_loc = 2.5; 
label("(a)", (x_label_loc_a, y_label_loc));
for (int i = 0; i <= N_sites; ++i)
{
    dot(( x_site_space*i,0));
}


for (int i = 0; i <= N_sites-1; ++i)
{
    draw((i*x_site_space+ x_tunnelling_gap, y0_tunnelling_curve) .. ((i+0.5)*x_site_space,y_height_tunnelling_turve+ y0_tunnelling_curve) .. ((i+1)*x_site_space - x_tunnelling_gap, y0_tunnelling_curve), arrow=ArcArrow(SimpleHead, size=4));
    draw(((i+1)*x_site_space - x_tunnelling_gap, -y0_tunnelling_curve)  .. ((i+0.5)*x_site_space,-y_height_tunnelling_turve- y0_tunnelling_curve) .. (i*x_site_space+ x_tunnelling_gap,-y0_tunnelling_curve), arrow=ArcArrow(SimpleHead, size=4));
}

// label
real y_j_label_add = 0.01;
real y_j_label =y0_tunnelling_curve+y_height_tunnelling_turve+y_j_label_add;
label("$J e^{i \Phi}$", (1.5*x_site_space, y_j_label),  N);
label("$J e^{-i \Phi}$", (1.5*x_site_space, -y_j_label),  S);

real x_fig_b_shift = N_sites*x_site_space - 7*x_label_loc_a;
label("(b)", (x_fig_b_shift + 2*x_label_loc_a, y_label_loc));
for (int i_x = 0; i_x < 2; ++i_x )
{
    for (int i_y = 0; i_y < 2; ++ i_y)
    {
        dot((x_fig_b_shift + i_x*x_site_space,-i_y*x_site_space));
    }
}
string complex_tunnelling = "$J e^{i \frac{\Phi}{4}}$";
pair point1 = (x_fig_b_shift + x_tunnelling_gap, y0_tunnelling_curve);
pair point2 = (x_fig_b_shift+0.5*x_site_space,y_height_tunnelling_turve+ y0_tunnelling_curve);
pair point3 = (x_fig_b_shift+x_site_space - x_tunnelling_gap, y0_tunnelling_curve);
draw(point1 .. point2 .. point3, arrow=ArcArrow(SimpleHead, size=4));
label(complex_tunnelling, point2+(0,y_j_label_add),   N);
pair point1 = (x_fig_b_shift + x_site_space + y0_tunnelling_curve , -x_tunnelling_gap);
pair point2 = (x_fig_b_shift + x_site_space + y0_tunnelling_curve + y_height_tunnelling_turve, -x_site_space/2);
pair point3 =  (x_fig_b_shift + x_site_space + y0_tunnelling_curve, -x_site_space+ x_tunnelling_gap);
draw(point1 .. point2 .. point3, arrow=ArcArrow(SimpleHead, size=4));
label(complex_tunnelling, point2 + (y_j_label_add,0),  E);
pair point1 = (x_fig_b_shift + x_site_space -x_tunnelling_gap, -x_site_space - y0_tunnelling_curve);
pair point2 =  (x_fig_b_shift+0.5*x_site_space ,- x_site_space - y_height_tunnelling_turve - y0_tunnelling_curve);
pair point3 = (x_fig_b_shift + x_tunnelling_gap, -x_site_space - y0_tunnelling_curve); 
draw( point1 .. point2 .. point3, arrow=ArcArrow(SimpleHead, size=4));
label(complex_tunnelling, point2 + (0,-y_j_label_add), S);
pair point1 =  (x_fig_b_shift - y0_tunnelling_curve, - x_site_space + x_tunnelling_gap);
pair point2 = (x_fig_b_shift  - y0_tunnelling_curve - y_height_tunnelling_turve, -x_site_space/2);
pair point3 = (x_fig_b_shift - y0_tunnelling_curve , -x_tunnelling_gap);
draw( point1 .. point2 .. point3, arrow=ArcArrow(SimpleHead, size=4));
label(complex_tunnelling, point2 + (-y_j_label_add,0),  W);

label("$\Phi$", (x_fig_b_shift+x_site_space/2, -x_site_space/2));




