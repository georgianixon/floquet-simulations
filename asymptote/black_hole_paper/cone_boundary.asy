settings.outformat = "pdf";
// settings.render=10;
defaultpen(fontsize(9pt));
unitsize(3mm);
settings.tex="pdflatex" ;

pen dotsize=linewidth(3.5pt);
pen shaded_region = gray(0.9);

int n_lat = 8;
real lat_space = 3;
real shaded_region_gap = 0.3;
real tunnelling_gap = 0.2;
real arrow_head_size = 4;
pen tunnelling_colour = rgb("#2E788F");
pen fnt_size_site_labs = fontsize(8pt);
pen mirror_line = rgb("#FF40F9")+linewidth(0.7pt)+linetype("2 3");




// fill 4x4 section
fill((4-shaded_region_gap,4-shaded_region_gap)*lat_space -- (4-shaded_region_gap,7+shaded_region_gap)*lat_space -- (7+shaded_region_gap,7+shaded_region_gap)*lat_space -- (7+shaded_region_gap,4-shaded_region_gap)*lat_space -- cycle, p=shaded_region);


// reflection symmetries
real symmetry_gap = 0.3;
draw((0-symmetry_gap,3.5)*lat_space -- (7+symmetry_gap,3.5)*lat_space, p=mirror_line);
draw((3.5,-symmetry_gap)*lat_space -- (3.5,7+ symmetry_gap)*lat_space, p=mirror_line);
draw((0-symmetry_gap,-symmetry_gap)*lat_space -- (7+symmetry_gap,7+ symmetry_gap)*lat_space, p=mirror_line);
draw((-symmetry_gap,7+symmetry_gap)*lat_space -- (7+symmetry_gap,- symmetry_gap)*lat_space, p=mirror_line);

// dots
for (int i = 0; i < n_lat; ++i)
{
    for (int j = 0; j < n_lat; ++j)
    {
        if (i>3 & j > 3)
        {
            dot((i,j)*lat_space, p=dotsize+rgb("#ffa600"));
        }
        else
        {
            dot((i,j)*lat_space, p=dotsize);
        }
    }
}

//tunnelling labels
// string horizontal_labs[][] = {{"J8", "J9", "J10"}, {"J6", "J8", "J9"}, {"J4", "J6", "J8"}, {"J2", "J4", "J6"}};
// for (int i = 0; i < 3; ++i)
// {
//     for (int j = 0; j < 4; ++j)
//     {
//         label(horizontal_labs[j][i], (i+4,7-j)*lat_space);
//     }
// }

// horiz tunnellings
for (int i = 4; i < 7; ++i)
{
    for (int j = 7; j > 3; --j)
    {
        draw((i+tunnelling_gap,j)*lat_space -- (i+1-tunnelling_gap,j)*lat_space);
    }
}

// vert tunnellings
for (int i = 4; i < 8; ++i)
{
    for (int j = 7; j > 4; --j)
    {
        draw((i,j-tunnelling_gap)*lat_space -- (i,j - 1 + tunnelling_gap)*lat_space);
    }
}

// // tunnellings
// draw((5,4-tunnelling_gap)*lat_space{down}..(3.5,3.5)*lat_space..{right}(4-tunnelling_gap,5)*lat_space,arrow=ArcArrows(SimpleHead, size=arrow_head_size), p=tunnelling_colour);
// draw((6,4-tunnelling_gap)*lat_space{down}..(3.35,3.35)*lat_space..{right}(4-tunnelling_gap,6)*lat_space,arrow=ArcArrows(SimpleHead, size=arrow_head_size), p=tunnelling_colour);
// draw((7,4-tunnelling_gap)*lat_space{down}..(3.2,3.2)*lat_space..{right}(4-tunnelling_gap,7)*lat_space,arrow=ArcArrows(SimpleHead, size=arrow_head_size), p=tunnelling_colour);
draw((5,4-tunnelling_gap)*lat_space{down}..(3.5,3.5)*lat_space..{right}(4-tunnelling_gap,5)*lat_space, p=tunnelling_colour);
draw((6,4-tunnelling_gap)*lat_space{down}..(3.35,3.35)*lat_space..{right}(4-tunnelling_gap,6)*lat_space, p=tunnelling_colour);
draw((7,4-tunnelling_gap)*lat_space{down}..(3.2,3.2)*lat_space..{right}(4-tunnelling_gap,7)*lat_space, p=tunnelling_colour);


// enumerate labels
int current_site = 1;
pair site_label_gap = (0.2,0.2)*lat_space;
for (int j = 7; j >= 4; --j)
{
    for (int i = 4; i <= 7; ++i)
    {
        label((string) current_site, lat_space*(i, j) + site_label_gap, p=fnt_size_site_labs);
        ++current_site;
    }
}



