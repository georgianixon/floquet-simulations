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
string colour6 = "C30934"; //red

pen fnt_size_A = fontsize(10pt);
pen fnt_size_dots = fontsize(8pt);
pen quantum_dot_colour = rgb("000000");
pen coupler_colour = rgb(colour4);

int lat_L = 5;
real lat_space = 3.2;
pair lat_label_gap = 0.7*(-1,+0.7);
int i_dot = 1;

real arrow_size = 1.6;
pen lw_arrows = linewidth(1.9pt);
pen lw_tunnellings = linewidth(0.1pt);
real arrow_head_size = 6;



// draw horizontal tunnellings
real tunnelling_line_gap = 0.4;
for (int i_x = 0; i_x < lat_L - 1; ++i_x)
{
    for (int i_y=0; i_y>-lat_L; i_y=i_y - 2)
    {
        pair spot = lat_space*(i_x, i_y);
        draw(spot+(tunnelling_line_gap,0)   -- spot+lat_space*(1,0)-(tunnelling_line_gap,0), p=rgb("000000")+lw_tunnellings);
    }

}

// draw vertical ltunnellings
for (int i_x = 0; i_x < lat_L ; i_x = i_x + 2)
{
     for (int i_y=0; i_y>-lat_L+1; --i_y)
    {
        pair spot = lat_space*(i_x, i_y);
        draw(spot-(0,tunnelling_line_gap)   -- spot-lat_space*(0,1)+(0,tunnelling_line_gap), p=rgb("000000")+lw_tunnellings);
    
    }

}

// draw arrows
pair spot = lat_space*(2, -1);
draw(spot - arrow_size*(0,-1)  -- spot+arrow_size*(0,-1), p=rgb(colour6)+lw_arrows, arrow=ArcArrows(SimpleHead, size=arrow_head_size));

// pair spot = lat_space*(2, -3);
// draw(spot - arrow_size*(0,-1)  -- spot+arrow_size*(0,-1), p=rgb(colour3)+lw_arrows, arrow=ArcArrows(SimpleHead, size=arrow_head_size));

pair spot = lat_space*(1, -2);
draw(spot - arrow_size*(0,-1)  -- spot+arrow_size*(0,-1), p=rgb(colour5)+lw_arrows, arrow=ArcArrows(SimpleHead, size=arrow_head_size));

// pair spot = lat_space*(3, -2);
// draw(spot - arrow_size*(0,-1)  -- spot+arrow_size*(0,-1), p=rgb(colour1)+lw_arrows, arrow=ArcArrows(SimpleHead, size=arrow_head_size));


// draw dots
for (int i_y=0; i_y>-lat_L; --i_y)
{ 
    for (int i_x = 0; i_x < lat_L; ++i_x)
    {
        if(i_y %2== 0) {
        // do all dots
            if(i_x %2 == 0){
                // quantum dot colour
                dot(lat_space*(i_x, i_y), p=quantum_dot_colour);
            } else {
                // coupler
                dot(lat_space*(i_x, i_y), p=coupler_colour);
            }
            // do label
            // if ((i_dot >=90 && i_dot <= 12) || (i_dot >=170 && i_dot <= 20) || (i_dot >=10 && i_dot <= 4) ) {
            //     label((string) i_dot, lat_space*(i_x, i_y) + lat_label_gap - (0,0.5), p=fnt_size_dots);

            // } else {
            //     label((string) i_dot,lat_space*(i_x, i_y) + lat_label_gap, p=fnt_size_dots);
            // }
            // }
            label((string) i_dot,lat_space*(i_x, i_y) + lat_label_gap, p=fnt_size_dots);
            i_dot = i_dot + 1;
            

        } else {
            if(i_x %2 ==0) {
                // all couplers
                dot(lat_space*(i_x, i_y), p=coupler_colour);

                label((string) i_dot, lat_space*(i_x, i_y) + lat_label_gap , p=fnt_size_dots);
                i_dot = i_dot + 1;

            }
        }
    }
}






