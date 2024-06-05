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

real dot_size = 3;
pen quantum_dot_colour = rgb("000000");
pen coupler_colour = rgb(colour4);
pen tunnelling_colour = rgb("6C6C6C");

int lat_L = 5;
real lat_space = 0.9;

pen lw_arrows = linewidth(1.9pt);
pen lw_tunnellings = linewidth(0.02pt);
pen lw_renorm_tunnellings = linewidth(1.5pt);

real tunnelling_line_gap = 0.1;



// draw horizontal tunnellings
for (int i_x = 0; i_x < lat_L - 1; ++i_x)
{
    for (int i_y=0; i_y>-lat_L; i_y=i_y - 2)
    {
        pair spot = lat_space*(i_x, i_y);
        draw(spot+(tunnelling_line_gap,0)   -- spot+lat_space*(1,0)-(tunnelling_line_gap,0), p=tunnelling_colour+lw_tunnellings);
    }

}

// draw vertical ltunnellings
for (int i_x = 0; i_x < lat_L ; i_x = i_x + 2)
{
     for (int i_y=0; i_y>-lat_L+1; --i_y)
    {
        pair spot = lat_space*(i_x, i_y);
        draw(spot-(0,tunnelling_line_gap)   -- spot-lat_space*(0,1)+(0,tunnelling_line_gap), p=tunnelling_colour+lw_tunnellings);
    
    }

}


//draw renormalised tunnelling
pair spot = lat_space*(2, -1);
string[] cols = {"FFFFFF", colour6};
for(string col : cols) {
    draw(spot+(0,tunnelling_line_gap)   -- spot+lat_space*(0,1)-(0,tunnelling_line_gap), p=rgb(col)+lw_renorm_tunnellings);
    draw(spot-(0,tunnelling_line_gap)   -- spot-lat_space*(0,1)+(0,tunnelling_line_gap), p=rgb(col)+lw_renorm_tunnellings);
}

pair spot = lat_space*(1, -2);
string[] cols = {"FFFFFF", colour5};
for(string col : cols) {
    draw(spot+(tunnelling_line_gap,0)   -- spot+lat_space*(1,0)-(tunnelling_line_gap,0), p=rgb(col)+lw_renorm_tunnellings);
    draw(spot-(tunnelling_line_gap,0)   -- spot-lat_space*(1,0)+(tunnelling_line_gap,0), p=rgb(col)+lw_renorm_tunnellings); 
}



// draw dots
for (int i_y=0; i_y>-lat_L; --i_y)
{ 
    for (int i_x = 0; i_x < lat_L; ++i_x)
    {
        if(i_y %2== 0) {
        // do all dots
            if(i_x %2 == 0){
                // quantum dot colour
                dot(lat_space*(i_x, i_y), p=dot_size+quantum_dot_colour);
            } else {
                // coupler
                dot(lat_space*(i_x, i_y), p=dot_size+coupler_colour);
            }
            

        } else {
            if(i_x %2 ==0) {
                // all couplers
                dot(lat_space*(i_x, i_y), p=dot_size+coupler_colour);

            }
        }
    }
}






