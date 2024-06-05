settings.outformat = "pdf";
// settings.render=20;
defaultpen(fontsize(10pt));
unitsize(3mm);
usepackage("amsfonts");

size(7cm);

string c6 = "AD7A99"; // pink
string c6 = "7CDEDC"; // light blue


string c5 = "0F1980"; //purple

string wh = "FFFFFF";

string c1 = "1565C0";
string c3 = "C30934";
string c2 = "006F63";
string c4 = "F57F17";
string c5 = "8E24AA";
string c6 = "F57F17";
string colour5 = "0F1980"; //purple
string colour2 = "7CDEDC"; // light blue
string colour1 = "AD7A99"; // pink


real dot_size = 3.4;
pen quantum_dot_colour = rgb("000000");
pen coupler_colour = rgb(c4);
pen tunnelling_colour = rgb("6C6C6C");
int lat_L_x = 11;
int lat_L_y = 4;

real lat_space = 2;

pen lw_arrows = linewidth(1.9pt);
pen lw_tunnellings = linewidth(1pt);

real tunnelling_line_gap = 0.43;


// pen fnt_size_dots = fontsize(8pt);
// pair lat_label_gap = 0.7*(-1,+0.7);
// int i_dot = 1;
// pen lw_tunnellings = linewidth(0.1pt);



string tunnelling_colours_horiz[] = {c1,c2,c2,c5,wh,c3,wh,c2,c3,c3,c3,c2,c2,c5,wh,c3,wh,c2,c6,c5,c1,c2,c2,c5,wh,c3,c4,c2,c6,c5,c1,c2,c2,c5,wh,c3,c4,c2,c6,c5,c1,c2,c2,c5,wh,c4,c3,c2,c6,c5,c1,c2,c2,c5,wh,c3,wh,c2,c6,c5};
string tunnelling_colours_vert[] =  {c1,c2,wh,c5,c6,c2,c6,c5,c1,c2,c2,c5,wh,c6,c5,c1,c2,c2,c5,c6,c2,c6,c5,c1,c2,c2,c5,c2,c6,c5,c1,c2,c2,c5,c2,c6,wh,c1,c2,c2,c5,c6,c2,c6,c5,c1,c1,c1,c5,c2,c3,c3,c4,c5,c6,c1,c1,c2,c3,c6};
                                   

int i_tun_horiz = 0;
// draw horizontal tunnellings
for (int i_x = 0; i_x < lat_L_x - 1; ++i_x)
{
    for (int i_y=0; i_y>-lat_L_y; i_y=i_y - 1)
    {
        pair spot = lat_space*(2*i_x, 2*i_y);
        draw(spot+(tunnelling_line_gap,0)   -- spot+lat_space*(1,0)-(tunnelling_line_gap,0), p=rgb(tunnelling_colours_horiz[i_tun_horiz])+lw_tunnellings);
        draw(spot+lat_space*(1,0)+(tunnelling_line_gap,0)   -- spot+lat_space*(2,0)-(tunnelling_line_gap,0), p=rgb(tunnelling_colours_horiz[i_tun_horiz])+lw_tunnellings);
        ++i_tun_horiz;
    }

}

// draw vertical ltunnellings
int i_tun_vert = 0;
for (int i_x = 0; i_x < lat_L_x ; i_x = i_x + 1)
{
     for (int i_y=0; i_y>-lat_L_y+1; --i_y)
    {
        pair spot = lat_space*(2*i_x, 2*i_y);
        draw(spot-(0,tunnelling_line_gap)   -- spot-lat_space*(0,1)+(0,tunnelling_line_gap), p=rgb(tunnelling_colours_vert[i_tun_vert])+lw_tunnellings);
        draw(spot-lat_space*(0,1)-(0,tunnelling_line_gap)   -- spot-lat_space*(0,2)+(0,tunnelling_line_gap), p=rgb(tunnelling_colours_vert[i_tun_vert])+lw_tunnellings);
        ++i_tun_vert;
    
    }

}


int lat_L_x = 2*lat_L_x - 1;
int lat_L_y = 2*lat_L_y-1;
// draw dots
for (int i_y=0; i_y>-lat_L_y; --i_y)
{ 
    for (int i_x = 0; i_x < lat_L_x; ++i_x)
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






// // draw horizontal tunnellings
// real tunnelling_line_gap = 0.4;
// for (int i_x = 0; i_x < lat_L - 1; ++i_x)
// {
//     for (int i_y=0; i_y>-lat_L; i_y=i_y - 2)
//     {
//         pair spot = lat_space*(i_x, i_y);
//         draw(spot+(tunnelling_line_gap,0)   -- spot+lat_space*(1,0)-(tunnelling_line_gap,0), p=rgb("000000")+lw_tunnellings);
//     }

// }

// // draw vertical ltunnellings
// for (int i_x = 0; i_x < lat_L ; i_x = i_x + 2)
// {
//      for (int i_y=0; i_y>-lat_L+1; --i_y)
//     {
//         pair spot = lat_space*(i_x, i_y);
//         draw(spot-(0,tunnelling_line_gap)   -- spot-lat_space*(0,1)+(0,tunnelling_line_gap), p=rgb("000000")+lw_tunnellings);
    
//     }

// }



// // draw dots
// for (int i_y=0; i_y>-lat_L; --i_y)
// { 
//     for (int i_x = 0; i_x < lat_L; ++i_x)
//     {
//         if(i_y %2== 0) {
//         // do all dots
//             if(i_x %2 == 0){
//                 // quantum dot colour
//                 dot(lat_space*(i_x, i_y), p=quantum_dot_colour);
//             } else {
//                 // coupler
//                 dot(lat_space*(i_x, i_y), p=coupler_colour);
//             }

//             label((string) i_dot,lat_space*(i_x, i_y) + lat_label_gap, p=fnt_size_dots);
//             i_dot = i_dot + 1;
            

//         } else {
//             if(i_x %2 ==0) {
//                 // all couplers
//                 dot(lat_space*(i_x, i_y), p=coupler_colour);

//                 label((string) i_dot, lat_space*(i_x, i_y) + lat_label_gap , p=fnt_size_dots);
//                 i_dot = i_dot + 1;

//             }
//         }
//     }
// }





