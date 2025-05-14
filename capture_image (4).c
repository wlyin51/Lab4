// #include "address_map_arm.h"
#include <stdio.h>
#include <time.h>
#include "bmp_utility.h" 

#define KEY_BASE              0xFF200050
#define VIDEO_IN_BASE         0xFF203060
#define FPGA_ONCHIP_BASE      0xC8000000

/* This program demonstrates the use of the D5M camera with the DE1-SoC Board
 * It performs the following: 
 * 	1. Capture one frame of video when any key is pressed.
 * 	2. Display the captured frame when any key is pressed.		  
*/
/* Note: Set the switches SW1 and SW2 to high and rest of the switches to low for correct exposure timing while compiling and the loading the program in the Altera Monitor program.
*/

#define WIDTH  240
#define HEIGHT 240
#define ROW    512  //available, after 320 not used


unsigned char bw[HEIGHT][WIDTH];
unsigned char scaled[28][28];

volatile int count = 0;



static const unsigned char font_5x7[12][7] = {
    {0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1F}, // '0'
    {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}, // '1'
    {0x1F, 0x01, 0x01, 0x1F, 0x10, 0x10, 0x1F}, // '2'
    {0x1F, 0x01, 0x01, 0x0F, 0x01, 0x01, 0x1F}, // '3'
    {0x11, 0x11, 0x11, 0x1F, 0x01, 0x01, 0x01}, // '4'
    {0x1F, 0x10, 0x10, 0x1F, 0x01, 0x01, 0x1F}, // '5'
    {0x1F, 0x10, 0x10, 0x1F, 0x11, 0x11, 0x1F}, // '6'
    {0x1F, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00}, // '7'
    {0x1F, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x1F}, // '8'
    {0x1F, 0x11, 0x11, 0x1F, 0x01, 0x01, 0x1F}, // '9'
    {0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00}, // ':' (index 10)
    {0x00, 0x00, 0x1F, 0x00, 0x00, 0x00, 0x00}  // '-' (index 11)
};

void draw_char(volatile short *video_mem_ptr, int x, int y, char ch) {
    int i, j, index;
    if(ch >= '0' && ch <= '9')
        index = ch - '0';
    else if(ch == ':')
        index = 10;
    else if(ch == '-') 
        index = 11;
    else 
        return;
    for(i = 0; i < 7; i++){
        unsigned char rowPattern = font_5x7[index][i];
        for(j = 0; j < 5; j++){
            if(rowPattern & (1 << (4 - j)))
                video_mem_ptr[(y + i) * ROW + (x + j)] = 0xFFFF;
        }
    }
}

void draw_string(volatile short *video_mem_ptr, int x, int y, const char *str) {
    int i = 0;
    while(str[i] != '\0'){
        draw_char(video_mem_ptr, x, y, str[i]);
        x += 6;
        i++;
    }
}

void add_timestamp(volatile short *video_mem_ptr) {
    time_t now;
    struct tm *tm_info;
    char time_str[16];
    char hour_str[3], min_str[3], sec_str[3];
    
    time(&now);
    tm_info = localtime(&now);
    
    int est = tm_info->tm_hour - 6;
    if (est < 0) {
        est += 24;
    }
    
    // Manually format hour, minute, and second as two-digit strings.
    // For hours:
    if (est < 10)
        sprintf(hour_str, "0%d", est);
    else
        sprintf(hour_str, "%d", est);
    
    // For minutes:
    if (tm_info->tm_min < 10)
        sprintf(min_str, "0%d", tm_info->tm_min);
    else
        sprintf(min_str, "%d", tm_info->tm_min);
    
    // For seconds:
    if (tm_info->tm_sec < 10)
        sprintf(sec_str, "0%d", tm_info->tm_sec);
    else
        sprintf(sec_str, "%d", tm_info->tm_sec);
    
    sprintf(time_str, "%s:%s:%s", hour_str, min_str, sec_str);
    
    draw_string(video_mem_ptr, 5, 5, time_str);
}

















void flip_mirror(volatile short *Video_mem_ptr) {
    int x, y;
    short temp;
    for (y = 0; y < HEIGHT / 2; y++) {
        for (x = 0; x < WIDTH; x++) {
            temp = Video_mem_ptr[y * ROW + x];
            Video_mem_ptr[y * ROW + x] = Video_mem_ptr[(HEIGHT - 1 - y) * ROW + x];
        	Video_mem_ptr[(HEIGHT - 1 - y) * ROW + x] = temp;
        }
    }
    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH / 2; x++) {
            temp = Video_mem_ptr[y * ROW + x];
            Video_mem_ptr[y * ROW + x] = Video_mem_ptr[y * ROW + (WIDTH - 1 - x)];
            Video_mem_ptr[y * ROW + (WIDTH - 1 - x)] = temp;
        }
    }
}

void convert_to_bw(volatile short *Video_mem_ptr) {
    int x, y;
    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH; x++) {
            short pixel = Video_mem_ptr[y * ROW + x];
            int r = (pixel >> 11) & 0x1F; //11111 mask, keep the 5 bits that are shifted to the end
            int g = (pixel >> 5)  & 0x3F; //111111 mask, keep the 6 bits that are shifted to the end
			int b = pixel & 0x1F;
            int r8 = (r * 255) / 31;
            int g8 = (g * 255) / 63;
            int b8 = (b * 255) / 31;
            int brightness = (r8 + g8 + b8) / 3;
            if (brightness > 128)
                Video_mem_ptr[y * ROW + x] = 0xFFFF; 
            else
                Video_mem_ptr[y * ROW + x] = 0x0000;
        }
    }
}

void invert_pixels(volatile short *Video_mem_ptr) {
    int x, y;
    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH; x++) {
            Video_mem_ptr[y * ROW + x] = ~(Video_mem_ptr[y * ROW + x]) & 0xFFFF;
        }
    }
}


void add_picture_counter(volatile short *video_mem_ptr, int count) {
    char buffer[16];
    sprintf(buffer, "%03d", count);
    draw_string(video_mem_ptr, WIDTH - 6 * 6, 5, buffer);
}




int main(void)
{
	volatile int * KEY_ptr				= (int *) KEY_BASE;
	volatile int * Video_In_DMA_ptr	= (int *) VIDEO_IN_BASE;
	volatile short * Video_Mem_ptr	= (short *) FPGA_ONCHIP_BASE;

	int x, y;

	*(Video_In_DMA_ptr + 3)	= 0x4;				// Enable the video

	while (1)
	{
		if (*KEY_ptr != 0)						// check if any KEY was pressed
		{
			*(Video_In_DMA_ptr + 3) = 0x0;			// Disable the video to capture one frame
			while (*KEY_ptr != 0);				// wait for pushbutton KEY release
			break;
		}
	}

	while (1)
	{
		if (*KEY_ptr != 0)						// check if any KEY was pressed
		{
			break;
		}
	}

// --- Copy the 240×240 frame into an 8-bit grayscale buffer ---
for (y = 0; y < HEIGHT; y++) {
    for (x = 0; x < WIDTH; x++) {
        short pixel = Video_Mem_ptr[y * ROW + x];
        int red5 = (pixel >> 11) & 0x1F;
        bw[y][x] = (red5 * 255) / 31;
    }
}

    scaleImagePreservingAspectRatio(&bw[0][0], &scaled[0][0], WIDTH, HEIGHT, 28, 28
    );


    saveImageGrayscale("first_image_mnist.bmp", &scaled[0][0], 28, 28);


    count++;
    add_timestamp(Video_Mem_ptr);
    add_picture_counter(Video_Mem_ptr, count);


    while (1) {
        int keys = *KEY_ptr;
        if (keys & 0x1) {
         
            count++;              
            add_picture_counter(Video_Mem_ptr, count); 
            flip_mirror(Video_Mem_ptr);
            *(Video_In_DMA_ptr + 3) = 0x0;			// Disable the video to capture one frame
            while (*KEY_ptr & 0x1); // Wait until KEY0 is released
        }
        if (keys & 0x2) {
     
            count++;              
            add_picture_counter(Video_Mem_ptr, count); 
            convert_to_bw(Video_Mem_ptr);
            *(Video_In_DMA_ptr + 3) = 0x0;
            while (*KEY_ptr & 0x2); // Wait until KEY1 is released
        }
        if (keys & 0x4) {
   
            count++;              
            add_picture_counter(Video_Mem_ptr, count); 
            invert_pixels(Video_Mem_ptr);
            *(Video_In_DMA_ptr + 3) = 0x0;
            while (*KEY_ptr & 0x4); // Wait until KEY2 is released
        }

        if (keys & 0x8) {
            *(Video_In_DMA_ptr + 3)	= 0x4;
            add_timestamp(Video_Mem_ptr);
        }
    
    }

    

	return 0;

}
