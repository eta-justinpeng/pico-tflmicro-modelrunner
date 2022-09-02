#include "main_functions.h"
#include<stdio.h>
#include <pico/stdlib.h>

// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.
int main(int argc, char* argv[]) {
    setup();
    while(1){
        infer();
        sleep_ms(1000);
    }
}