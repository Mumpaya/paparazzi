#ifndef AVOIDER_H
#define AVOIDER_H

#include <stdint.h>

extern float max_speed;
extern float heading_rate;
extern float center_threshold;
extern float danger_threshold;

extern void avoider_init(void);
extern void avoider_periodic(void);
extern void orange_avoider_guided_retreat(void);

#endif