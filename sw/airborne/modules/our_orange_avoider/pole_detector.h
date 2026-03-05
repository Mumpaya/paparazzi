/*
 * @file "modules/our_orange_avoider/pole_detector.h"
 * Detects orange poles in the front camera image.
 * Publishes centroid x-position and pixel count via ABI VISUAL_DETECTION.
 */

#ifndef POLE_DETECTOR_H
#define POLE_DETECTOR_H

extern void pole_detector_init(void);
extern void pole_detector_periodic(void);

#endif