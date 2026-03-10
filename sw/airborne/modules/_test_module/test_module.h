/*
 * Copyright (C) 2026 Douwe Rijs <Douwe_r@standofl.nl>
 *
 * This file is part of paparazzi
 *
 * paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with paparazzi; see the file COPYING.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

/** @file "modules/_test_module/test_module.h"
 * @author Douwe Rijs <Douwe_r@standofl.nl>
 * Edge-based floor avoider: uses Canny edges on the bottom camera to
 * detect non-floor areas and steer away from them.
 */

#ifndef TEST_MODULE_H
#define TEST_MODULE_H

#include <stdint.h>

// Tunable settings (exposed to GCS)
extern float green_max_speed;       // forward speed [m/s]
extern float green_heading_rate;    // turn rate [rad/s]
extern float edge_threshold;        // edge fraction above which we avoid

extern void init_func(void);
extern void periodic_func(void);
extern void orange_avoider_guided_retreat(void);

#endif  // TEST_MODULE_H
