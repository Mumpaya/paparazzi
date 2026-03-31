# Bottom Camera CV Modules (Extern Variable Output)

This adds two standalone modules for real-drone runtime use. They do not use ABI messages.
Both modules expose their outputs via extern globals protected with a mutex in the processing callback.

## Modules

- `cv_bottom_flow_velocity`: planar velocity from bottom-camera optical flow.
- `cv_bottom_edge_detector`: edge/ground quality detector from bottom-camera color + texture.

## Exported state

- `cv_bottom_flow_velocity_state` in `cv_bottom_flow_velocity.h`
- `cv_bottom_edge_detector_state` in `cv_bottom_edge_detector.h`

Each state has:
- computed values
- `updated` flag set when new output is available
- `valid` flag indicating whether output passes basic quality gates

## Airframe usage

Add modules to your airframe:

```xml
<module name="cv_bottom_flow_velocity">
  <define name="CV_BOTTOM_FLOW_VELOCITY_CAMERA" value="bottom_camera"/>
</module>

<module name="cv_bottom_edge_detector">
  <define name="CV_BOTTOM_EDGE_DETECTOR_CAMERA" value="bottom_camera"/>
</module>
```

Then read extern variables from your guidance/control module.

## Notes

- Default velocity scale uses a pinhole approximation and configurable altitude/FOV.
- For real drone use, tune thresholds and scaling constants in the airframe defines.

