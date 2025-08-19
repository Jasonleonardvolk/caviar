# ELFIN Robotics Example Project

This project demonstrates the ELFIN module system with a simple robotics application. It shows how to use imports and templates to build a modular robotic control system.

## Project Structure

```
robotics/
├── README.md             # This file
├── math/
│   ├── vector.elfin      # Vector template
│   └── matrix.elfin      # Matrix template
├── sensors/
│   ├── distance.elfin    # Distance sensor template
│   └── camera.elfin      # Camera sensor template
├── controllers/
│   ├── pid.elfin         # PID controller template
│   └── lqr.elfin         # LQR controller template
├── components/
│   ├── arm.elfin         # Robot arm component
│   └── wheel.elfin       # Robot wheel component
└── robot.elfin           # Main robot concept
```

## Running the Example

To compile and run the example:

```bash
cd robotics
../../module_compiler.py robot.elfin
```

## Key Features Demonstrated

This example showcases several key features of the ELFIN module system:

1. **Module Organization**: Code is organized into logical directories and files
2. **Imports**: Components are imported from other modules
3. **Templates**: Reusable components with parameters
4. **Template Instantiation**: Creating instances with specific values
5. **Template Composition**: Building complex components from simpler ones

## Learning Path

To understand the example, start with these files in order:

1. `math/vector.elfin`: Shows a basic template definition
2. `controllers/pid.elfin`: Shows a template with required parameters
3. `components/wheel.elfin`: Shows importing and using templates
4. `robot.elfin`: Shows putting everything together

## Extending the Example

Try adding new components to extend the example:

1. Add a new sensor type (e.g., gyroscope)
2. Add a new controller type (e.g., MPC)
3. Add a new component (e.g., gripper)
4. Modify the robot to use your new components
