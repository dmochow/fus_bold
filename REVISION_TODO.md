## Retrospective Derated In Situ Ultrasound Estimate

### Objective

Build a first-pass, subject-specific derated estimate of ultrasound exposure at the brain target using:anatomical MRI

- skull mask (you already have them)
- target coordinate (you already calculated these)
- scalp transducer location / beam direction (you calculated the scalp transducer location)
- free-water calibration values (we have these measurements)
    - https://github.com/dmochow/fus_bold/blob/main/code/visualize_beampattern.ipynb
    - the data is at REPO_ROOT/data/watertank/ and is used by the notebook above

The purpose is to estimate, for each participant:
- skull thickness along beam axis
- intracranial path length to target
- estimated in situ pressure at target
- estimated in situ intensity at target
- estimated mechanical index (MI)
- uncertainty range via sensitivity analysis

This is a straight-line analytical estimate, not a full-wave simulation. 

For the brain path, use **0.5 dB/cm/MHz** as the default attenuation coefficient. 

### Inputs

For each subject:
- T1
- binary skull mask
- target coordinate in image space
- transducer/scalp entry coordinate
- beam direction vector, or enough information to define the line from scalp to target
    - I think it's fine to assume this to be the vector pointing from the transducer center to the target

Global constants:
- ultrasound frequency = 0.5 MHz
- free-water peak pressure p_water_MPa = 0.258 MPa
- free-water intensity I_water = 2.22 W/cm^2
- assumed skull attenuation coefficient range: {20, 40, 60} dB/cm/MHz 
- assumed fixed interface/insertion loss range: {2, 4, 6} dB

### Main tasks

#### Task 1: Geometry extraction

For each subject:

	1.	Define beam axis from scalp entry point toward target.

	2.	Find intersections of beam axis with:
	    - outer skull table
	    - inner skull table
	
    3.	Compute:
	    - d_skull_mm = skull thickness along beam axis
	    - d_brain_mm = distance from inner skull table to target
	
    4.	Save QC image showing skull mask, beam axis, and target.

#### Task 2: Derating calculator

For each subject, compute attenuation:

    brain attenuation:
    $L_{brain} = 0.5 \cdot f_{MHz} \cdot d_{brain,cm}$

    skull attenuation:
    $L_{skull} = \alpha_{skull} \cdot f_{MHz} \cdot d_{skull,cm}$

    total attenuation:
    $L_{total} = L_{interface} + L_{skull} + L_{brain}$

Then compute:

    pressure transmission:
    $T_p = 10^{-L_{total}/20}$

    in situ pressure:
    $p_{target} = p_{water} \cdot T_p$

    intensity transmission:
    $T_I = 10^{-L_{total}/10}$

    in situ intensity:
    $I_{target} = I_{water} \cdot T_I$

    MI estimate:
    $MI = \frac{p_{target}}{\sqrt{f_{MHz}}}$


#### Task 3: Sensitivity analysis

Run the calculator under multiple assumptions:
- low / medium / high skull attenuation
- low / medium / high interface loss

For each subject, report:
- central estimate
- low estimate
- high estimate

#### Task 4: Summary outputs

Create:
- per-subject CSV
- group summary CSV
- histogram of skull thickness
- histogram or boxplot of estimated in situ pressure
- histogram or boxplot of estimated MI


### Pseudocode


    for subject in subjects:

        # Load data
        anat = load_volume(subject.anatomical_image)
        skull = load_mask(subject.skull_mask)
        target = load_target_coordinate(subject.target_coord)
        scalp_point = load_scalp_coordinate(subject.scalp_coord)

        # Define beam axis
        beam_vec = normalize(target - scalp_point)

        # Find skull intersections along beam axis
        outer_pt = first_intersection_with_mask_boundary(scalp_point, beam_vec, skull)
        inner_pt = second_intersection_with_mask_boundary(scalp_point, beam_vec, skull)

        # Distances
        d_skull_mm = distance(outer_pt, inner_pt)
        d_brain_mm = distance(inner_pt, target)

        # Convert to cm
        d_skull_cm = d_skull_mm / 10.0
        d_brain_cm = d_brain_mm / 10.0

        # Sensitivity grid
        results = []
        for alpha_skull in skull_alpha_values:
            for L_interface in interface_loss_values:

                L_brain = 0.5 * f_MHz * d_brain_cm
                L_skull = alpha_skull * f_MHz * d_skull_cm
                L_total = L_interface + L_brain + L_skull

                T_p = 10 ** (-L_total / 20.0)
                T_I = 10 ** (-L_total / 10.0)

                p_target = p_water_MPa * T_p
                I_target = I_water * T_I
                MI_est = p_target / sqrt(f_MHz)

                results.append({
                    "alpha_skull": alpha_skull,
                    "L_interface": L_interface,
                    "L_total_dB": L_total,
                    "p_target_MPa": p_target,
                    "I_target": I_target,
                    "MI_est": MI_est
                })

        # Save summary for subject
        save_subject_results(subject.id, d_skull_mm, d_brain_mm, results)



### Minimum required
	1.	Geometry table for all subjects with:
	    - subject ID
        - skull thickness along beam axis
        - brain path length to target

	2.	Derating script that:
        - reads geometry table
        - computes in situ pressure/intensity
        - runs sensitivity analysis
	
    3.	Results table with:
        - central estimate
        - low/high range
        - MI estimate per subject

	4.	QC figures
        - one image per subject showing beam axis and target
        - one group figure summarizing skull thickness and pressure estimates




### Note to self

In the paper we will need to state explicitly our assumptions:

- straight-ray model only
- no refraction correction
- no focal shift estimate
- no subject-specific skull acoustic property mapping from CT
- no thermal simulation
- intended as a conservative retrospective estimate, not precise dosimetry
