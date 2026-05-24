# Plank — Complete Feature Vector

| ID | Feature | Computation | Class Targeted |
|---|---|---|---|
| P1 | Left Hip Angle | L_Shoulder – L_Hip – L_Knee | H (high back), L (low back) |
| P2 | Right Hip Angle | R_Shoulder – R_Hip – R_Knee | H, L (asymmetry signal) |
| P3 | Left Shoulder Stack | L_Hip – L_Shoulder – L_Elbow | Body line at shoulder |
| P4 | Right Shoulder Stack | R_Hip – R_Shoulder – R_Elbow | Body line at shoulder |
| P5 | Left Knee Angle | L_Hip – L_Knee – L_Ankle | Leg straightness |
| P6 | Right Knee Angle | R_Hip – R_Knee – R_Ankle | Leg straightness |
| P7 | Hip Symmetry Delta | abs(P1 − P2) | Asymmetric hip twist |
| P8 | Shoulder–Wrist Vertical Offset | norm_shoulder_y - norm_wrist_y | Wrist under shoulder check |

**Total Features:** 8

> **P7** is a derived symmetry feature — in a correct plank, left and right hip angles should be nearly identical. A large delta signals rotation or twist in the torso that doesn't show up in individual angle readings.

---

# Warrior II — Complete Feature Vector

| ID | Feature | Computation | Class Targeted |
|---|---|---|---|
| W1 | Front Knee Angle | Front_Hip – Front_Knee – Front_Ankle | Shallow_Front_Lunge |
| W2 | Back Knee Angle | Back_Hip – Back_Knee – Back_Ankle | Correct_Form validation |
| W3 | Left Shoulder Angle | L_Hip – L_Shoulder – L_Elbow | Drooping_Arms |
| W4 | Right Shoulder Angle | R_Hip – R_Shoulder – R_Elbow | Drooping_Arms |
| W5 | Arm Symmetry Delta | abs(W3 − W4) | One-arm droop vs both |
| W6 | Torso Vertical Angle | Shoulder_mid→Hip_mid vs Y-axis | Leaning_Torso |
| W7 | Stance Width Ratio | Ankle–Ankle dist / Torso Length | Shallow_Front_Lunge support |
| W8 | Front Knee Lateral Offset | norm_front_knee_x − norm_front_ankle_x | Front_Knee_Cave |
| W9 | Hip Openness Angle | L_Hip – Hip_mid – R_Hip vs frontal plane | Validates hips facing sideways |

**Total Features:** 9

> **W5** is important — a person can droop one arm while keeping the other level. Looking only at individual shoulder angles won't tell you whether one or both are affected, but the symmetry delta immediately separates "single arm droop" from "bilateral fatigue droop."

> **W9** captures whether the hips are properly rotated open to face the long side of the mat. A closed hip (hips facing forward instead of sideways) is a structural error that none of W1–W8 can detect.

---

# Mountain Pose — Complete Feature Vector

| ID | Feature | Computation | Class Targeted |
|---|---|---|---|
| M1 | Left Knee Angle | L_Hip – L_Knee – L_Ankle | Hyperextended_Knees |
| M2 | Right Knee Angle | R_Hip – R_Knee – R_Ankle | Hyperextended_Knees |
| M3 | Knee Symmetry Delta | abs(M1 − M2) | Unilateral hyperextension |
| M4 | Pelvis–Spine Angle | Shoulder_mid – Hip_mid – Knee_mid | Swayback, Posterior_Tilt |
| M5 | Spine Vertical Angle | Shoulder_mid→Hip_mid vs Y-axis | Both pelvic tilt classes |
| M6 | Ear–Shoulder Forward Offset | norm_ear_x − norm_shoulder_x | Rounded_Shoulders_FHP |
| M7 | Shoulder–Hip Forward Offset | norm_shoulder_x − norm_hip_x | Rounded_Shoulders_FHP |
| M8 | Full Body Vertical Alignment | norm_ear_x − norm_ankle_x | Overall posture lean |

**Total Features:** 8

> **M8** is a holistic alignment feature — in perfect Mountain Pose, the ear, shoulder, hip, and ankle all lie on the same vertical line. Expressing this as a single ear-to-ankle normalized horizontal offset gives your model a fast overall alignment check that complements the per-segment features.

> **M6** and **M7** together allow your model to distinguish pure forward head posture (M6 large, M7 small) from full upper body lean (both large) — which helps the `Rounded_Shoulders_FHP` class generalize correctly across different body types.