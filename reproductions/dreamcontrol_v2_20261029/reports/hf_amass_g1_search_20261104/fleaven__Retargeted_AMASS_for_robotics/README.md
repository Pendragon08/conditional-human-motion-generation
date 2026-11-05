---
license: cc-by-4.0
task_categories:
- robotics
language:
- en
tags:
- AMASS
- Retarget
- Robotics
- Humanoid
pretty_name: Retargeted AMASS for Robotics
size_categories:
- 10K<n<100K
---
# Retargeted AMASS for Robotics

## Project Overview

This project aims to retarget motion data from the AMASS dataset to various robot models and open-source the retargeted data to facilitate research and applications in robotics and human-robot interaction. AMASS (Archive of Motion Capture as Surface Shapes) is a high-quality human motion capture dataset, and the SMPL-X model is a powerful tool for generating realistic human motion data.

By adapting the motion data from AMASS to different robot models, we hope to provide a more diverse and accessible motion dataset for robot training and human-robot interaction.

## Dataset Content

This open-source project includes the following:
1. **Retargeted Motions**: Motion files retargeted from AMASS to various robot models.

   - **Unitree G1**: 

     <iframe src="//player.bilibili.com/player.html?bvid=BV1zd6iYkEZ2&page=1&high_quality=1&danmaku=0" allowfullscreen="allowfullscreen" width="100%" height="500" scrolling="no" frameborder="0" sandbox="allow-top-navigation allow-same-origin allow-forms allow-scripts"></iframe>

     The retargeted motions for the Unitree G1 robot are generated based on the official open-source model provided by Unitree. 

     https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_rev_1_0.xml

     The joint positions comply with the constraints defined in the XML file.

     data shape：[-1,36]

     ​		0:3 	root world position

     ​		3:7	 root quaternion rotation, order: xyzw

     ​		7:36   joint positions

     joint order:

     ```txt
     	left_hip_pitch_joint
         left_hip_roll_joint
         left_hip_yaw_joint
         left_knee_joint
         left_ankle_pitch_joint
         left_ankle_roll_joint
         right_hip_pitch_joint
         right_hip_roll_joint
         right_hip_yaw_joint
         right_knee_joint
         right_ankle_pitch_joint
         right_ankle_roll_joint
         waist_yaw_joint
         waist_roll_joint
         waist_pitch_joint
         left_shoulder_pitch_joint
         left_shoulder_roll_joint
         left_shoulder_yaw_joint
         left_elbow_joint
         left_wrist_roll_joint
         left_wrist_pitch_joint
         left_wrist_yaw_joint
         right_shoulder_pitch_joint
         right_shoulder_roll_joint
         right_shoulder_yaw_joint
         right_elbow_joint
         right_wrist_roll_joint
         right_wrist_pitch_joint
         right_wrist_yaw_joint
     ```

2. **update**
    2025-4-15: Optimized the handling of elbow and knee joints.


3. **Usage Examples**: Code examples on how to use the retargeted data.

   ./g1/visualize.py

   

4. **License Files**: Original license information for each sub-dataset within AMASS.

   

## License

The retargeted data in this project is derived from the AMASS dataset and therefore adheres to the original license terms of AMASS. Each sub-dataset within AMASS may have different licenses, so please ensure compliance with the following requirements when using the data:
- **Propagate Original Licenses**: When using or distributing the retargeted data, you must include and comply with the original licenses of the sub-datasets within AMASS.
- **Attribution Requirements**: Properly cite this work and the original authors and sources of the AMASS dataset and its sub-datasets.

For detailed license information, please refer to the `LICENSE` file in this project.



## Acknowledgments

This project is built on the AMASS dataset and the SMPL-X model. Special thanks to the research team at the Max Planck Institute for Intelligent Systems for providing this valuable resource.



## Citation

If you use the data or code from this project, please cite this work and relevant papers for AMASS and SMPL-X:
```bibtex
@misc{Retargeted_AMASS_R,
  title={Retargeted AMASS for Robotics},
  author={Kun Zhao},
  url={https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics}
}

@inproceedings{AMASS2019,
  title={AMASS: Archive of Motion Capture as Surface Shapes},
  author={Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2019}
}

@inproceedings{SMPL-X2019,
  title={Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author={Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## Contact

For any questions or suggestions, please contact:
- **Kun Zhao**: fleaven@hotmail.com



For more information, follow my Xiaohongshu and Bilibili:

https://www.xiaohongshu.com/user/profile/60cdc5360000000001007e33

https://space.bilibili.com/678369952