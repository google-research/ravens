## Robotiq 2F 85 gripper
For this gripper, the following Github repo can be used as a reference: https://github.com/Shreeyak/robotiq.git

### mimic tag in URDF
This gripper is developed for ROS and uses the `mimic` tag within the URDF files to make the gripper move. From our research `mimic` tag within URDF is not supported by pybullet. To overcome this, one can use the `createConstraint` function. Please refer to [this](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/mimicJointConstraint.py) example from the bullet3 repo to see how to replicate a `mimic` joint:

```python
#a mimic joint can act as a gear between two joints
#you can control the gear ratio in magnitude and sign (>0 reverses direction)

import pybullet as p
import time
p.connect(p.GUI)
p.loadURDF("plane.urdf",0,0,-2)
wheelA = p.loadURDF("differential/diff_ring.urdf",[0,0,0])
for i in range(p.getNumJoints(wheelA)):
	print(p.getJointInfo(wheelA,i))
	p.setJointMotorControl2(wheelA,i,p.VELOCITY_CONTROL,targetVelocity=0,force=0)


c = p.createConstraint(wheelA,1,wheelA,3,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=1, maxForce=10000)

c = p.createConstraint(wheelA,2,wheelA,4,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, maxForce=10000)

c = p.createConstraint(wheelA,1,wheelA,4,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, maxForce=10000)


p.setRealTimeSimulation(1)
while(1):
	p.setGravity(0,0,-10)
	time.sleep(0.01)
#p.removeConstraint(c)

```


Details on `createConstraint` can be found in the pybullet [getting started](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.fq749wu22x4c) guide.

### Files in folder
Since parameters like gear ratio and direction are required, one can find the `robotiq_2f_85_mimic_joints.urdf` which contains the mimic tags as in original URDF, which can be used as a reference. It was generated from `robotiq/robotiq_2f_robot/robot/simple_rq2f85_pybullet.urdf.xacro` as so:
```
rosrun xacro xacro --inorder simple_rq2f85_pybullet.urdf.xacro
adaptive_transmission:="true" > robotiq_2f_85_mimic_joints.urdf
```

The URDF meant for use in pybullet is `robotiq_2f_85.urdf` and it is generated in a similar manner as above by running:
```
rosrun xacro xacro --inorder simple_rq2f85_pybullet.urdf.xacro > robotiq_2f_85.urdf
```
