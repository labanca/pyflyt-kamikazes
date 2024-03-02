import time

import numpy as np
import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf", [0, 0, 0.05])

cubeStartPos = [0.5, -2.0, 0.5]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, np.pi ])
duckId = p.loadURDF(
    "C:\\projects\\pyflyt-kamikazes\\venv\\Lib\\site-packages\\PyFlyt\\models\\vehicles\\cf2x\\cf2x.urdf",
    cubeStartPos,
    cubeStartOrientation,
    globalScaling=10,
    useFixedBase=True,
)

x_axis = p.addUserDebugLine(
    lineFromXYZ=[0, 0, 0],
    lineToXYZ=[-1, 0, 0],
    lineColorRGB=[1, 0, 0],
    lineWidth=5.0,
    lifeTime=100000,
    parentObjectUniqueId=duckId,
    parentLinkIndex=-1,
)

y_axis = p.addUserDebugLine(
    lineFromXYZ=[0, 0, 0],
    lineToXYZ=[0, 0, 1],
    lineColorRGB=[0, 0, 1],
    lineWidth=5.0,
    lifeTime=100000,
    parentObjectUniqueId=duckId,
    parentLinkIndex=-1,
)

z_axis = p.addUserDebugLine(
    lineFromXYZ=[0, 0, 0],
    lineToXYZ=[0, -1, 0],
    lineColorRGB=[0, 1, 0],
    lineWidth=5.0,
    lifeTime=100000,
    parentObjectUniqueId=duckId,
    parentLinkIndex=-1,
)

x_axis_ground = p.addUserDebugLine(
    lineFromXYZ=[0, 0, 0.05],
    lineToXYZ=[1, 0, 0.05],
    lineColorRGB=[1, 0, 0],
    lineWidth=5.0,
    lifeTime=100000,
    parentObjectUniqueId=planeId,
    parentLinkIndex=-1,
)
# x_G = p.addUserDebugText(
#     text="X_G",
#     textPosition=[1, 0, 0.05],
#     textColorRGB=[1, 0, 0],
#     parentObjectUniqueId=planeId,
#     parentLinkIndex=-1,
# )

y_axis_ground = p.addUserDebugLine(
    lineFromXYZ=[0, 0, 0.05],
    lineToXYZ=[0, 1, 0.05],
    lineColorRGB=[0, 1, 0],
    lineWidth=5.0,
    lifeTime=100000,
    parentObjectUniqueId=planeId,
    parentLinkIndex=-1,
)
# y_G = p.addUserDebugText(
#     text="Y_G",
#     textPosition=[0, 1, 0.05],
#     textColorRGB=[0, 1, 0],
#     parentObjectUniqueId=planeId,
#     parentLinkIndex=-1,
# )

z_axis_ground = p.addUserDebugLine(
    lineFromXYZ=[0, 0, 0.05],
    lineToXYZ=[0, 0, 1.05],
    lineColorRGB=[0, 0, 1],
    lineWidth=5.0,
    lifeTime=100000,
    parentObjectUniqueId=planeId,
    parentLinkIndex=-1,
)
# z_G = p.addUserDebugText(
#     text="Z_G",
#     textPosition=[0, 0, 1.05],
#     textColorRGB=[0, 0, 1],
#     parentObjectUniqueId=planeId,
#     parentLinkIndex=-1,
# )

while True:
    p.stepSimulation()
    time.sleep(1/60)
