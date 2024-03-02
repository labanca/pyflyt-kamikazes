import time

import numpy as np
import pybullet as p
import pybullet_data

from modules.utils import generate_start_pos_orn

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf", [0, 0, 0.05])




#cubeStartPos, _ =   [[0.5, -2.0, 0.5], [2.0, 1.0, 2]]

cubeStartPos, start_orn, formation_center = generate_start_pos_orn(num_lm=20, num_lw=5)
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, np.pi ])

for cube in cubeStartPos:
    duckId = p.loadURDF(
        "C:\\projects\\pyflyt-kamikazes\\venv\\Lib\\site-packages\\PyFlyt\\models\\vehicles\\cf2x\\cf2x.urdf",
        cube,
        cubeStartOrientation,
        globalScaling=5,
        useFixedBase=True,
    )

LightBlue = [0.5, 0.5, 1, 1]
Red = [1, 0, 0, 1]
LightRed = [1, 0.5, 0.5, 1]
DarkBlue = [0, 0, 0.8, 1]
[p.changeVisualShape(i, -1, rgbaColor=DarkBlue) for i in range(20,26)]

cameraDistance=2.84
cameraPitch=-8.8
cameraYaw = -25.60
cameraTargetPosition = [2,0,3]

p.resetDebugVisualizerCamera(cameraDistance=cameraDistance,
                             cameraYaw=cameraYaw,
                             cameraPitch=cameraPitch,
                             cameraTargetPosition=cameraTargetPosition
                             )



# x_axis_ground = p.addUserDebugLine(
#     lineFromXYZ=[0, 0, 0.05],
#     lineToXYZ=[1, 0, 0.05],
#     lineColorRGB=[1, 0, 0],
#     lineWidth=5.0,
#     lifeTime=100000,
#     parentObjectUniqueId=planeId,
#     parentLinkIndex=-1,
# )
# # x_G = p.addUserDebugText(
# #     text="X_G",
# #     textPosition=[1, 0, 0.05],
# #     textColorRGB=[1, 0, 0],
# #     parentObjectUniqueId=planeId,
# #     parentLinkIndex=-1,
# # )
#
# y_axis_ground = p.addUserDebugLine(
#     lineFromXYZ=[0, 0, 0.05],
#     lineToXYZ=[0, 1, 0.05],
#     lineColorRGB=[0, 1, 0],
#     lineWidth=5.0,
#     lifeTime=100000,
#     parentObjectUniqueId=planeId,
#     parentLinkIndex=-1,
# )
# # y_G = p.addUserDebugText(
# #     text="Y_G",
# #     textPosition=[0, 1, 0.05],
# #     textColorRGB=[0, 1, 0],
# #     parentObjectUniqueId=planeId,
# #     parentLinkIndex=-1,
# # )
#
# z_axis_ground = p.addUserDebugLine(
#     lineFromXYZ=[0, 0, 0.05],
#     lineToXYZ=[0, 0, 1.05],
#     lineColorRGB=[0, 0, 1],
#     lineWidth=5.0,
#     lifeTime=100000,
#     parentObjectUniqueId=planeId,
#     parentLinkIndex=-1,
# )
# # z_G = p.addUserDebugText(
# #     text="Z_G",
# #     textPosition=[0, 0, 1.05],
# #     textColorRGB=[0, 0, 1],
# #     parentObjectUniqueId=planeId,
# #     parentLinkIndex=-1,
# # )

while True:
    p.stepSimulation()
    time.sleep(1/60)
