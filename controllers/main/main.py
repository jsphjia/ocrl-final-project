# This file should be set as the controller for the Tesla robot node.
# Please do not alter this file - it may cause the simulation to fail.

import os
from pathlib import Path

# Import Webots-specific functions
from controller import Display
from vehicle import Driver

# Import functions from other scripts in controller folder
from util import *
from your_controller import CustomController
from evaluation import evaluation
from sl_dataset import ExpertDataLogger

TRACK_FILE = os.environ.get("TRACK_FILE", "raceline_xy.csv")
trajectory = getTrajectory(TRACK_FILE)

N_LAPS = 3  # change to however many laps you want
lapCount = 0
lapCooldown = False
ENABLE_EXPERT_LOG = os.environ.get("EXPERT_LOG", "1") == "1"
EXPERT_LOG_DIR = Path(__file__).resolve().parent / "data" / "expert" / Path(TRACK_FILE).stem
expertLogger = ExpertDataLogger(EXPERT_LOG_DIR, track_name=Path(TRACK_FILE).stem) if ENABLE_EXPERT_LOG else None

# Instantiate supervisor and functions
driver = Driver()
driver.setDippedBeams(True)
driver.setGear(1) # Torque control mode
throttleConversion = 15737
msToKmh = 3.6

# Access and set up displays
console = driver.getDevice("console")
speedometer = driver.getDevice("speedometer")
console.setFont("Arial Black", 10, True)
speedometerGraphic = speedometer.imageLoad("speedometer.png")
speedometer.imagePaste(speedometerGraphic, 0, 0, True)

consoleObject = DisplayUpdate(console)
speedometerObject = DisplayUpdate(speedometer)

# Get the time step of the current world
timestep = int(driver.getBasicTimeStep())

# Instantiate controller and start sensors
customController = CustomController(trajectory, expert_logger=expertLogger)
customController.startSensors(timestep)

# Initialize state storage vectors and completion conditions
XVec = []
YVec = []
deltaVec = []
xdotVec = []
ydotVec = []
psiVec = []
psidotVec = []
FVec = []
batterySocVec = []
minDist = []
passMiddlePoint = False
nearGoal = False
finish = False

while driver.step() != -1:

    # Call control update method
    X, Y, xdot, ydot, psi, psidot, F, delta, batterySoc = \
    customController.update(timestep)

    # Set control update output
    throttleCmd = clamp(F/throttleConversion, 0, 1)
    brakeCmd = clamp(-F/throttleConversion, 0, 1)
    driver.setThrottle(throttleCmd)
    driver.setBrakeIntensity(brakeCmd)
    driver.setSteeringAngle(-clamp(delta, np.radians(-30), np.radians(30)))
    
    # Check for halfway point/completion
    disError, nearIdx = closestNode(X, Y, trajectory)
    
    consoleObject.consoleUpdate(disError, nearIdx)
    console.drawText("SoC: " + str(round(batterySoc, 1)) + "%", 5, 80)
    speedometerObject.speedometerUpdate(speedometerGraphic, xdot*msToKmh)

    stepToMiddle = nearIdx - len(trajectory)/2.0
    if abs(stepToMiddle) < 100.0 and passMiddlePoint == False:
        passMiddlePoint = True
        
    if passMiddlePoint == True:
        console.drawText("Middle point passed.", 5, 40)
        
    nearGoal = nearIdx >= len(trajectory) - 50
    
    if not nearGoal:
        lapCooldown = False  # reset cooldown once car moves away from finish
    
    if nearGoal and passMiddlePoint and not lapCooldown:
        lapCount += 1
        lapCooldown = True
        if lapCount >= N_LAPS:
            console.drawText("Destination reached! :)", 5, 50)
            finalPosition = trajectory[-25]
            finish = True
            break
        else:
            console.drawText(f"Lap {lapCount} complete!", 5, 60)
            passMiddlePoint = False
            nearGoal = False
        
    XVec.append(X)
    YVec.append(Y)
    deltaVec.append(delta)
    xdotVec.append(xdot)
    ydotVec.append(ydot)
    psiVec.append(psi)
    psidotVec.append(psidot)
    FVec.append(F)
    batterySocVec.append(batterySoc)
    minDist.append(disError)
    
# Reset position and physics once loop is completed,
# and print evaluation to console
driver.setCruisingSpeed(0)
driver.setSteeringAngle(0)
if finish:
    evaluation(minDist, trajectory, XVec, YVec)
    showResult(trajectory, timestep, \
               XVec, YVec, deltaVec, xdotVec, ydotVec, \
               FVec, psiVec, psidotVec, minDist, batterySocVec)

if expertLogger is not None:
    expertLogger.close()
