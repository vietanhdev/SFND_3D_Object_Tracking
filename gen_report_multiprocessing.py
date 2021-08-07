import subprocess
import multiprocessing

detectors = "SHITOMASI HARRIS FAST BRISK ORB AKAZE SIFT".split()
descriptors = "BRISK BRIEF ORB FREAK AKAZE SIFT".split()

with open("report2.md", "w") as f:
    f.write("# Mean Absolute Error Table\n")
    f.write("|Detector + Descriptor|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|Mean Error|\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

# Create list of combination
processingList = []
for detector in detectors:
    for descriptor in descriptors:
        processingList.append((detector, descriptor))

def processFunc(data):
    detector, descriptor = data
    print("{} + {}".format(detector, descriptor))
    out = subprocess.check_output(["./3D_object_tracking", detector, descriptor, "MAT_BF"], cwd="build").decode()
    print(out)
    with open("report2.md", "a") as f:
        f.write(out)
    return True

pool = multiprocessing.Pool(processes=6)
result_list = pool.map(processFunc, processingList)