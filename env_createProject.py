import os

# Create needed dirs given a main path and its name 

def create_path(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

def create_dirs(main, project_name):
    MAINname = main

    PROJECTname = str(project_name)
    SAVE_RESULTSname = "results"
    SAVE_WEIGHTSname = "weights"
    SAVE_RESULTvideo = "videoResults"
    
    PROJECTdir = os.path.join(MAINname, PROJECTname)
    create_path(PROJECTdir)

    SAVE_RESULTSdir = os.path.join(PROJECTdir, SAVE_RESULTSname)
    SAVE_WEIGHTSdir = os.path.join(PROJECTdir, SAVE_WEIGHTSname)
    SAVE_RESULTS_video = os.path.join(PROJECTdir, SAVE_RESULTvideo)
    
    create_path(SAVE_RESULTSdir)
    create_path(SAVE_WEIGHTSdir)
    create_path(SAVE_RESULTS_video)

    return PROJECTdir, SAVE_RESULTSdir, SAVE_WEIGHTSdir, SAVE_RESULTS_video