from sktime.datasets import load_from_tsfile_to_dataframe
import os
import pandas as pd
import pathlib

def load_datasets():
    train_x, train_y = load_from_tsfile_to_dataframe(os.getcwd() + r"\OpenPose_MP\Full25BodyParts" + r"\TRAIN_full_X.ts")
    test_x, test_y = load_from_tsfile_to_dataframe(os.getcwd() + r"\OpenPose_MP\Full25BodyParts" + r"\TEST_full_X.ts")

    train_x.columns = ['Nose_X', 'Neck_X', 'RShoulder_X', 'RElbow_X', 'RWrist_X', 'LShoulder_X', 'LElbow_X', 'LWrist_X', 'MidHip_X', 'RHip_X', 'RKnee_X', 'RAnkle_X', 'LHip_X', 'LKnee_X', 'LAnkle_X', 'REye_X', 'LEye_X', 'REar_X', 'LEar_X', 'LBigToe_X', 'LSmallToe_X', 'LHeel_X', 'RBigToe_X', 'RSmallToe_X', 'RHeel_X', 'Nose_Y', 'Neck_Y', 'RShoulder_Y', 'RElbow_Y', 'RWrist_Y', 'LShoulder_Y', 'LElbow_Y', 'LWrist_Y', 'MidHip_Y', 'RHip_Y', 'RKnee_Y', 'RAnkle_Y', 'LHip_Y', 'LKnee_Y', 'LAnkle_Y', 'REye_Y', 'LEye_Y', 'REar_Y', 'LEar_Y', 'LBigToe_Y', 'LSmallToe_Y', 'LHeel_Y', 'RBigToe_Y', 'RSmallToe_Y', 'RHeel_Y',]
    test_x.columns = ['Nose_X', 'Neck_X', 'RShoulder_X', 'RElbow_X', 'RWrist_X', 'LShoulder_X', 'LElbow_X', 'LWrist_X', 'MidHip_X', 'RHip_X', 'RKnee_X', 'RAnkle_X', 'LHip_X', 'LKnee_X', 'LAnkle_X', 'REye_X', 'LEye_X', 'REar_X', 'LEar_X', 'LBigToe_X', 'LSmallToe_X', 'LHeel_X', 'RBigToe_X', 'RSmallToe_X', 'RHeel_X', 'Nose_Y', 'Neck_Y', 'RShoulder_Y', 'RElbow_Y', 'RWrist_Y', 'LShoulder_Y', 'LElbow_Y', 'LWrist_Y', 'MidHip_Y', 'RHip_Y', 'RKnee_Y', 'RAnkle_Y', 'LHip_Y', 'LKnee_Y', 'LAnkle_Y', 'REye_Y', 'LEye_Y', 'REar_Y', 'LEar_Y', 'LBigToe_Y', 'LSmallToe_Y', 'LHeel_Y', 'RBigToe_Y', 'RSmallToe_Y', 'RHeel_Y',]

    train_x = train_x.drop(columns=['REye_Y', 'LEye_Y', 'REye_X', 'LEye_X','Nose_X', 'Nose_Y', 'LBigToe_X', 'LBigToe_Y', 'RBigToe_X', 'RBigToe_Y', 'LSmallToe_X', 'LSmallToe_Y', 'RSmallToe_X', 'RSmallToe_Y'])
    test_x = test_x.drop(columns=['REye_Y', 'LEye_Y', 'REye_X', 'LEye_X','Nose_X', 'Nose_Y', 'LBigToe_X', 'LBigToe_Y', 'RBigToe_X', 'RBigToe_Y', 'LSmallToe_X', 'LSmallToe_Y', 'RSmallToe_X', 'RSmallToe_Y'])
    
    return train_x, train_y, test_x, test_y