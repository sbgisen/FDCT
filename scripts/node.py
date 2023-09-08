#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

# Copyright (c) 2023 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cv_bridge
import message_filters
import Model
import numpy as np
import rospkg
import rospy
import torch
from sensor_msgs.msg import Image


class DepthCompletion(object):

    def __init__(self) -> None:
        pkg_path = rospkg.RosPack().get_path('fdct')
        model_path = rospy.get_param('~model_path', pkg_path + '/config/ClearPose.tar')
        self.model = Model.FDCT()
        ckpt = torch.load(model_path, map_location=torch.device('cuda:0'))
        print(ckpt.keys())
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.pub_img = rospy.Publisher('~output', Image, queue_size=1)
        self.__cv_bridge = cv_bridge.CvBridge()
        rgb_sub = message_filters.Subscriber('~rgb', Image)
        depth_sub = message_filters.Subscriber('~depth', Image)
        self.__sync = message_filters.TimeSynchronizer([rgb_sub, depth_sub], 10)
        self.__sync.registerCallback(self.callback)

    @torch.no_grad()
    def callback(self, rgb: Image, depth: Image) -> None:
        rgb_img = self.__cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        depth_img = self.__cv_bridge.imgmsg_to_cv2(depth, '32FC1')
        rgb_img = rgb_img[np.newaxis, :, :, :]
        depth_img = depth_img[np.newaxis, :, :]
        torch_img = torch.from_numpy(rgb_img).permute(0, 3, 1, 2).float()
        torch_depth = torch.from_numpy(depth_img).float()
        output = self.model(torch_img, torch_depth)
        output = output.squeeze().cpu().numpy()
        output_msg = self.__cv_bridge.cv2_to_imgmsg(output, 'passthrough')
        self.pub_img.publish(output_msg)


if __name__ == '__main__':
    rospy.init_node('depth_completion')
    DepthCompletion()
    rospy.spin()
