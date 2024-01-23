from demo_util import NextChatInference
import sys, os

model_path, vit_path = sys.argv[1], sys.argv[2]

model = NextChatInference(model_path, vit_path, 576)

# Please follow the example here
# input = {"text": "What is the possible relationship between the two people? Please include object locations.", "image": "./COCO_val2014_000000222628.jpg"}
# response, boxes, masks, ret_img = model(input)
# print(response)
# if ret_img is not None:
#     ret_img.save("demo.png")

import IPython
IPython.embed()