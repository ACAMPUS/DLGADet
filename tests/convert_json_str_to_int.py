import json

js_dict=json.load(open('D:/dataset/data/new_dataset/annotations/mytt100_origion/test.json'))
list=js_dict["annotations"]
for l in list:
    l["image_id"]=int(l["image_id"])

