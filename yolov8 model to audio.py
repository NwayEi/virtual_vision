## pip install roboflow
# pip install ultralytics
##pyenv virtualenv 3.10.6 virtualvision
##pyenv local virtualvision


from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Use the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

boxes=results[0].boxes
box= boxes[0]
# print (f'result : {boxes.xyxy}')
# print (f'data : {boxes.data}')
# print (f'conf : {boxes.conf}')
# print (f'conf : {boxes.cls}')
# print (f'result : {box.xyxy}')
# print (f'data : {box.data}')
# print (f'conf : {box.conf}')

model_result_list = (boxes.cls).numpy()

model_names = (model.names)

result = {i: model_names[model_result_list[i]] for i in range(len(model_result_list))}
print(result)

from collections import Counter
counts = Counter(result.values())
result_dict = dict(counts)
print(result_dict)

text = ''
for key, value in result_dict.items():
    text += f'{value}{key}\n'

print(text)


from gtts import gTTS

tts = gTTS(text,lang='en', slow=False)
tts.save('text.mp3')















# import ultralytics
# ultralytics.checks()


#   system
#   3.10.6
#   3.10.6/envs/lewagon
#   3.10.6/envs/taxifare-env
# * lewagon (set by /Users/thm/.pyenv/version)
#   taxifare-env
