# NExT-Chat
NExT-Chat: An LMM for Chat, Detection and Segmentation

## Easy Run
I current make an easy run. The inference code will be released soon (this week).

Please install `pip install modelscope`.
Then run:
```python
from modelscope import pipeline
pipe = pipeline('my-nextchat-task', 'ZhangAo6/nextchat', model_size="7b") # 7b model takes around 21G GPU mem, 13b takes around 35G GPU mem
response, ret_image = pipe({"text": "xxxx?", "image": "xxx.jpg"})
# response: the text answer
# ret_image: image annotated with boxes and masks
```