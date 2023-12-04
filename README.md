# NExT-Chat
NExT-Chat: An LMM for Chat, Detection and Segmentation

## Easy Run
I current make an easy run. The inference joint with the model can be now found at [Modelscope](https://www.modelscope.cn/models/ZhangAo6/nextchat).

Please install `pip install modelscope`.
Then run:
```python
from modelscope import pipeline
pipe = pipeline('my-nextchat-task', 'ZhangAo6/nextchat', model_size="7b") # 7b model takes around 21G GPU mem, 13b takes around 35G GPU mem
response, ret_image = pipe({"text": "xxxx?", "image": "xxx.jpg"})
# response: the text answer
# ret_image: image annotated with boxes and masks
```

## Acknowledgement
Thanks to [Shikra](https://github.com/shikras/shikra), [LLaVA](https://github.com/haotian-liu/LLaVA), [CogVLM](https://github.com/THUDM/CogVLM) for their excellent codes.

Our bibtex:
```bibtex
@misc{zhang2023nextchat,
      title={NExT-Chat: An LMM for Chat, Detection and Segmentation},
      author={Ao Zhang and Wei Ji and Tat-Seng Chua},
      year={2023},
      eprint={2311.04498},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```