import torch
from torch import nn
import math
from torchvision import models
from transformers import BertConfig, BertModel, BertTokenizer
import pdb

class TextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        config = BertConfig(num_attention_heads=args.num_attention_heads, num_hidden_layers=args.num_hidden_layers)
        self.main = BertModel(config)
        bert_model_state_dict = BertModel.from_pretrained('bert-base-uncased').state_dict()
        embedding_weights = {x:bert_model_state_dict[x] for x in bert_model_state_dict if 'embedding' in x}
        self.main.load_state_dict(embedding_weights, strict=False)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_outputs = self.main(input_ids, attention_mask, token_type_ids, output_attentions=True)
        cls_outputs = bert_outputs[0][:, 0]
        return cls_outputs, bert_outputs[2]


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        # self.main = models.resnext101_32x8d(pretrained=True)
        self.main = models.resnet50(pretrained=True)
        num_feat = self.main.fc.in_features
        self.main.fc = nn.Linear(num_feat, 768)

    def forward(self, img):
        return self.main(img)


if __name__ == '__main__':
    import pdb
    from types import SimpleNamespace

    device = 'cuda'

    txt = [
        "Hello, I'm a single sentence!", 
        'Yo, I like eat Pineapple!',
        'This is my house.',
        'an apple on the desk is deliciouse.'
    ]
    bs = len(txt)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    args = SimpleNamespace(
        num_attention_heads=2, 
        num_hidden_layers=2,
    )

    txt_encoder = TextEncoder(args=args).to(device)
    img_encoder = ImageEncoder(args=args).to(device)

    txt_encoder = nn.DataParallel(txt_encoder)
    img_encoder = nn.DataParallel(img_encoder)

    text_inputs = tokenizer(txt, truncation=True, padding=True, return_tensors="pt").to(device)
    text_outputs, attentions = txt_encoder(**text_inputs)
    print(text_outputs.shape)
    print(text_outputs.mean(), text_outputs.std())

    image_inputs = torch.randn(bs, 3, 224, 224).to(device)
    image_outputs = img_encoder(image_inputs)
    print(image_outputs.shape)
    print(image_outputs.mean(), image_outputs.std())

    # pdb.set_trace()
