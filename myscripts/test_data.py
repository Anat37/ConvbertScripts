import torch
from transformers import Trainer, TrainingArguments
from transformers import AlbertConfig, AlbertTokenizer, AlbertForPreTraining, ConvbertForPreTraining
from dataset import SOPDataset, MyTrainer, collate_batch


model_dir = 'E:/ConvbertData/convbert/model_dir'
#model_dir = 'D:/ConvbertData/albert_model_dir'


tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
config = AlbertConfig(hidden_size=768, num_attention_heads=12, intermediate_size=3072, attention_probs_dropout_prob=0, num_hidden_groups=1, num_hidden_layers=12)
#config.save_pretrained(model_dir)
#model = ConvbertForPreTraining(config)
#model = AlbertForPreTraining(config)
model = AlbertForPreTraining.from_pretrained('albert-base-v1')
#model = ConvbertForPreTraining.from_pretrained(model_dir)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train_dataset = SOPDataset(directory='E:/ConvbertData/text_data/cache', batch_size=11, tokenizer=tokenizer, mlm_probability=0.15)

#seg1 = tokenizer.tokenize("Planar graph. In graph theory,") # a planar graph is a graph that can be embedded in the plane, i.e.") # , it can be drawn on the plane in such a way that its edges intersect only at their endpoints. In other words, it can be drawn in such a way that no edges cross each other. Such a drawing is called a plane graph or planar embedding of the graph.")
#seg2 = tokenizer.tokenize("A plane graph can be defined as a planar graph with a mapping from every node to a point on a plane") # , and from every edge to a plane curve on that plane, such that the extreme points of each curve are the points mapped from its end nodes, and all curves are disjoint except on their extreme points. Every graph that can be drawn on a plane can be drawn on the sphere as well, and vice versa, by means of stereographic projection. Plane graphs can be encoded by combinatorial maps. The equivalence class of topologically equivalent drawings on the sphere is called a planar map.")
#seg1 = tokenizer.tokenize("It is John.")
#seg2 = tokenizer.tokenize("I love him.")
#input_ids = tokenizer(seg2, seg1, is_split_into_words=True, truncation=True)
#input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1

#outputs = model(input_ids, attention_mask=torch.ones(8).unsqueeze(0), token_type_ids=torch.LongTensor([0]*8).unsqueeze(0), labels=-100*torch.ones(8).unsqueeze(0)) # sentence_order_label
#outputs = model(torch.LongTensor(input_ids["input_ids"]).unsqueeze(0), attention_mask=torch.LongTensor(input_ids["attention_mask"]).unsqueeze(0), token_type_ids=torch.LongTensor(input_ids["token_type_ids"]).unsqueeze(0), labels=torch.LongTensor([[-100]*13]), sentence_order_label=torch.LongTensor([[1]]))
#batch = collate_batch([input_ids], tokenizer, 0.250).data
#batch = next(iter(train_dataset))
#print(batch)
#print(input_ids)
#input_ids['input_ids'][2] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
#print(input_ids)
#outputs = model(torch.cuda.LongTensor([input_ids['input_ids']]), token_type_ids=torch.cuda.LongTensor([input_ids['token_type_ids']])) #, output_hidden_states=True, output_attentions=True)
#tokens_id = list(outputs[0][0].argmax(-1).cpu().numpy())
#print(tokens_id)
#print(tokenizer.convert_ids_to_tokens(tokens_id))
#model.train(False)

import time

for batch in iter(train_dataset):
    #outputs = model(torch.cuda.LongTensor([input_ids['input_ids']]), token_type_ids=torch.cuda.LongTensor([input_ids['token_type_ids']]))
    start = time.time()
    model(**batch.data)
    print(time.time() - start)
