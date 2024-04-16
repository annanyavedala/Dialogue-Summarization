from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

def make_prompt(example_full, example_to_summ):
    prompt =''
    for idx in example_full:
        dialogue= dataset['test'][idx]['dialogue']
        summary= dataset['test'][idx]['summary']

        prompt += f"""
        Dialogue: 
        {dialogue}
        What was going on? 
        {summary}
        """

    dialogue= dataset['test'][example_to_summ]['dialogue']
    summary= dataset['test'][example_to_summ]['summary']

    prompt+= f"""
    Dialogue: 
        {dialogue}
        What was going on? 
    """
    return prompt
        


hf_dataset_name = "knkarthick/dialogsum"
dataset= load_dataset(hf_dataset_name)

example_indices=[40]
dash_line='-'*100

model_name= "google/flan-t5-base"
model= AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer= AutoTokenizer.from_pretrained(model_name, use_fast=True)

setence ="What time is it, Tom?"

#Zero shot prompting
for i, idx in enumerate(example_indices):
    dialogue= dataset['test'][idx]['dialogue']
    summary= dataset['test'][idx]['summary']

    prompt = """ Summarize the following conversation.
    {dialogue}
    Summary:
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    output= tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True
        )

    
    print(dash_line)
    print(dash_line)
    print('ZERO SHOT')
    print(dash_line)
    print(dash_line)
    print('Basline:\n' + summary)
    print(dash_line)
    print('Model outpiut\n'+ output)



#One shot inference
example_full= [200]
example_to_summ=40

one_shot_prompt =make_prompt(example_full, example_to_summ)


summary= dataset['test'][example_to_summ]['summary']
inputs = tokenizer(one_shot_prompt, return_tensors="pt")
output= tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True
)

print(dash_line)
print(dash_line)
print('ONE SHOT')
print(dash_line)
print(dash_line)
print('Basline:\n' + summary)
print(dash_line)
print('Model outpiut\n'+ output)

#Few shot inference

example_full= [200, 150, 60]
example_to_summ=40

few_shot_prompt = make_prompt(example_full, example_to_summ)


summary= dataset['test'][example_to_summ]['summary']
inputs = tokenizer(one_shot_prompt, return_tensors="pt")
output= tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True
)

print(dash_line)
print(dash_line)
print('FEW SHOT')
print(dash_line)
print(dash_line)
print('Basline:\n' + summary)
print(dash_line)
print('Model output\n'+ output)










