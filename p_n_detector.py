from transformers import pipeline

classifier = pipeline("sentiment-analysis",
                      "blanchefort/rubert-base-cased-sentiment")


file_path = "text_.txt"


# Чтение текстового файла
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        return text


text = read_text_file(file_path)



sentences = text.split('.')



for sentence in sentences:
    
    dict_list = classifier(sentence)  
    print(sentence)
    print(dict_list)